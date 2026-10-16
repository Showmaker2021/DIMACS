# Handoff Report: Milestone M4 (GRPO Reward Engine)

**Agent ID:** worker_m4_1  
**Working Directory:** `.agents/teamwork/worker_m4_1`  
**Date:** 2026-10-16T04:05:00Z  
**Type:** Hard Handoff (Task Complete)

---

## 1. Observation

### 1.1 Baseline State
Prior to code modification, baseline testing ran against the existing 12 test suites:
Command: `pytest`
```
tests\test_adversarial_m1.py .....                                       [  4%]
tests\test_challenger_m1_it2.py .............                            [ 15%]
tests\test_dataset.py ......                                             [ 21%]
tests\test_evaluator.py .......                                          [ 27%]
tests\test_frozen_files.py .....                                         [ 31%]
tests\test_loop_telemetry.py ......                                      [ 37%]
tests\test_provisional_labels.py .                                       [ 38%]
tests\test_resume.py .                                                   [ 38%]
tests\test_sandbox.py .............................                      [ 64%]
tests\test_seeds.py ...................                                  [ 81%]
tests\test_splits.py .............                                       [ 92%]
tests\test_verify.py ........                                            [100%]
============================ 113 passed in 17.72s =============================
```

### 1.2 Files Implemented
1. `train/reward.py` (483 lines, 17,302 bytes):
   - Defined `CandidateResult`:
     ```python
     @dataclass
     class CandidateResult:
         status: str
         colors_per_graph: List[int] = field(default_factory=list)
         runtime_per_graph: List[float] = field(default_factory=list)
         probe_orderings: Optional[List[Any]] = None
         c_base_per_graph: Optional[List[int]] = None
         c_ref_per_graph: Optional[List[int]] = None
         t_dsatur_per_graph: Optional[List[float]] = None
         color_classes_per_graph: Optional[List[List[int]]] = None
         code: Optional[str] = None
     ```
   - Defined `RewardConfig`:
     ```python
     @dataclass
     class RewardConfig:
         a: float = 0.0                 # Weight for r_tb (tie-breaking, default 0.0 OFF)
         b: float = 0.1                 # Weight for r_time (runtime penalty, default 0.1)
         d: float = 0.0                 # Weight for r_div (behavioral diversity, default 0.0)
         time_factor: float = 3.0       # Multiplier: t_budget = time_factor * time_dsatur
         failure_reward: float = -1.0   # Exact reward assigned to any non-ok candidate
         eps_std: float = 1e-8          # Numerical stability for advantage normalization
         clip_sav_lo: float = -1.0      # Lower bound for r_sav clipping
         clip_sav_hi: float = 1.0       # Upper bound for r_sav clipping
         fallback_ref_offset: int = 1   # Fallback when c_ref is None: c_base(g) - offset
     ```
   - Defined `RewardBreakdown` tracking `status`, `r_sav`, `r_time`, `r_tb`, `r_pre`, `r_div`, `reward`, `is_eligible_diversity`.
   - Core functions:
     * `get_reward_bounds(config)`
     * `kendall_tau_distance(ord1, ord2)`
     * `pairwise_probe_distance(probe_list_a, probe_list_b)`
     * `compute_r_sav(colors, c_base, c_ref, config)`
     * `compute_r_time(runtimes, t_dsatur, config)`
     * `compute_r_tb(color_classes_per_graph)`
     * `compute_group_rewards_detailed(candidates, config, c_base_per_graph, ...)`
     * `compute_group_rewards(...)`
     * `compute_advantages(rewards, eps_std=1e-8)` (population std `ddof=0`, zero-variance safeguard)
2. `train/__init__.py`: package initialization.
3. `tests/test_reward.py` (325 lines): 23 test functions organized in 6 test classes.

### 1.3 Verification Results
1. Targeted unit test run: `pytest tests/test_reward.py -v`
   Result: `23 passed in 4.38s`
2. Full test suite run: `pytest`
   Result: `136 passed in 23.36s` (113 existing + 23 new)
3. Frozen files check: `pytest tests/test_frozen_files.py`
   Result: `5 passed in 0.29s`
4. Syntax / byte-compilation: `python -m py_compile train/reward.py tests/test_reward.py`
   Result: exit code 0, clean.

---

## 2. Logic Chain

1. **Decoupled Architecture:** As mandated in Section R4 and Dispatch, `train/reward.py` contains zero imports or calls to sandbox execution or graph generators. It purely evaluates precomputed candidate metrics (`CandidateResult`) and group context.
2. **Component Formulation:**
   - $r_{sav}$: For each graph $g$, calculates $\text{clip}\left(\frac{c_{base}(g) - c_h(g)}{\max(1, c_{base}(g) - c_{ref}(g))}, -1, 1\right)$. When $c_{ref}$ is None, fallback uses $c_{base}(g) - 1$, setting denominator to 1.
   - $r_{time}$: Evaluates mean candidate runtime $t_h$ against budget $t_{budget} = \text{time\_factor} \times t_{dsatur}$. Evaluates $\text{clip}(t_h / t_{budget} - 1, 0, 1)$. Zero DSATUR runtime or under budget yields 0 penalty.
   - $r_{tb}$: Optional Herfindahl index of color classes when $a > 0.0$ (default $a=0.0$).
   - $r_{pre} = r_{sav} + a \cdot r_{tb} - b \cdot r_time$. For any non-ok status, $r_{pre} = -1.0$.
3. **Median-Gated Diversity & Peer Exclusion:**
   - Group median of $r_{pre}$ is computed across all group candidates.
   - Diversity $r_{div}$ is gated strictly to candidates where $status == \text{"ok"}$ AND $r_{pre} > \text{median}(r_{pre})$.
   - Diversity peers are filtered strictly to valid $status == \text{"ok"}$ candidates, completely excluding failed candidates.
   - When $d=0.0$, diversity bonus is 0.
   - When only 1 candidate succeeds or all fail, peer set is empty, yielding $r_{div} = 0.0$ without exceptions.
4. **Kendall-Tau Distance:**
   - Normalized fraction of discordant pairs out of $\binom{N}{2}$ total pairs.
   - Proved identical permutations $\to 0.0$ and reversed $\to 1.0$.
   - Handles permutations of arbitrary vertex labels, empty/single-element inputs safely.
5. **Advantage Normalization:**
   - Uses population standard deviation ($\sigma$, `ddof=0`).
   - If $\sigma \le \epsilon_{std}$ or $\sigma \approx 0.0$, all advantages return exactly 0.0 without NaN or inf.
6. **Mathematical Bounds:**
   - Lower bound: $\min(-1.0, \text{clip\_sav\_lo} - b) = -1.0 - b$.
   - Upper bound: $\max(-1.0, \text{clip\_sav\_hi} + a + d) = 1.0 + a + d$.
   - Documented and verified via randomized property-based testing over 500 configurations with fixed seed.

---

## 3. Caveats

- `train/reward.py` expects candidate orderings to represent permutations of comparable vertex sets when computing Kendall-tau distance. If non-permutation float scores are passed, it falls back to rank correlation via `scipy.stats.kendalltau`.
- No RL training loop (TRL/GRPO policy rollout) is implemented in this batch, as Phase 1 / Phase 6 Batch A scope is strictly limited to the reward module.

---

## 4. Conclusion

Milestone M4 (GRPO Reward Engine) is 100% complete and verified:
- `train/reward.py` and `tests/test_reward.py` satisfy every requirement from Section R4 of `ORIGINAL_REQUEST.md` and the dispatch prompt.
- All 136 tests pass green with zero regressions and zero modifications to frozen files or unauthorized paths.

---

## 5. Verification Method

To independently verify the implementation:
1. Run reward suite:
   ```bash
   pytest tests/test_reward.py -v
   ```
2. Run full regression suite:
   ```bash
   pytest
   ```
3. Run frozen file integrity guard:
   ```bash
   pytest tests/test_frozen_files.py
   ```
4. Verify Python byte-compilation:
   ```bash
   python -m py_compile train/reward.py tests/test_reward.py
   ```
