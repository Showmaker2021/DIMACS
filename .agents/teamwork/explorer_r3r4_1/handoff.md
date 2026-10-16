# Handoff Report: R3 (Evaluation Harness & Coverage Metrics) and R4 (GRPO Reward Engine)

**Agent:** `explorer_r3r4_1`  
**Working Directory:** `d:\Graph Coloring\.agents\teamwork\explorer_r3r4_1`  
**Handoff Type:** Hard (Task complete)  
**Recipient:** Orchestrator (`553217b8-f1c7-4ae8-b2fd-86f4b1a6dc9b`) / Builder Subagent

---

## 1. Observation
1. **Existing Test Suite Baseline:**
   - Ran `pytest -q` on repo root:
     `52 passed in 8.36s` (Output confirmed via task log `task-47`).
2. **Current Codebase Structure:**
   - Evaluator framework in `evaluator/coloring.py`:
     - Line 16-30: `dsatur_priority` implementing `sat * (n + 1) + uncolored_deg`.
     - Line 32-74: `rlf_priority` implementing vectorized RLF logic.
     - Line 77-137: `greedy_color(adj, priority_fn)` enforcing read-only views on `adj`, `colors`, `sat`, `deg`, `uncolored_deg`.
   - Independent verifier in `evaluator/verify.py`:
     - Line 13-64: `verify(adj, colors)` checking dimensions, types, non-negativity, and proper vertex coloring.
     - Line 66-78: `count_colors(colors)` returning unique color counts.
   - 4-layer sandbox in `gate/evolved/sandbox.py`:
     - Line 104-183: `SecurityASTVisitor` enforcing AST constraints (5 positional args, no default args, no `_` prefixes, no imports, NumPy whitelist).
     - Line 235-294: `check_statelessness_and_purity(code_obj, adj)`.
     - Line 474-535: `evaluate_candidate_safe` using `mp.get_context("spawn")`, scrubbing `*_API_KEY`, setting `OMP_NUM_THREADS=1`, monitoring RSS <= 1.5 GB.
   - Pilot lock pattern in `scripts/pilot_test.py`:
     - Lines 41-52: Checking `{lock_dir}/{family_id}.done` and blocking duplicate test evaluations to prevent snooping.
3. **Absence of Target Modules:**
   - Target directories `eval/` and `train/` do not yet exist in the repository.
   - Requirements in `ORIGINAL_REQUEST.md` (R3, R4) and `PLAN_PHASE6_ONWARD.md` (Sections 9.2, 11.1, 14, 14.3) outline exact mathematical and architectural requirements for:
     - `eval/coverage.py`
     - `eval/eval_model.py`
     - `train/reward.py`

---

## 2. Logic Chain
1. **Decoupling Pure Metrics from Sandboxed Execution:**
   - From Observation 2 and `ORIGINAL_REQUEST.md` line 56, `eval/coverage.py` must be pure functions with zero I/O and zero dependencies on the sandbox.
   - Following Observation 2's definition of $R(F)$, `beats_rf` must strictly check $\bar{c}_h \le R(F) - \epsilon$ (where $\epsilon = 0.05$), with ties explicitly not counting as wins.
   - For `pass_at_k(n, c, k)`, Chen et al.'s formula $1 - \binom{n-c}{k}/\binom{n}{k}$ is implemented using stable product $\prod_{j=0}^{k-1}(1 - c/(n-j))$ or log-gamma to avoid combination overflow when $n$ is large.
   - Edge cases are mapped directly: $k > n$ raises `ValueError`, $c=0 \implies 0.0$, $n - c < k \implies 1.0$, $k=1 \implies c/n$.
2. **Evaluation Harness Concurrency on Windows:**
   - From Observation 2, `gate/evolved/sandbox.py` already spawns a dedicated worker process for each candidate evaluation.
   - In Windows, spawning worker processes inside a `multiprocessing.Pool` or `ProcessPoolExecutor` can cause nested spawn issues and pickling overhead.
   - Using `concurrent.futures.ThreadPoolExecutor` in `eval/eval_model.py` to dispatch `evaluate_candidate_safe` invocations allows parallel evaluation where each candidate is isolated in its own OS process, perfectly safe for Windows.
   - Caching using SHA256 of `(code_hash, graph_set_id, config_version)` avoids redundant sandbox evaluations.
   - The test-lock mechanism mirrors `scripts/pilot_test.py` (Observation 2): checking and writing a `.done` lock file for any split starting with `test*`, raising an error on repeats unless `--force-unlock` is passed.
3. **GRPO Reward Engine Decoupling & Boundedness:**
   - From `ORIGINAL_REQUEST.md` line 80-94 and Section 14.3 of the Plan, `train/reward.py` consumes precomputed `CandidateResult` without calling the sandbox.
   - All failed candidates (`status != "ok"`) receive exactly $-1.0$ and are excluded from diversity pairs.
   - Savings reward $r_{sav} = \text{mean}_g \text{clip}((c_{base}(g) - c_h(g)) / \max(1, c_{base}(g) - c_{ref}(g)), -1, 1)$ ensures baseline reproduction yields $0.0$, improvements yield $(0, 1]$, and regressions yield $[-1, 0)$.
   - Runtime penalty $r_{time} = \text{clip}(t_h / t_{budget} - 1, 0, 1)$ penalizes only when $t_h > t_{budget}$.
   - Diversity reward $r_{div}$ uses mean pairwise Kendall-tau distance on probe orderings, applied only when $r_{pre} > \text{median}(r_{pre})$.
   - Advantage normalization uses population std (`ddof=0`) with zero-variance protection: when all rewards are identical (e.g. all fail), advantages are set to $0.0$ exactly.
   - Boundedness proof: reward is analytically guaranteed to lie in $[-1.0 - b, 1.0 + a + d]$, or $[-1.1, 1.0]$ under defaults ($a=0.0, b=0.1, d=0.0$).

---

## 3. Caveats
- No production source code in `eval/` or `train/` was created or modified during this turn, adhering strictly to the explorer read-only constraint.
- The default `time_factor = 2.0` for `t_budget` is configured based on Section 14.3; actual hardware calibration against vectorized DSATUR runtime should be verified in runtime benchmarks.
- In `r_div`, if only 1 candidate succeeds in a group, pairwise distance cannot be computed, so $r_{div}$ naturally defaults to $0.0$.

---

## 4. Conclusion
The technical architecture, exact formulas, data schemas, edge cases, and CLI specifications for R3 (`eval/coverage.py`, `eval/eval_model.py`) and R4 (`train/reward.py`) are fully resolved and documented in `d:\Graph Coloring\.agents\teamwork\explorer_r3r4_1\report.md`. The design is fully actionable for immediate implementation by the builder subagent.

---

## 5. Verification Method
1. **Inspection:**
   - Read `d:\Graph Coloring\.agents\teamwork\explorer_r3r4_1\report.md` for complete design specifications.
2. **Pre-Implementation Test Invariant:**
   - Run `pytest` to confirm all 52 existing tests remain passing.
3. **Post-Implementation Verification (by Builder):**
   - Run `pytest tests/test_eval_harness.py tests/test_reward.py` once implemented.
   - Execute mock evaluation:
     `python eval/eval_model.py --backend mock --split val --k 1 --n-samples 2 --run-id smoke_test --seed 42`
   - Verify outputs in `runs/eval/smoke_test/per_card.jsonl` and `runs/eval/smoke_test/summary.json`.
