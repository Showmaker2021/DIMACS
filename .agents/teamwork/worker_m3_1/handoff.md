# Handoff Report — Milestone M3 (Offline Evaluation Harness & Coverage Metrics)

**Agent ID:** worker_m3_1  
**Working Directory:** `d:\Graph Coloring\.agents\teamwork\worker_m3_1`  
**Date:** 2026-10-16  
**Status:** Completed (Hard Handoff)

---

## 1. Observation

1. **Assigned Scope & Exclusive Ownership:**
   - Files owned: `eval/coverage.py`, `eval/eval_model.py`, `tests/test_eval_harness.py`.
   - Invariants: Frozen files in `data/data_v1/`, `data/data_v1_1/`, `evaluator/verify.py`, sandbox whitelist in `gate/evolved/sandbox.py`, zero network access, and zero torch import during tests.
2. **Pre-Implementation State:**
   - Baseline test execution `pytest` completed with code 0: `113 passed in 21.63s`.
   - `eval/` did not exist; `runs/eval/` was not populated.
3. **Implementation Details:**
   - Created `eval/coverage.py`: pure mathematical functions without I/O.
     - `compute_rf(dsatur_mean, rlf_mean)`: computes $\min(\bar{c}_{\text{DSATUR}}, \bar{c}_{\text{RLF}})$.
     - `beats_rf(candidate_mean_colors, rf_mean_colors, eps=0.05)`: canonical test requiring $\bar{c}_h \le R(F) - \epsilon$; ties explicitly do not count as wins; handles non-finite/None candidates.
     - `pass_at_k(n, c, k)`: unbiased Chen et al. estimator computed via numerically stable product $\prod_{j=0}^{k-1} (1 - c / (n - j))$ clamped to $[0.0, 1.0]$. Validates $k \le n$, positive integer bounds, and $0 \le c \le n$.
     - `coverage_at_k(card_win_counts, n_samples, k)` and `empirical_coverage_at_k(card_candidate_wins)`: returns expected or empirical card coverage fraction; handles zero-card edge case gracefully ($0.0$).
     - `best_of_k(card_candidate_colors, rf_colors, k, failed_savings_fallback=0.0)`: selects best candidate colors and savings vs $R(F)$, handling all-candidate failures on cards without crashing.
   - Created `eval/eval_model.py`: offline evaluation harness.
     - CLI flags: `--backend {mock,hf,vllm,api}`, `--split`, `--k`, `--n-samples`, `--temperature`, `--n-graphs`, `--run-id`, `--seed`, `--force-unlock`, `--model-name`, `--eps`, `--output-dir`, `--cache-dir`, `--lock-dir`, `--workers`, `--prompt-version`, `--families`, `--per-graph-timeout`, `--overall-timeout`.
     - `build_eval_prompt(card, prompt_version="v1")`: canonical prompt using vectorized API contract `priority(adj, colors, sat, deg, uncolored_deg)`.
     - `MockBackend`: deterministic fake generator producing DSATUR-like (sample 0), RLF-like (sample 1), broken AST with forbidden import (sample 2), and slow timeout loop (sample 3).
     - Lazy backend stubs for `HFBackend` (imports `torch`/`transformers` lazily; fails with clear `ImportError`), `VLLMBackend` (imports `vllm` lazily), and `APIBackend` (validates `api_key_env` presence; forbids live network calls).
     - Sandboxing via `gate.evolved.sandbox.evaluate_candidate_safe` coordinated using `ThreadPoolExecutor` (Windows-safe child process isolation with RSS memory monitoring and hard wall-clock timeouts).
     - Behavioral probe fingerprinting: evaluates candidates and baselines on probe seeds 9000-9009 using the exact sandbox `behavior_hash` format (`-`-delimited SHA256 of coloring bytes on first 3 probe graphs).
     - Disk cache under `runs/eval/_cache/` keyed by `sha256(code_hash + "_" + graph_set_id + "_" + config_version)`.
     - Test-lock via `TestLockManager`: splits starting with `"test"` create a `.done` lock file containing `(model_name, split, config_hash)`. Repeated evaluation without `--force-unlock` raises `RuntimeError`. Overriding with `--force-unlock` records `"force_unlock_used": true`. Non-test splits (`val`, `train`, `probe`) are never locked.
     - Outputs created under `runs/eval/{run_id}/`: `per_card.jsonl`, `summary.json`, and `config.json`. Overwriting existing `run_id` raises `FileExistsError`.
   - Created `tests/test_eval_harness.py`: 33 dedicated tests covering all pure metrics, prompt formatting, mock backend archetypes, lazy backend imports, end-to-end evaluation on tiny split, disk cache hits, test-lock refusal and force-unlock, run-id protection, and environment invariants.
4. **Verification Output:**
   - Dedicated test suite `pytest tests/test_eval_harness.py -v`: `33 passed in 14.68s`.
   - Full repository test suite `pytest`: `169 passed in 43.15s`.
   - Frozen files test `pytest tests/test_frozen_files.py`: `5 passed in 0.29s`.

---

## 2. Logic Chain

1. **Pure Metrics Guarantees (eval/coverage.py):**
   - Observation: Requirement R3 mandates that ties do not count as wins and requires a configurable margin $\epsilon$.
   - Deduction: In `beats_rf`, `(rf_mean_colors - candidate_mean_colors) >= eps` is evaluated only after verifying candidate finiteness and non-None status. Exact equality (`rf_mean == candidate_mean`) yields difference $0.0 < 0.05$, correctly returning `False`.
   - Observation: Combinatorial expressions $\binom{n-c}{k} / \binom{n}{k}$ overflow standard integer types for $n \approx 1000$.
   - Deduction: Telescoping the ratio as $\prod_{j=0}^{k-1} (1 - c / (n - j))$ computes purely in floating-point operations where each factor is in $[0, 1]$, guaranteeing numerical stability for arbitrary $n \ge k$. Clamping to $[0.0, 1.0]$ prevents IEEE 754 precision drift.
2. **Windows Concurrency & Sandbox Safety (eval/eval_model.py):**
   - Observation: On Windows, `multiprocessing` uses `spawn` contexts. Creating nested ProcessPoolExecutors inside worker processes risks pipe deadlocks and zombie processes.
   - Deduction: `eval/eval_model.py` uses `ThreadPoolExecutor` at the outer layer while delegating process spawning and resource killing to `gate.evolved.sandbox.evaluate_candidate_safe`. This cleanly enforces single-thread BLAS limits (`OMP_NUM_THREADS=1`), 1.5 GB memory limits, and subprocess hard timeouts.
3. **Deterministic Cache & Lock Guarantees:**
   - Observation: The cache key combines candidate code hash, graph set identifier (family ID, seed range, graph count), and sandbox version string.
   - Deduction: Re-running evaluation with identical configuration accesses cached results, achieving $100\%$ cache hit rate and $0$ evaluator invocations, as verified by `test_disk_cache_hits_on_rerun`.
   - Observation: Data snooping on test splits must be prevented.
   - Deduction: Test splits create `.done` files inside `runs/eval/_locks/` (satisfying `test_frozen_files.py` constraints). Subsequent runs detect this lock and raise `RuntimeError` unless `--force-unlock` is explicitly specified and logged.

---

## 3. Caveats

- **Optional GPU Dependencies:** `torch`, `transformers`, and `vllm` are not installed in the current Windows test environment; `HFBackend` and `VLLMBackend` remain lazy stubs that raise informative `ImportError` messages until deployment on GPU clusters.
- **Probe Graph Evaluation:** Behavioral fingerprinting evaluates probe seeds 9000-9009 (10 graphs). If a candidate fails on the split graphs, probe graph evaluation is bypassed, and fingerprint is recorded as `None`.
- No other caveats.

---

## 4. Conclusion

Milestone M3 is fully completed and verified. `eval/coverage.py`, `eval/eval_model.py`, and `tests/test_eval_harness.py` satisfy all specifications in Section R3 of `ORIGINAL_REQUEST.md` and `PROJECT.md`. All 33 new tests and all 169 tests across the repository pass green.

---

## 5. Verification Method

To independently verify this milestone:

1. **Run dedicated eval harness test suite:**
   ```powershell
   pytest tests/test_eval_harness.py -v
   ```
   *Expected:* 33 passed, 0 failures, 0 warnings.

2. **Run full regression test suite:**
   ```powershell
   pytest
   ```
   *Expected:* 169 passed across all 14 test modules.

3. **Verify frozen files and integrity:**
   ```powershell
   pytest tests/test_frozen_files.py
   ```
   *Expected:* 5 passed (manifest matches frozen files, whitelist intact, no rogue lock files outside `runs/`).

4. **Execute CLI mock evaluation:**
   ```powershell
   python -m eval.eval_model --backend mock --split val --families gnp_d0.10_n50-100 --n-graphs 2 --n-samples 2 --k 1 --run-id smoke_cli_test --output-dir runs/test_cli
   ```
   *Expected:* Output directory created with `per_card.jsonl`, `summary.json`, `config.json`, exit code 0. Clean up with `Remove-Item -Recurse -Force runs/test_cli`.
