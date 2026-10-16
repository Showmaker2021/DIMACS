# Handoff Report: R1 & R2 Investigation

**From:** `explorer_r1r2_1`  
**To:** Orchestrator (`553217b8-f1c7-4ae8-b2fd-86f4b1a6dc9b`)  
**Date:** 2026-10-16  
**Type:** Hard Handoff (Investigation Complete)  

---

## 1. Observation
1. **Existing Seed Usages**:
   - `data/data_v1/config.json`: `train: [0, 100]`, `val: [1000, 20]`, `test: [2000, 50]`, `probe: [9000, 10]`.
   - `data/data_v1_1/config.json`: `teacher_val: [3000, 20]`, `test_pilot: [4000, 200]`.
   - `tests/test_dataset.py`: asserts pairwise disjointness of `train`, `val`, `test`, `teacher_val`, `test_pilot`, `probe`.
   - `tests/test_evaluator.py`: lines 26 (`1000+i`) and 143 (`6000+i`) use ephemeral in-memory random graphs for unit tests.
2. **LLM Config**:
   - `configs/pricing.json` defines pricing for `deepseek-chat`, `gemini-flash-lite-latest`, `claude-3-5-sonnet-20241022`, and `mock`.
   - `loop.py` lines 114-123 reads keys strictly from `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`.
3. **Telemetry in `gate/evolved/loop.py`**:
   - Single LLM call site at line 492 in `run_evolution`, calling `call_llm_with_retry`.
   - `log_entry` at lines 569-589 logs `tokens: usage`, `valid`, `rejection_reason`, but lacks top-level `prompt_tokens`, `completion_tokens`, `latency_s`, and explicit `status`.
4. **Dataset Metadata**:
   - `data/data_v1_1/families_v1_1.json` defines 106 families across 11 structure types: 55 `train_pool`, 26 `heldout_type`, 15 `heldout_size`, 10 `heldout_combo`.
   - `data/generate_dataset.py` lines 397-403 defines deterministic binning: `density_bin` via `searchsorted([0.05, 0.10, 0.20, 0.40])`, `degree_variance_bin` via `deg_cv` (<0.2, 0.2-0.5, >=0.5), and `clustering_bin` via `clustering / max(density, 1e-9)` (<1.5, 1.5-4.0, >=4.0).
   - In `heldout_combo`, all 10 families share combination `[2, 'low', 'low']`, which appears 0 times in `train_pool`.
   - In `heldout_size`, all 15 families have $n \in [320, 500]$, whereas `train_pool` has $\max(n) = 300$.

---

## 2. Logic Chain
1. *From Obs 1*: Sorting the 6 frozen ranges and 7 proposed ranges shows all interval bounds $E_i < S_{i+1}$ with positive gaps (smallest gap: $7049 \to 7100$ = 51).
   $\to$ Proposed seed ranges have zero collisions with existing ranges or code.
2. *From Obs 2*: YAML schema with fields `provider`, `model`, `base_temperature: 0.7`, `max_tokens: 1024`, `rate_limit: {requests_per_minute: 12, pacing_seconds: 5.0}`, `retry: {max_retries: 2, backoff_factor: 2.0, initial_wait_seconds: 10.0, max_wait_seconds: 60.0}`, and `api_key_env: "DEEPSEEK_API_KEY"` satisfies all prompt and safety constraints.
3. *From Obs 3*: Measuring `latency_s = time.perf_counter() - t_start` around `call_llm_with_retry` and adding top-level keys to `log_entry` logs all required cost and telemetry metrics while preserving existing fields and backward compatibility with `tests/test_resume.py`.
4. *From Obs 4*: Using probe statistics and the established thresholds in `data/generate_dataset.py` allows deterministic generation of attributes `n_range`, `density_bin`, `degree_variance_bin`, and `clustering_bin`.
5. *From Obs 4*: Assigning `heldout_combo` (10) to `test_comp`, `heldout_size` (15) to `test_extrap`, and all 4 heldout types (26) to `heldout_type`, then splitting the 55 train families into 41 `train_pool`, 7 `val`, and 7 `test_interp` guarantees:
   - All 106 families partitioned into exactly one group.
   - `scale_free` is in `train_pool` (5 families).
   - `small_world` is 100% held out from `train_pool` and `val`.
   - No held-out combination appears in `train_pool`.
   - `test_extrap` families exceed `train_pool` maximum $n$ ($320 > 300$).
   - All 6 splits have $\ge 7$ families (none $< 3$).

---

## 3. Caveats
1. `tests/test_evaluator.py:143` generates in-memory ER graphs with seeds 6000-6024. This is an ephemeral unit test and does not collide with dataset loading or storage.
2. `heldout_type` retains all 26 families of held-out structure types from v1.1 (`small_world` [7], `geometric` [7], `hybrid_geometric` [5], `queen` [7]), with `small_world` designated as the primary target type for Gate evaluation.

---

## 4. Conclusion
The implementation of R1 (`SEEDS.md`, `configs/teacher_llm.yaml`, `loop.py` telemetry) and R2 (`data/data_v1_2/splits.json`, `scripts/build_splits.py`) is fully specified, verified for mathematical consistency, and ready to be built.

---

## 5. Verification Method
1. Run test suite: `pytest` (assert 52/52 passing).
2. Range verification: execute pairwise interval overlap assertion script.
3. Split invariants: execute verification script checking `(2, 'low', 'low')` absence from `train_pool` and `test_extrap` $n \ge 320 > \max(\text{train\_pool } n) = 300$.
