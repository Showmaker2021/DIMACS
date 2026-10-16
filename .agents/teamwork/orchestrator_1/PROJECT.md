# Project: Phase 6 / Batch A Modules for Graph Coloring

## Architecture
- `SEEDS.md`: Global seed registry with closed ranges, pairwise disjointness.
- `tests/test_frozen_files.py`: Integrity guard for frozen data/evaluator files.
- `configs/teacher_llm.yaml`: Teacher model configuration (env-var name only, zero secrets).
- `gate/evolved/loop.py`: Cost and telemetry logging per LLM call (additive only).
- `scripts/build_splits.py` & `data/data_v1_2/splits.json`: Deterministic splits of 106 existing families into 6 groups.
- `eval/coverage.py`: Pure metrics for "beats R(F)", coverage@k, best-of-k, pass@k.
- `eval/eval_model.py`: Offline evaluation harness with mock/lazy backends, caching, test-locking.
- `train/reward.py`: Pure GRPO reward and advantage engine consuming CandidateResult dataclass.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Frozen Files Guard | Manifest hash check for data_v1, data_v1_1, verify.py, locks | M1 | ORIGINAL_REQUEST §Rule 1 |
| 2 | SEEDS Registry | SEEDS.md with 13 closed, disjoint ranges + test | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Teacher LLM Config | configs/teacher_llm.yaml with api_key_env only | M1 | ORIGINAL_REQUEST §R1 |
| 4 | Loop Cost Telemetry | Add latency_s, status, tokens, cost fields to loop.py | M1 | ORIGINAL_REQUEST §R1 |
| 5 | Dataset v1.2 Splits | Partition 106 families into 6 groups with attributes/binning | M2 | ORIGINAL_REQUEST §R2 |
| 6 | Splits Builder Script | scripts/build_splits.py deterministic split generator | M2 | ORIGINAL_REQUEST §R2 |
| 7 | Splits Tests | tests/test_splits.py validating all partition rules | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Coverage Metrics | eval/coverage.py: beats R(F), coverage@k, best_of_k, pass@k | M3 | ORIGINAL_REQUEST §R3 |
| 9 | Offline Eval CLI | eval/eval_model.py with mock backend, caching, test-lock | M3 | ORIGINAL_REQUEST §R3 |
| 10 | Eval Harness Tests | tests/test_eval_harness.py testing metrics, CLI, caching | M3 | ORIGINAL_REQUEST §R3 |
| 11 | GRPO Reward Engine | train/reward.py: CandidateResult, r_sav, r_time, r_div, A_i | M4 | ORIGINAL_REQUEST §R4 |
| 12 | Reward Tests | tests/test_reward.py for formulas, clipping, diversity, bounds | M4 | ORIGINAL_REQUEST §R4 |
| 13 | Step 0 Summary & Notes | NOTES.md update with all required sections & reproducibility | M5 | ORIGINAL_REQUEST §Acceptance |
| 14 | Victory Audit | Full pytest test run (52 original + all new suites green) | M5 | ORIGINAL_REQUEST §Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | R1 Seeds & Config & Telemetry | SEEDS.md, test_seeds.py, test_frozen_files.py, teacher_llm.yaml, loop.py logging | none | DONE |
| M2 | R2 Dataset v1.2 Splits | build_splits.py, data_v1_2/splits.json, test_splits.py | M1 | IMPLEMENTED (tests pass) |
| M3 | R3 Offline Evaluation Harness | eval/coverage.py, eval/eval_model.py, test_eval_harness.py | M1, M2 | DONE |
| M4 | R4 GRPO Reward Engine | train/reward.py, test_reward.py | M1 | DONE |
| M5 | Final Integration & Audit | NOTES.md update, full pytest regression suite, mock eval run | M1, M2, M3, M4 | DONE |

## Code Layout
- `SEEDS.md`
- `configs/teacher_llm.yaml`
- `gate/evolved/loop.py`
- `scripts/build_splits.py`
- `data/data_v1_2/splits.json`
- `eval/coverage.py`
- `eval/eval_model.py`
- `train/reward.py`
- `tests/test_frozen_files.py`
- `tests/test_seeds.py`
- `tests/test_splits.py`
- `tests/test_eval_harness.py`
- `tests/test_reward.py`
- `NOTES.md`
