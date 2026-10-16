## 2026-10-16T03:44:28Z
You are Worker subagent worker_m2_1.
Your working directory is: d:\Graph Coloring\.agents\teamwork\worker_m2_1
You MUST read:
1. d:\Graph Coloring\.agents\teamwork\ORIGINAL_REQUEST.md
2. d:\Graph Coloring\.agents\teamwork\orchestrator_1\PROJECT.md
3. d:\Graph Coloring\.agents\teamwork\explorer_r1r2_1\report.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope & Write Ownership:
You exclusively own:
- scripts/build_splits.py
- data/data_v1_2/splits.json
- tests/test_splits.py

Tasks:
1. Review explorer_r1r2_1/report.md which specifies the binning rules, attribute derivations, and family partitioning for all 106 existing families.
2. Implement scripts/build_splits.py: generates data/data_v1_2/splits.json deterministically.
   - For every family: family_id, structure_type, and attributes n_range, density_bin, degree_variance_bin, clustering_bin.
   - Document binning thresholds under a "binning" key in splits.json.
   - Groups: train_pool (must include scale_free), val, test_interp, test_extrap, test_comp, heldout_type (= small_world).
   - In test_comp, list explicitly the held-out attribute COMBINATIONS under a "heldout_combinations" key; those combinations must be absent from train_pool.
   - Each group references its seed range name from SEEDS.md (e.g. test_interp -> "test_interp", train_pool -> "train" or "rl_train_pool").
   - Confirm test_extrap families exceed train_pool maxima in n or density.
   - Do NOT touch data/data_v1/ or data/data_v1_1/.
3. Run scripts/build_splits.py to generate data/data_v1_2/splits.json.
4. Implement tests/test_splits.py covering all 6 checks from ORIGINAL_REQUEST.md line 46-53:
   - every family appears in exactly one group;
   - no held-out compositional combination appears in train_pool;
   - small_world appears neither in train_pool nor in val;
   - extrapolation families really exceed the train range in n or density;
   - every referenced seed-range name exists in SEEDS.md and ranges are consistent;
   - scale_free is in train_pool.
5. Run pytest tests/test_splits.py and full pytest (including test_frozen_files.py) to confirm all tests pass cleanly.
6. Write your handoff report to d:\Graph Coloring\.agents\teamwork\worker_m2_1\handoff.md.
7. Send a message to orchestrator (553217b8-f1c7-4ae8-b2fd-86f4b1a6dc9b) with your test results and summary.
