# BRIEFING — 2026-10-16T03:54:00Z

## Mission
Implement `scripts/build_splits.py`, generate `data/data_v1_2/splits.json`, and implement `tests/test_splits.py` satisfying all milestone M2 requirements.

## 🔒 My Identity
- Archetype: worker
- Roles: implementer, qa
- Working directory: d:\Graph Coloring\.agents\teamwork\worker_m2_1
- Original parent: 553217b8-f1c7-4ae8-b2fd-86f4b1a6dc9b
- Milestone: M2

## 🔒 Key Constraints
- Exclusively own: `scripts/build_splits.py`, `data/data_v1_2/splits.json`, `tests/test_splits.py`.
- Do NOT touch `data/data_v1/` or `data/data_v1_1/`.
- Do NOT touch frozen files (`test_frozen_files.py` enforces integrity of frozen files).
- Pure genuine implementation, no cheating, no hardcoded dummy outputs.

## Current Parent
- Conversation ID: 553217b8-f1c7-4ae8-b2fd-86f4b1a6dc9b
- Updated: 2026-10-16T03:54:00Z

## Task Summary
- **What to build**: `scripts/build_splits.py`, `data/data_v1_2/splits.json`, `tests/test_splits.py`.
- **Success criteria**: All 6 checks pass in `tests/test_splits.py`, full `pytest` passes cleanly including `test_frozen_files.py`.
- **Interface contracts**: `PROJECT.md`, `ORIGINAL_REQUEST.md`, `explorer_r1r2_1/report.md`, `SEEDS.md`.

## Key Decisions Made
- Partitioned 106 existing families into 6 disjoint groups:
  - `train_pool`: 41 families, seed "train" & "rl_train_pool", includes 5 scale_free families, excludes small_world.
  - `val`: 7 families (1 per structure type), seed "val".
  - `test_interp`: 7 families (1 per structure type with intermediate parameters), seed "test_interp".
  - `test_extrap`: 15 families (all n in [320, 500] > 300), seed "test_extrap".
  - `test_comp`: 10 families, held-out combination [2, 'low', 'low'] (absent from train_pool), seed "test_comp".
  - `heldout_type`: 26 families (all 7 small_world <= 300, 7 geometric, 7 queen, 5 hybrid), seed "test_heldout_type".
- Derived deterministic attributes: `n_range`, `density_bin`, `degree_variance_bin`, `clustering_bin`.
- Fully documented binning rules and thresholds in `splits.json` under `"binning"` key.

## Artifact Index
- `scripts/build_splits.py` — Deterministic split builder CLI script and library function
- `data/data_v1_2/splits.json` — Dataset v1.2 split definitions for all 106 families
- `tests/test_splits.py` — 13 unit tests verifying schema, all 6 checks, determinism, and CLI
- `.agents/teamwork/worker_m2_1/handoff.md` — Self-contained handoff report

## Change Tracker
- **Files modified**:
  - `scripts/build_splits.py`: created
  - `data/data_v1_2/splits.json`: created
  - `tests/test_splits.py`: created
- **Build status**: 113/113 tests passed (pytest)
- **Pending issues**: None

## Quality Status
- **Build/test result**: 13/13 passed in `test_splits.py`, 113/113 passed in full test suite
- **Lint status**: Clean
- **Tests added/modified**: `tests/test_splits.py` (13 test cases)
