# Handoff Report: Milestone M2 — Dataset v1.2 Splits Definition

**Author**: Worker Subagent `worker_m2_1`  
**Date**: 2026-10-16  
**Milestone**: M2 (R2 Dataset v1.2 Splits on Existing Families)  
**Assigned Artifacts**:
- `scripts/build_splits.py`
- `data/data_v1_2/splits.json`
- `tests/test_splits.py`

---

## 1. Observation

1. **Source Data & Preconditions**:
   - `data/data_v1_1/families_v1_1.json` contains 106 families across 11 structure types.
   - `data/data_v1_1/config.json` sets `heldout_types: ["geometric", "hybrid_geometric", "queen", "small_world"]` and `heldout_size_min: 320`.
   - `SEEDS.md` defines 13 closed, pairwise disjoint ranges:
     - `train`: [0, 99] (frozen)
     - `val`: [1000, 1019] (frozen)
     - `test_interp`: [7000, 7049] (proposed)
     - `test_extrap`: [7100, 7149] (proposed)
     - `test_comp`: [7200, 7249] (proposed)
     - `test_heldout_type`: [7300, 7349] (proposed)
     - `rl_train_pool`: [100000, 199999] (proposed)
2. **Explorer Proposal (`explorer_r1r2_1/report.md`)**:
   - Sections 5.2 & 6.2 specify binning thresholds:
     - `density_edges = [0.05, 0.10, 0.20, 0.40]`
     - `degree_variance_bin`: `low` (`deg_cv < 0.2`), `medium` (`0.2 <= deg_cv < 0.5`), `high` (`deg_cv >= 0.5`)
     - `clustering_bin`: `low` (`ratio < 1.5`), `moderate` (`1.5 <= ratio < 4.0`), `high` (`ratio >= 4.0`), where `ratio = clustering / max(density, 1e-9)`
   - Verified against probe stats in `families_v1_1.json`: exact 0 mismatches across all 106 families.
3. **Partition Results**:
   - `python scripts/build_splits.py` executed successfully and produced `data/data_v1_2/splits.json`.
   - Tool output verbatim:
     ```
     Generated Dataset v1.2 splits successfully -> data\data_v1_2\splits.json
     Total families partitioned: 106
       - train_pool     : 41 families (seed: train)
       - val            :  7 families (seed: val)
       - test_interp    :  7 families (seed: test_interp)
       - test_extrap    : 15 families (seed: test_extrap)
       - test_comp      : 10 families (seed: test_comp)
       - heldout_type   : 26 families (seed: test_heldout_type)
     ```
   - Zero splits have fewer than 3 families (all $\ge 7$).
4. **Test Execution**:
   - `pytest tests/test_splits.py -v`:
     ```
     tests/test_splits.py::test_splits_file_exists PASSED                     [  7%]
     tests/test_splits.py::test_splits_json_schema PASSED                     [ 15%]
     tests/test_splits.py::test_family_record_schema PASSED                   [ 23%]
     tests/test_splits.py::test_check1_every_family_appears_in_exactly_one_group PASSED [ 30%]
     tests/test_splits.py::test_check2_no_heldout_compositional_combination_in_train_pool PASSED [ 38%]
     tests/test_splits.py::test_check3_small_world_neither_in_train_pool_nor_in_val PASSED [ 46%]
     tests/test_splits.py::test_check4_extrapolation_families_really_exceed_train_range PASSED [ 53%]
     tests/test_splits.py::test_check5_referenced_seed_ranges_exist_in_seeds_md PASSED [ 61%]
     tests/test_splits.py::test_check6_scale_free_is_in_train_pool PASSED     [ 69%]
     tests/test_splits.py::test_no_split_under_minimum_families PASSED        [ 76%]
     tests/test_splits.py::test_attribute_binning_exact_match_with_probe_stats PASSED [ 84%]
     tests/test_splits.py::test_build_splits_determinism PASSED               [ 92%]
     tests/test_splits.py::test_build_splits_cli_execution PASSED             [100%]
     ============================= 13 passed in 0.35s ==============================
     ```
   - Full test suite run (`pytest`):
     ```
     ============================ 113 passed in 20.55s =============================
     ```
     All 100 pre-existing tests (including `test_frozen_files.py`) + 13 new unit tests passed cleanly.

---

## 2. Logic Chain

1. **Disjoint Partitioning (Check 1)**:
   - Each family from `families_v1_1.json` is assigned to exactly one group based on its pool and selection criteria:
     - `heldout_combo` $\to$ `test_comp` (10 families)
     - `heldout_size` $\to$ `test_extrap` (15 families)
     - `heldout_type` $\to$ `heldout_type` (26 families)
     - `train_pool`:
       - 7 selected families (1 per training structure type) $\to$ `val`
       - 7 selected families with intermediate parameters $\to$ `test_interp`
       - remaining 41 families $\to$ `train_pool`
   - Sum: $10 + 15 + 26 + 7 + 7 + 41 = 106$.
   - Pairwise intersection between all 6 groups is empty. Every family appears in exactly one group.
2. **Compositional Holdout (Check 2)**:
   - All 10 families in `test_comp` share combination `[2, 'low', 'low']` (`density_bin = 2`, `degree_variance_bin = 'low'`, `clustering_bin = 'low'`).
   - None of the 41 families in `train_pool` have combination `[2, 'low', 'low']`. The intersection between train combinations and held-out combinations is empty.
3. **Small-World Segregation (Check 3)**:
   - All 9 `small_world` families are partitioned into `heldout_type` (7 families with $n \le 300$) and `test_extrap` (2 families with $n \in [320, 500]$).
   - Zero `small_world` families exist in `train_pool` or `val`.
4. **Extrapolation Invariant (Check 4)**:
   - Maximum node count in `train_pool` is $\max(n) = 300$.
   - For all 15 families in `test_extrap`, minimum node count is $\min(n) = 320$.
   - Because $320 > 300$, all extrapolation families strictly exceed the training range in $n$.
5. **Seed Registry Consistency (Check 5)**:
   - Every group references its corresponding seed range name in `SEEDS.md`:
     - `train_pool` $\to$ `train` (and `rl_seed_range: rl_train_pool`)
     - `val` $\to$ `val`
     - `test_interp` $\to$ `test_interp`
     - `test_extrap` $\to$ `test_extrap`
     - `test_comp` $\to$ `test_comp`
     - `heldout_type` $\to$ `test_heldout_type`
   - All referenced names exist in `SEEDS.md` and their ranges are verified to be pairwise disjoint from training and validation.
6. **Scale-Free Inclusion (Check 6)**:
   - `train_pool` contains 5 `scale_free` families (`ba_m8_n100-300`, `ba_m8_n200-300`, `ba_m12_n200-300`, `plc_m5_p0.5_n100-200`, `plc_m8_p0.7_n200-300`).
   - In addition, `val` contains 1 (`plc_m3_p0.3_n100-200`) and `test_interp` contains 1 (`ba_m5_n100-200`). All 7 scale-free families are accounted for.
7. **Frozen Files Integrity**:
   - Neither `data/data_v1/`, `data/data_v1_1/`, nor `evaluator/verify.py` were modified.
   - Verified by `tests/test_frozen_files.py`.

---

## 3. Caveats

- **Scope Boundary**: Task 8.1 in this batch partitions the 106 existing families without synthesizing new parametric-variant generators. A future phase (Phase 8 full) may generate ~300 variant families.
- **Held-out Type Scope**: `heldout_type` contains all 26 families from v1.1 `heldout_type` (7 `small_world`, 7 `geometric`, 7 `queen`, 5 `hybrid_geometric`), with `target_type: "small_world"` designated as the benchmark target. All 26 families remain strictly held out from training.
- **No Small Splits**: None of the splits have fewer than 3 families (counts are 41, 7, 7, 15, 10, 26).

---

## 4. Conclusion

Milestone M2 is complete and verified:
1. `scripts/build_splits.py` provides deterministic, CLI-executable generation of `data/data_v1_2/splits.json`.
2. `data/data_v1_2/splits.json` partitions all 106 existing families into 6 groups with exact attribute binning metadata and documented held-out combinations.
3. `tests/test_splits.py` covers all 6 required invariant checks plus schema, determinism, and CLI tests.
4. All 113 tests in the test suite pass cleanly (100% green).

---

## 5. Verification Method

To independently verify this work:
1. Run the splits unit tests:
   ```powershell
   pytest tests/test_splits.py -v
   ```
   *Expected*: 13 passed in ~0.35s.
2. Run the full regression test suite (verifying frozen files and existing baselines):
   ```powershell
   pytest
   ```
   *Expected*: 113 passed in ~21s.
3. Verify CLI split regeneration:
   ```powershell
   python scripts/build_splits.py
   ```
   *Expected*: Prints 106 total families partitioned across all 6 groups with exit code 0.
