"""Unit tests for Dataset v1.2 splits definition and generator script.

Verifies all 6 invariant checks from ORIGINAL_REQUEST.md lines 46-53:
1. Every family appears in exactly one group.
2. No held-out compositional combination appears in train_pool.
3. small_world appears neither in train_pool nor in val.
4. Extrapolation families really exceed the train range in n or density.
5. Every referenced seed-range name exists in SEEDS.md and ranges are consistent.
6. scale_free is in train_pool.

Plus comprehensive schema, determinism, and CLI execution tests.
"""

import json
from pathlib import Path
import subprocess
import sys
import pytest
import numpy as np

from scripts.build_splits import build_splits, derive_attributes, DENSITY_EDGES

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = REPO_ROOT / "data" / "data_v1_2" / "splits.json"
V1_1_FAMILIES_PATH = REPO_ROOT / "data" / "data_v1_1" / "families_v1_1.json"
SEEDS_PATH = REPO_ROOT / "SEEDS.md"

EXPECTED_GROUPS = {
    "train_pool",
    "val",
    "test_interp",
    "test_extrap",
    "test_comp",
    "heldout_type",
}


@pytest.fixture(scope="module")
def splits_data():
    """Load and return data/data_v1_2/splits.json."""
    assert SPLITS_PATH.is_file(), f"splits.json missing at {SPLITS_PATH}"
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def v1_1_families():
    """Load and return data/data_v1_1/families_v1_1.json."""
    assert V1_1_FAMILIES_PATH.is_file(), f"families_v1_1.json missing at {V1_1_FAMILIES_PATH}"
    with open(V1_1_FAMILIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def seeds_registry():
    """Parse SEEDS.md and return dictionary mapping range name -> record."""
    assert SEEDS_PATH.is_file(), f"SEEDS.md missing at {SEEDS_PATH}"
    content = SEEDS_PATH.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    registry = {}
    in_table = False
    for line in lines:
        if line.startswith("|") and line.endswith("|"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if not parts:
                continue
            if parts[0] == "name":
                in_table = True
                continue
            if in_table and all(c in "- :" for c in "".join(parts)):
                continue
            if in_table:
                name, start_s, end_s, status, purpose = parts
                registry[name] = {
                    "name": name,
                    "start": int(start_s),
                    "end": int(end_s),
                    "status": status,
                    "purpose": purpose,
                }
    return registry


# ----------------------------------------------------------------------------
# 1. Basic File & Schema Tests
# ----------------------------------------------------------------------------

def test_splits_file_exists():
    assert SPLITS_PATH.is_file()


def test_splits_json_schema(splits_data):
    """Verify top-level keys and structure of splits.json."""
    assert "version" in splits_data
    assert splits_data["version"] == "v1.2"
    assert "binning" in splits_data
    assert "heldout_combinations" in splits_data
    assert "splits" in splits_data

    splits = splits_data["splits"]
    assert set(splits.keys()) == EXPECTED_GROUPS

    # Verify binning metadata documentation
    binning = splits_data["binning"]
    assert "density_edges" in binning
    assert binning["density_edges"] == DENSITY_EDGES
    assert "degree_variance_thresholds" in binning
    assert "clustering_thresholds" in binning


def test_family_record_schema(splits_data):
    """Verify schema and value types for every family in every split."""
    for group_name, group in splits_data["splits"].items():
        assert "seed_range" in group, f"Group {group_name} missing seed_range"
        assert "count" in group, f"Group {group_name} missing count"
        assert "families" in group, f"Group {group_name} missing families"
        assert group["count"] == len(group["families"]), (
            f"Group {group_name} count mismatch: {group['count']} vs {len(group['families'])}"
        )

        for fam in group["families"]:
            assert isinstance(fam["family_id"], str) and fam["family_id"]
            assert isinstance(fam["structure_type"], str) and fam["structure_type"]
            assert isinstance(fam["n_range"], list) and len(fam["n_range"]) == 2
            assert fam["n_range"][0] <= fam["n_range"][1]
            assert isinstance(fam["density_bin"], int) and 0 <= fam["density_bin"] <= 4
            assert fam["degree_variance_bin"] in {"low", "medium", "high"}
            assert fam["clustering_bin"] in {"low", "moderate", "high"}


# ----------------------------------------------------------------------------
# 2. Six Invariant Checks from ORIGINAL_REQUEST.md
# ----------------------------------------------------------------------------

def test_check1_every_family_appears_in_exactly_one_group(splits_data, v1_1_families):
    """Check 1: Every family appears in exactly one group (disjoint partition of all 106 families)."""
    splits = splits_data["splits"]
    family_to_groups = {}

    for group_name, group in splits.items():
        for fam in group["families"]:
            fid = fam["family_id"]
            if fid not in family_to_groups:
                family_to_groups[fid] = []
            family_to_groups[fid].append(group_name)

    # Check 1a: No family in multiple groups
    overlaps = {fid: grps for fid, grps in family_to_groups.items() if len(grps) > 1}
    assert not overlaps, f"Families appearing in multiple groups: {overlaps}"

    # Check 1b: Exact 106 families total
    assert len(family_to_groups) == 106, f"Expected 106 families, got {len(family_to_groups)}"

    # Check 1c: Matches all families from v1.1
    expected_fids = {f["family_id"] for f in v1_1_families}
    actual_fids = set(family_to_groups.keys())
    assert actual_fids == expected_fids, (
        f"Missing families: {expected_fids - actual_fids}, Extra families: {actual_fids - expected_fids}"
    )


def test_check2_no_heldout_compositional_combination_in_train_pool(splits_data):
    """Check 2: No held-out compositional combination appears in train_pool."""
    splits = splits_data["splits"]
    train_pool = splits["train_pool"]
    test_comp = splits["test_comp"]

    assert "heldout_combinations" in test_comp, "test_comp must define heldout_combinations"
    heldout_combos = {tuple(c) for c in test_comp["heldout_combinations"]}
    assert heldout_combos, "heldout_combinations must not be empty"

    # Extract all combinations present in train_pool
    train_combos = set()
    for fam in train_pool["families"]:
        combo = (fam["density_bin"], fam["degree_variance_bin"], fam["clustering_bin"])
        train_combos.add(combo)
        assert combo not in heldout_combos, (
            f"Train family {fam['family_id']} has held-out combination {combo}!"
        )

    # Intersection must be empty
    overlap = train_combos.intersection(heldout_combos)
    assert not overlap, f"Forbidden held-out combinations present in train_pool: {overlap}"

    # Also assert that all test_comp families indeed belong to held-out combinations
    for fam in test_comp["families"]:
        combo = (fam["density_bin"], fam["degree_variance_bin"], fam["clustering_bin"])
        assert combo in heldout_combos, (
            f"test_comp family {fam['family_id']} combination {combo} not in heldout_combinations {heldout_combos}"
        )


def test_check3_small_world_neither_in_train_pool_nor_in_val(splits_data):
    """Check 3: small_world appears neither in train_pool nor in val."""
    splits = splits_data["splits"]

    train_pool_sw = [f["family_id"] for f in splits["train_pool"]["families"] if f["structure_type"] == "small_world"]
    assert not train_pool_sw, f"small_world found in train_pool: {train_pool_sw}"

    val_sw = [f["family_id"] for f in splits["val"]["families"] if f["structure_type"] == "small_world"]
    assert not val_sw, f"small_world found in val: {val_sw}"

    # Assert small_world families exist and are placed in heldout_type and test_extrap
    heldout_type_sw = [f["family_id"] for f in splits["heldout_type"]["families"] if f["structure_type"] == "small_world"]
    test_extrap_sw = [f["family_id"] for f in splits["test_extrap"]["families"] if f["structure_type"] == "small_world"]
    assert len(heldout_type_sw) == 7, f"Expected 7 small_world in heldout_type, got {len(heldout_type_sw)}"
    assert len(test_extrap_sw) == 2, f"Expected 2 small_world in test_extrap, got {len(test_extrap_sw)}"
    assert len(heldout_type_sw) + len(test_extrap_sw) == 9, "All 9 small_world families must be accounted for"


def test_check4_extrapolation_families_really_exceed_train_range(splits_data):
    """Check 4: Extrapolation families really exceed the train range in n or density."""
    splits = splits_data["splits"]
    train_pool = splits["train_pool"]
    test_extrap = splits["test_extrap"]

    train_max_n = max(f["n_range"][1] for f in train_pool["families"])
    train_max_density_bin = max(f["density_bin"] for f in train_pool["families"])

    # In our dataset, train_max_n is 300
    assert train_max_n == 300, f"Expected train_pool max n to be 300, got {train_max_n}"

    # Every extrapolation family must exceed train maxima in n or density
    for fam in test_extrap["families"]:
        min_n = fam["n_range"][0]
        dens_bin = fam["density_bin"]
        exceeds_n = min_n > train_max_n
        exceeds_density = dens_bin > train_max_density_bin

        assert exceeds_n or exceeds_density, (
            f"Family {fam['family_id']} does not exceed train range! "
            f"n_range={fam['n_range']} (train max n={train_max_n}), "
            f"density_bin={dens_bin} (train max density_bin={train_max_density_bin})"
        )
        # All 15 heldout_size families have n in [320, 500] > 300
        assert min_n >= 320 > train_max_n


def test_check5_referenced_seed_ranges_exist_in_seeds_md(splits_data, seeds_registry):
    """Check 5: Every referenced seed-range name exists in SEEDS.md and ranges are consistent."""
    splits = splits_data["splits"]

    expected_mapping = {
        "train_pool": "train",
        "val": "val",
        "test_interp": "test_interp",
        "test_extrap": "test_extrap",
        "test_comp": "test_comp",
        "heldout_type": "test_heldout_type",
    }

    for group_name, expected_seed_name in expected_mapping.items():
        group = splits[group_name]
        primary_seed = group["seed_range"]
        assert primary_seed == expected_seed_name, (
            f"Group {group_name} seed_range mismatch: expected {expected_seed_name}, got {primary_seed}"
        )
        assert primary_seed in seeds_registry, f"Seed range '{primary_seed}' not found in SEEDS.md"

        # Check all referenced seed ranges in seed_ranges list
        for s_name in group.get("seed_ranges", [primary_seed]):
            assert s_name in seeds_registry, f"Referenced seed range '{s_name}' not in SEEDS.md"

    # In addition, check that train_pool references rl_train_pool
    assert splits["train_pool"]["rl_seed_range"] == "rl_train_pool"
    assert "rl_train_pool" in seeds_registry

    # Verify that test evaluation seed ranges are strictly disjoint from train and val
    train_interval = (seeds_registry["train"]["start"], seeds_registry["train"]["end"])
    val_interval = (seeds_registry["val"]["start"], seeds_registry["val"]["end"])

    for test_grp in ["test_interp", "test_extrap", "test_comp", "heldout_type"]:
        seed_name = splits[test_grp]["seed_range"]
        test_int = (seeds_registry[seed_name]["start"], seeds_registry[seed_name]["end"])

        # No overlap with train
        assert test_int[1] < train_interval[0] or test_int[0] > train_interval[1], (
            f"Test split {test_grp} [{test_int}] collides with train [{train_interval}]"
        )
        # No overlap with val
        assert test_int[1] < val_interval[0] or test_int[0] > val_interval[1], (
            f"Test split {test_grp} [{test_int}] collides with val [{val_interval}]"
        )


def test_check6_scale_free_is_in_train_pool(splits_data):
    """Check 6: scale_free is in train_pool."""
    splits = splits_data["splits"]
    train_pool = splits["train_pool"]

    sf_families = [f for f in train_pool["families"] if f["structure_type"] == "scale_free"]
    assert len(sf_families) >= 1, "train_pool must contain at least one scale_free family"
    assert len(sf_families) == 5, f"Expected 5 scale_free families in train_pool, got {len(sf_families)}"

    sf_ids = {f["family_id"] for f in sf_families}
    expected_sf = {
        "ba_m8_n100-300",
        "ba_m8_n200-300",
        "ba_m12_n200-300",
        "plc_m5_p0.5_n100-200",
        "plc_m8_p0.7_n200-300",
    }
    assert sf_ids == expected_sf


# ----------------------------------------------------------------------------
# 3. Additional Robustness, Determinism & Execution Tests
# ----------------------------------------------------------------------------

def test_no_split_under_minimum_families(splits_data):
    """Ensure no split has fewer than 3 families (all >= 7 in our design)."""
    splits = splits_data["splits"]
    small_splits = []

    for group_name, group in splits.items():
        count = group["count"]
        if count < 3:
            small_splits.append((group_name, count))
        assert count >= 7, f"Group {group_name} has {count} families, expected at least 7"

    assert not small_splits, f"Splits with fewer than 3 families detected: {small_splits}"


def test_attribute_binning_exact_match_with_probe_stats(splits_data, v1_1_families):
    """Verify that attributes in splits.json match independent derivation from probe stats."""
    v1_1_by_id = {f["family_id"]: f for f in v1_1_families}

    for group in splits_data["splits"].values():
        for fam in group["families"]:
            fid = fam["family_id"]
            orig = v1_1_by_id[fid]
            derived = derive_attributes(orig)

            assert fam["n_range"] == derived["n_range"], f"n_range mismatch for {fid}"
            assert fam["density_bin"] == derived["density_bin"], f"density_bin mismatch for {fid}"
            assert fam["degree_variance_bin"] == derived["degree_variance_bin"], f"deg_cv mismatch for {fid}"
            assert fam["clustering_bin"] == derived["clustering_bin"], f"clustering mismatch for {fid}"


def test_build_splits_determinism():
    """Verify build_splits is 100% deterministic across multiple calls."""
    run1 = build_splits(V1_1_FAMILIES_PATH)
    run2 = build_splits(V1_1_FAMILIES_PATH)

    dump1 = json.dumps(run1, sort_keys=True)
    dump2 = json.dumps(run2, sort_keys=True)
    assert dump1 == dump2, "build_splits returned non-deterministic output across runs"


def test_build_splits_cli_execution(tmp_path):
    """Verify scripts/build_splits.py can execute cleanly via CLI and produce valid output."""
    out_file = tmp_path / "splits_test.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_splits.py"),
        "--input",
        str(V1_1_FAMILIES_PATH),
        "--output",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == "v1.2"
    assert sum(g["count"] for g in data["splits"].values()) == 106
