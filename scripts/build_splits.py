#!/usr/bin/env python3
"""Build Dataset v1.2 splits definition on existing 106 families.

Partitions existing families into 6 disjoint groups:
- train_pool: primary heuristic training pool (includes scale_free, excludes small_world)
- val: validation families across 7 training structure types
- test_interp: in-distribution parameter interpolation families
- test_extrap: graph size and density extrapolation families (n >= 320)
- test_comp: compositional holdout families (held-out attribute combination: [2, 'low', 'low'])
- heldout_type: unseen structure types (primary target: small_world; also geometric, queen, hybrid)

Usage:
    python scripts/build_splits.py [--input PATH] [--output PATH]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

DENSITY_EDGES = [0.05, 0.10, 0.20, 0.40]

VAL_FAMILY_IDS = [
    "gnp_d0.10_n50-100",           # gnp
    "sbm_k4_in0.7_out0.10_n50-100", # planted_partition
    "myciel_t1_p0.3_n50-100",       # mycielski
    "cave_k8_p0.1_n100-200",        # caveman
    "plc_m3_p0.3_n100-200",         # scale_free
    "kpart_k8_p0.3_n100-200",       # equipartite
    "interval_L0.03_n100-200",      # interval
]

TEST_INTERP_FAMILY_IDS = [
    "gnp_d0.40_n100-200",           # gnp (density 0.40 interpolates between 0.30 and 0.50)
    "sbm_k6_in0.4_out0.05_n100-200", # planted_partition (k=6 interpolates between 4 and 8)
    "myciel_t2_p0.05_n100-200",     # mycielski (t=2, p=0.05)
    "cave_k15_p0.1_n100-200",       # caveman (k=15 interpolates between 10 and 20)
    "ba_m5_n100-200",               # scale_free (m=5 interpolates between 3 and 8)
    "kpart_k15_p0.4_n200-300",      # equipartite (k=15 interpolates between 10 and 20)
    "interval_L0.15_n50-100",       # interval (L=0.15 interpolates between 0.10 and 0.20)
]

HELDOUT_COMBINATIONS = [
    [2, "low", "low"]
]


def derive_attributes(fam: Dict[str, Any]) -> Dict[str, Any]:
    """Derive deterministic attribute bins from family probe_stats.

    Attributes:
    - n_range: [min_n, max_n]
    - density_bin: 0 (<0.05), 1 (0.05-0.10), 2 (0.10-0.20), 3 (0.20-0.40), 4 (>=0.40)
    - degree_variance_bin: 'low' (cv < 0.2), 'medium' (0.2 <= cv < 0.5), 'high' (cv >= 0.5)
    - clustering_bin: 'low' (ratio < 1.5), 'moderate' (1.5 <= ratio < 4.0), 'high' (ratio >= 4.0)
      where ratio = clustering / max(density, 1e-9)
    """
    stats = fam.get("probe_stats", {})
    density = float(stats.get("density", 0.0))
    deg_cv = float(stats.get("deg_cv", 0.0))
    clustering = float(stats.get("clustering", 0.0))
    ratio = clustering / max(density, 1e-9)

    density_bin = int(np.searchsorted(DENSITY_EDGES, density))

    if deg_cv < 0.2:
        dv_bin = "low"
    elif deg_cv < 0.5:
        dv_bin = "medium"
    else:
        dv_bin = "high"

    if ratio < 1.5:
        cl_bin = "low"
    elif ratio < 4.0:
        cl_bin = "moderate"
    else:
        cl_bin = "high"

    return {
        "n_range": list(fam.get("n_range", [])),
        "density_bin": density_bin,
        "degree_variance_bin": dv_bin,
        "clustering_bin": cl_bin,
    }


def format_family_entry(fam: Dict[str, Any]) -> Dict[str, Any]:
    """Format single family entry for splits.json."""
    attrs = derive_attributes(fam)
    return {
        "family_id": fam["family_id"],
        "structure_type": fam["structure_type"],
        "n_range": attrs["n_range"],
        "density_bin": attrs["density_bin"],
        "degree_variance_bin": attrs["degree_variance_bin"],
        "clustering_bin": attrs["clustering_bin"],
    }


def build_splits(src_families_path: Path) -> Dict[str, Any]:
    """Partition families from families_v1_1.json into 6 disjoint splits.

    Groups:
    1. train_pool (41 families): Primary training pool, includes scale_free, excludes small_world.
    2. val (7 families): One family per training structure type for hyperparameter selection.
    3. test_interp (7 families): In-distribution interpolation test set.
    4. test_extrap (15 families): Size extrapolation test set (n in [320, 500] > 300).
    5. test_comp (10 families): Compositional holdout test set with held-out combination [2, 'low', 'low'].
    6. heldout_type (26 families): Out-of-distribution structure types (primary: small_world).
    """
    with open(src_families_path, "r", encoding="utf-8") as f:
        fams = json.load(f)

    fams_by_id = {f["family_id"]: f for f in fams}
    val_set = set(VAL_FAMILY_IDS)
    test_interp_set = set(TEST_INTERP_FAMILY_IDS)

    # Validate that predefined val and test_interp families exist and come from v1.1 train_pool
    for fid in VAL_FAMILY_IDS:
        if fid not in fams_by_id:
            raise ValueError(f"Predefined val family not found in source dataset: {fid}")
        if fams_by_id[fid].get("pool") != "train_pool":
            raise ValueError(f"Predefined val family {fid} is not from train_pool in v1.1")

    for fid in TEST_INTERP_FAMILY_IDS:
        if fid not in fams_by_id:
            raise ValueError(f"Predefined test_interp family not found in source dataset: {fid}")
        if fams_by_id[fid].get("pool") != "train_pool":
            raise ValueError(f"Predefined test_interp family {fid} is not from train_pool in v1.1")

    val_families = []
    test_interp_families = []
    test_comp_families = []
    test_extrap_families = []
    heldout_type_families = []
    train_pool_families = []

    for fam in fams:
        fid = fam["family_id"]
        v1_pool = fam.get("pool")
        formatted = format_family_entry(fam)

        if v1_pool == "heldout_combo":
            test_comp_families.append(formatted)
        elif v1_pool == "heldout_size":
            test_extrap_families.append(formatted)
        elif v1_pool == "heldout_type":
            heldout_type_families.append(formatted)
        elif v1_pool == "train_pool":
            if fid in val_set:
                val_families.append(formatted)
            elif fid in test_interp_set:
                test_interp_families.append(formatted)
            else:
                train_pool_families.append(formatted)
        else:
            raise ValueError(f"Unrecognized pool '{v1_pool}' for family {fid}")

    # Sort each split deterministically by family_id
    val_families.sort(key=lambda x: x["family_id"])
    test_interp_families.sort(key=lambda x: x["family_id"])
    test_comp_families.sort(key=lambda x: x["family_id"])
    test_extrap_families.sort(key=lambda x: x["family_id"])
    heldout_type_families.sort(key=lambda x: x["family_id"])
    train_pool_families.sort(key=lambda x: x["family_id"])

    splits_data = {
        "version": "v1.2",
        "binning": {
            "combination_attributes": [
                "density_bin",
                "degree_variance_bin",
                "clustering_bin",
            ],
            "density_edges": DENSITY_EDGES,
            "density_bins_description": {
                "0": "density < 0.05",
                "1": "0.05 <= density < 0.10",
                "2": "0.10 <= density < 0.20",
                "3": "0.20 <= density < 0.40",
                "4": "density >= 0.40",
            },
            "degree_variance_thresholds": {
                "metric": "deg_cv = std(deg) / mean(deg)",
                "low": "deg_cv < 0.2",
                "medium": "0.2 <= deg_cv < 0.5",
                "high": "deg_cv >= 0.5",
            },
            "clustering_thresholds": {
                "metric": "clustering_ratio = clustering / max(density, 1e-9)",
                "low": "ratio < 1.5",
                "moderate": "1.5 <= ratio < 4.0",
                "high": "ratio >= 4.0",
            },
        },
        "heldout_combinations": HELDOUT_COMBINATIONS,
        "splits": {
            "train_pool": {
                "seed_range": "train",
                "rl_seed_range": "rl_train_pool",
                "seed_ranges": ["train", "rl_train_pool"],
                "count": len(train_pool_families),
                "families": train_pool_families,
            },
            "val": {
                "seed_range": "val",
                "seed_ranges": ["val"],
                "count": len(val_families),
                "families": val_families,
            },
            "test_interp": {
                "seed_range": "test_interp",
                "seed_ranges": ["test_interp"],
                "count": len(test_interp_families),
                "families": test_interp_families,
            },
            "test_extrap": {
                "seed_range": "test_extrap",
                "seed_ranges": ["test_extrap"],
                "count": len(test_extrap_families),
                "families": test_extrap_families,
            },
            "test_comp": {
                "seed_range": "test_comp",
                "seed_ranges": ["test_comp"],
                "heldout_combinations": HELDOUT_COMBINATIONS,
                "count": len(test_comp_families),
                "families": test_comp_families,
            },
            "heldout_type": {
                "seed_range": "test_heldout_type",
                "seed_ranges": ["test_heldout_type"],
                "target_type": "small_world",
                "count": len(heldout_type_families),
                "families": heldout_type_families,
            },
        },
    }

    return splits_data


def main():
    parser = argparse.ArgumentParser(description="Generate Dataset v1.2 splits.json")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/data_v1_1/families_v1_1.json"),
        help="Path to source families_v1_1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/data_v1_2/splits.json"),
        help="Path to output splits.json",
    )
    args = parser.parse_args()

    splits = build_splits(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    total_fams = sum(g["count"] for g in splits["splits"].values())
    print(f"Generated Dataset v1.2 splits successfully -> {args.output}")
    print(f"Total families partitioned: {total_fams}")
    for group_name, group_data in splits["splits"].items():
        print(f"  - {group_name:15s}: {group_data['count']:2d} families (seed: {group_data['seed_range']})")


if __name__ == "__main__":
    main()
