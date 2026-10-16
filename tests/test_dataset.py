"""Tests for dataset loading, manifest integrity, and split properties."""

import pytest
import numpy as np
from gate.dataset import families, load_family


def test_families_metadata():
    fams = families()
    assert len(fams) == 106
    for f in fams:
        assert "family_id" in f
        assert "structure_type" in f
        assert "pool" in f
        assert "labels" in f
        assert "cards" in f
        assert "n_range" in f
        assert f["pool"] in {"train_pool", "heldout_type", "heldout_size", "heldout_combo"}


def test_families_pool_filter():
    train_fams = families(pool="train_pool")
    assert len(train_fams) > 0
    assert all(f["pool"] == "train_pool" for f in train_fams)


def test_load_family_splits_and_sha():
    fams = families()
    sample_fam = fams[0]["family_id"]

    # Baseline train (seeds 0-19)
    bt_graphs = load_family(sample_fam, "baseline_train", check_sha=True)
    assert len(bt_graphs) == 20
    bt_seeds = [s for s, _ in bt_graphs]
    assert bt_seeds == list(range(20))

    # Val (20 graphs)
    val_graphs = load_family(sample_fam, "val", check_sha=True)
    assert len(val_graphs) == 20
    val_seeds = [s for s, _ in val_graphs]

    # Test (50 graphs)
    test_graphs = load_family(sample_fam, "test", check_sha=True)
    assert len(test_graphs) == 50
    test_seeds = [s for s, _ in test_graphs]

    # Verify disjoint seeds
    assert set(bt_seeds).isdisjoint(set(val_seeds))
    assert set(val_seeds).isdisjoint(set(test_seeds))
    assert set(bt_seeds).isdisjoint(set(test_seeds))

    # Verify graph properties
    for _, adj in bt_graphs[:5]:
        assert adj.dtype == bool
        assert adj.ndim == 2
        assert adj.shape[0] == adj.shape[1]
        assert np.array_equal(adj, adj.T)
        assert not np.any(np.diag(adj))


def test_seed_ranges_disjoint():
    import json
    from pathlib import Path

    config_v1_1_path = Path("data/data_v1_1/config.json")
    assert config_v1_1_path.exists()
    with open(config_v1_1_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    splits = cfg["splits"]
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert "teacher_val" in splits
    assert "test_pilot" in splits
    assert "probe" in splits
    assert splits["test_pilot"] == [4000, 200]

    ranges = {}
    for name, (start, count) in splits.items():
        ranges[name] = set(range(start, start + count))

    # Pairwise disjoint checks
    split_names = list(ranges.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            n1, n2 = split_names[i], split_names[j]
            overlap = ranges[n1].intersection(ranges[n2])
            assert not overlap, f"Seed collision between {n1} and {n2}: {overlap}"


def test_dataset_v1_1_metadata():
    import json
    from pathlib import Path

    v1_1_fam_path = Path("data/data_v1_1/families_v1_1.json")
    assert v1_1_fam_path.exists()
    with open(v1_1_fam_path, "r", encoding="utf-8") as f:
        fams = json.load(f)

    assert len(fams) == 106
    for f in fams:
        assert "card_detailed" in f["cards"], f"Missing card_detailed in {f['family_id']}"
        assert len(f["cards"]["card_detailed"]) > len(f["cards"]["card"])

    # Scale-free families moved to train_pool
    sf_fams = [f for f in fams if f["structure_type"] == "scale_free"]
    assert len(sf_fams) == 7
    for f in sf_fams:
        assert f["pool"] == "train_pool", f"{f['family_id']} pool is {f['pool']}, expected train_pool"

    # Small-world families: 7 moved to heldout_type, 2 kept as heldout_size
    sw_fams = [f for f in fams if f["structure_type"] == "small_world"]
    assert len(sw_fams) == 9
    assert sum(1 for f in sw_fams if f["pool"] == "heldout_type") == 7
    assert sum(1 for f in sw_fams if f["pool"] == "heldout_size") == 2

    # Mycielski families: kept in train_pool (8) and heldout_size (2), NOT heldout_type
    myc_fams = [f for f in fams if f["structure_type"] == "mycielski"]
    assert len(myc_fams) == 10
    assert sum(1 for f in myc_fams if f["pool"] == "train_pool") == 8
    assert sum(1 for f in myc_fams if f["pool"] == "heldout_size") == 2
    assert sum(1 for f in myc_fams if f["pool"] == "heldout_type") == 0

    # Config checks
    cfg_path = Path("data/data_v1_1/config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["heldout_types"] == ["geometric", "hybrid_geometric", "queen", "small_world"]
    assert cfg["splits"]["teacher_val"] == [3000, 20]
    assert cfg["splits"]["test_pilot"] == [4000, 200]
    assert cfg["counts"]["test_pilot"] == 200


def test_load_family_teacher_val_and_test_pilot():
    sample_fam = "kpart_k10_p0.3_n100-200"
    tv_graphs = load_family(sample_fam, "teacher_val")
    assert len(tv_graphs) == 20
    assert [s for s, _ in tv_graphs] == list(range(3000, 3020))

    tp_graphs = load_family(sample_fam, "test_pilot")
    assert len(tp_graphs) == 200
    assert [s for s, _ in tp_graphs] == list(range(4000, 4200))


