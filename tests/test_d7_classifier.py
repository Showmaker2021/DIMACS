"""Unit tests for Task 9.1: D7 Card-Based Algorithm Selector Baseline."""

import json
from pathlib import Path
import numpy as np
import pytest

from baselines.d7_classifier import (
    PureLogisticRegression,
    PureDecisionTree,
    extract_card_features,
    train_d7_classifiers,
    evaluate_split,
    run_d7_benchmark,
)


def test_pure_logistic_regression_toy_data():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [-1.0, -2.0], [-2.0, -1.0]])
    y = np.array([1, 1, 0, 0])
    clf = PureLogisticRegression(l2_reg=0.1)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.array_equal(preds, y)


def test_pure_decision_tree_toy_data():
    X = np.array([[0.1, 10.0], [0.2, 12.0], [0.8, 1.0], [0.9, 2.0]])
    y = np.array([0, 0, 1, 1])
    dt = PureDecisionTree(max_depth=2)
    dt.fit(X, y)
    preds = dt.predict(X)
    assert np.array_equal(preds, y)


def test_extract_card_features():
    dummy_record = {
        "family_id": "test_fam",
        "n_range": [100, 200],
        "probe_stats": {
            "density": 0.35,
            "deg_cv": 0.45,
            "clustering": 0.15,
        }
    }
    feats = extract_card_features(dummy_record)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (4,)
    assert np.isclose(feats[0], 0.35)
    assert np.isclose(feats[1], 0.45)
    assert np.isclose(feats[2], 0.15)
    assert np.isclose(feats[3], 150.0 / 300.0)


def test_d7_training_on_train_pool():
    lr_model, dt_model, meta = train_d7_classifiers()
    assert meta["n_train_families"] == 41
    assert meta["rlf_wins_train"] + meta["dsatur_wins_train"] == 41
    assert lr_model.weights is not None
    assert dt_model.tree is not None


def test_d7_evaluation_on_val_split():
    with open("data/data_v1_2/splits.json", "r", encoding="utf-8") as f:
        splits_data = json.load(f)
    with open("data/data_v1_1/families_v1_1.json", "r", encoding="utf-8") as f:
        fams_list = json.load(f)
    fam_map = {f["family_id"]: f for f in fams_list}

    _, dt_model, _ = train_d7_classifiers()
    val_res = evaluate_split(dt_model, "val", splits_data, fam_map)

    assert val_res["split"] == "val"
    assert val_res["n_families"] == 7
    assert 0.0 <= val_res["accuracy"] <= 1.0
    assert val_res["mean_regret_colors"] >= -1e-6
    assert len(val_res["details"]) == 7


def test_d7_benchmark_output_artifacts(tmp_path):
    out_dir = tmp_path / "d7_test_run"
    summary = run_d7_benchmark(out_dir=str(out_dir))

    assert "train_metadata" in summary
    assert "decision_tree" in summary
    assert "logistic_regression" in summary

    res_json = out_dir / "results.json"
    res_md = out_dir / "summary.md"
    assert res_json.is_file()
    assert res_md.is_file()

    with open(res_json, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert "decision_tree" in saved_data
