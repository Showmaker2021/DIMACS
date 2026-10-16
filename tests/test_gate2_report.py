import json
import pytest
from pathlib import Path

from scripts.gate2_report import (
    evaluate_gate2,
    TARGET_FAMILIES,
    CONTROL_FAMILIES,
    ALL_PILOT_FAMILIES,
    compute_wilcoxon_paired,
)
import numpy as np


def test_wilcoxon_paired_helper():
    # If target uses fewer colors (ref > cand), p should be small
    ref = np.array([15, 14, 16, 15, 14] * 5)
    cand = np.array([12, 11, 13, 12, 11] * 5)
    p = compute_wilcoxon_paired(ref, cand)
    assert p < 0.05

    # If identical, p should be 1.0
    p_ident = compute_wilcoxon_paired(ref, ref)
    assert p_ident == 1.0


def test_gate2_decision_rules():
    # Helper to test decision mapping
    def get_decision(pass_count, marginal_count, valid_100):
        if pass_count >= 2 and valid_100:
            return "PASS"
        elif pass_count >= 1 or marginal_count >= 1:
            return "MARGINAL"
        else:
            return "FAIL"

    assert get_decision(2, 0, True) == "PASS"
    assert get_decision(3, 1, True) == "PASS"
    assert get_decision(2, 0, False) == "MARGINAL"  # valid rate not 100% -> MARGINAL
    assert get_decision(1, 0, True) == "MARGINAL"
    assert get_decision(0, 2, True) == "MARGINAL"
    assert get_decision(0, 0, True) == "FAIL"


def test_gate2_report_mock_and_locks(tmp_path):
    out_dir = tmp_path / "runs" / "gate2"
    locks_dir = tmp_path / "runs" / "gate2" / "locks"

    # Pre-populate lock files for 5 families with mock data, so only 1 evaluates or all evaluate from mock
    # To test locks: create synthetic lock files for all 6 families
    locks_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup mock locked results representing a PASS outcome
    for idx, fid in enumerate(ALL_PILOT_FAMILIES):
        is_target = fid in TARGET_FAMILIES
        status = "PASS" if is_target and idx < 2 else ("FAIL" if is_target else "PASS")
        fam_data = {
            "family_id": fid,
            "role": "target" if is_target else "control",
            "rf_name": "RLF",
            "mean_top1": 12.0,
            "mean_dsatur": 14.0,
            "mean_rlf": 13.0,
            "mean_rf": 13.0,
            "gap_vs_rf_pct": 7.69 if status == "PASS" else 0.0,
            "gap_vs_rf_ci": [5.0, 10.0] if status == "PASS" else [-2.0, 2.0],
            "p_wilcoxon": 0.001 if status == "PASS" else 0.5,
            "behavioral_probe": {"equal_dsatur": False, "equal_rlf": False},
            "valid_rate": 1.0,
            "total_calls": 20,
            "fail_counts": {},
            "status": status,
        }
        lock_file = locks_dir / f"{fid}.done"
        with open(lock_file, "w", encoding="utf-8") as f:
            json.dump(fam_data, f, indent=2)

    # Evaluate using the locked directory
    res_pass = evaluate_gate2(
        runs_dir=str(tmp_path / "pilot"),
        out_dir=str(out_dir),
        locks_dir=str(locks_dir),
        force_unlock=False,
        mock=True,
    )

    assert res_pass["summary"]["gate_decision"] == "PASS"
    assert res_pass["summary"]["target_pass_count"] == 2
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.json").exists()

    with open(out_dir / "report.md", "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "GATE 2 OUTCOME: PASS" in md_text
    assert "Target Families" in md_text
    assert "Control Families" in md_text

    # 2. Test FAIL outcome by altering locks
    for fid in TARGET_FAMILIES:
        lock_file = locks_dir / f"{fid}.done"
        with open(lock_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["status"] = "FAIL"
        d["gap_vs_rf_pct"] = -1.0
        with open(lock_file, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    res_fail = evaluate_gate2(
        runs_dir=str(tmp_path / "pilot"),
        out_dir=str(out_dir),
        locks_dir=str(locks_dir),
        force_unlock=False,
        mock=True,
    )
    assert res_fail["summary"]["gate_decision"] == "FAIL"

    # 3. Test MARGINAL outcome by altering locks
    lock_file = locks_dir / f"{TARGET_FAMILIES[0]}.done"
    with open(lock_file, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["status"] = "PASS"
    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

    res_marginal = evaluate_gate2(
        runs_dir=str(tmp_path / "pilot"),
        out_dir=str(out_dir),
        locks_dir=str(locks_dir),
        force_unlock=False,
        mock=True,
    )
    assert res_marginal["summary"]["gate_decision"] == "MARGINAL"
