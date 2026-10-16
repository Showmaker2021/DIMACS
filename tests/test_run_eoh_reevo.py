import json
import pytest
from pathlib import Path

from baselines.run_eoh import run_eoh
from baselines.run_reevo import run_reevo


def test_eoh_runner_mock(tmp_path):
    family_id = "ba_m12_n200-300"
    budget = 4

    summary = run_eoh(
        family_id=family_id,
        run_seed=123,
        budget=budget,
        mock=True,
        out_dir=str(tmp_path / "eoh"),
    )

    assert summary["family_id"] == family_id
    assert summary["method"] == "EoH"
    assert summary["actual_calls"] == budget
    assert summary["total_prompt_tokens"] > 0
    assert summary["total_completion_tokens"] > 0

    run_dir = tmp_path / "eoh" / family_id / summary["run_id"]
    assert (run_dir / "candidates.jsonl").exists()
    assert (run_dir / "best.json").exists()
    assert (run_dir / "summary.json").exists()

    with open(run_dir / "candidates.jsonl", "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == budget
    for line in lines:
        assert "prompt_tokens" in line
        assert "completion_tokens" in line
        assert "latency_s" in line
        assert "operator" in line
        assert "status" in line


def test_reevo_runner_mock(tmp_path):
    family_id = "ba_m12_n200-300"
    budget = 5

    summary = run_reevo(
        family_id=family_id,
        run_seed=456,
        budget=budget,
        mock=True,
        out_dir=str(tmp_path / "reevo"),
    )

    assert summary["family_id"] == family_id
    assert summary["method"] == "ReEvo"
    assert summary["actual_calls"] == budget
    assert summary["total_reflections"] >= 1

    run_dir = tmp_path / "reevo" / family_id / summary["run_id"]
    assert (run_dir / "candidates.jsonl").exists()
    assert (run_dir / "best.json").exists()
    assert (run_dir / "summary.json").exists()

    with open(run_dir / "candidates.jsonl", "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == budget
    operators = [l["operator"] for l in lines]
    assert "reevo_reflection" in operators
