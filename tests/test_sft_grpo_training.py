import json
import pytest
from pathlib import Path

from train.train_sft import run_sft_training
from train.train_grpo import run_grpo_training
from train.rollout import sample_rl_graphs, RL_SEED_START, RL_SEED_END


def test_sft_mock_training(tmp_path):
    out_dir = tmp_path / "sft_out"
    meta = run_sft_training(
        output_dir=str(out_dir),
        mock=True,
    )

    assert meta["status"] == "completed"
    assert (out_dir / "training_history.json").exists()
    assert (out_dir / "checkpoint_metadata.json").exists()

    with open(out_dir / "training_history.json", "r", encoding="utf-8") as f:
        hist = json.load(f)
    assert len(hist) == 20
    assert "probe_savings_pct" in hist[0]


def test_grpo_mock_training_and_rl_seeds(tmp_path):
    # Test RL seed sampling
    graphs = sample_rl_graphs("ba_m12_n200-300", n_graphs=10)
    assert len(graphs) == 10
    for s, adj in graphs:
        assert RL_SEED_START <= s <= RL_SEED_END
        assert adj.shape[0] > 0

    # Test GRPO run across arms
    for arm in ["A2", "A3", "A4"]:
        out_dir = tmp_path / f"grpo_{arm}"
        summary = run_grpo_training(
            arm=arm,
            output_dir=str(out_dir),
            steps=5,
            group_size=4,
            mock=True,
        )

        assert summary["arm"] == arm
        assert summary["status"] == "completed"
        assert (out_dir / "step_logs.jsonl").exists()
        assert (out_dir / "summary.json").exists()

        with open(out_dir / "step_logs.jsonl", "r", encoding="utf-8") as f:
            logs = [json.loads(l) for l in f if l.strip()]
        assert len(logs) == 5
        for l in logs:
            assert "mean_reward" in l
            assert "mean_advantage" in l
            assert "fail_rate" in l
