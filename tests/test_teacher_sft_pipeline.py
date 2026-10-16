import json
import pytest
from pathlib import Path

from scripts.build_teacher_data import build_teacher_dataset
from scripts.filter_sft_data import filter_and_build_sft_dataset


def test_build_teacher_and_filter_sft_pipeline(tmp_path):
    t_dir = tmp_path / "teacher"
    sft_dir = tmp_path / "sft"

    # 1. Run build_teacher_dataset with 2 families in mock mode
    fams = ["ba_m12_n200-300", "ws_k10_p0.1_n100-200"]
    summary_t = build_teacher_dataset(
        families_list=fams,
        runs_dirs=[],
        out_dir=str(t_dir),
        mock=True,
    )

    records_file = t_dir / "records.jsonl"
    assert records_file.exists()
    assert summary_t["total_records"] > 0

    with open(records_file, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Verify Section 14.1 Schema
    required_keys = {
        "id", "family_id", "card", "card_detailed", "attributes",
        "paraphrase_idx", "teacher", "teacher_run_id", "budget", "code",
        "role", "rank_in_card", "score", "fingerprint", "is_baseline_like",
        "split", "seeds",
    }
    for r in records:
        assert required_keys.issubset(r.keys()), f"Record missing keys: {required_keys - set(r.keys())}"
        assert r["seeds"]["selection"] == "3000-3019"
        assert r["seeds"]["rescore"] == "6000-6049"
        assert r["seeds"]["probe"] == "9000-9009"
        assert r["role"] in {"win", "tie", "contrast", "loss"}

    # 2. Run filter_and_build_sft_dataset
    summary_sft = filter_and_build_sft_dataset(
        records_path=str(records_file),
        out_dir=str(sft_dir),
        val_ratio=0.5,
        seed=42,
    )

    assert (sft_dir / "train.jsonl").exists()
    assert (sft_dir / "val.jsonl").exists()
    assert (sft_dir / "contrast.jsonl").exists()

    with open(sft_dir / "train.jsonl", "r", encoding="utf-8") as f:
        train_samples = [json.loads(l) for l in f if l.strip()]
    with open(sft_dir / "val.jsonl", "r", encoding="utf-8") as f:
        val_samples = [json.loads(l) for l in f if l.strip()]

    # Invariant: card-level disjoint partitioning (no card appears in both train and val)
    train_cards = set(s["card"] for s in train_samples)
    val_cards = set(s["card"] for s in val_samples)
    assert train_cards.isdisjoint(val_cards), f"Data leakage! Overlapping cards: {train_cards & val_cards}"

    # Sample weights
    for s in train_samples + val_samples:
        assert s["weight"] == pytest.approx(1.0 / max(1, s["rank"]))
        assert "[SYSTEM]" in s["prompt_text"]
        assert "[USER]" in s["prompt_text"]
        assert "[ASSISTANT]" in s["prompt_text"]
