"""Filter and partition teacher data into SFT training datasets.

Implements Task 8.3 and Task 10.1 from PLAN_PHASE6_ONWARD.md:
- Selects win and tie records for primary SFT (D-9).
- Separates contrast records for ablation.
- Partitions train and val strictly by CARD (not by paraphrase) to prevent data leakage.
- Builds training prompts conforming to Section 14.2 format.
- Computes sample weights inversely proportional to rank within each card.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.evolved.prompts import SYSTEM_PROMPT


SFT_PROMPT_TEMPLATE = """[SYSTEM] You write Python priority functions for greedy graph coloring.
[USER]
Task: write `priority(adj, colors, sat, deg, uncolored_deg)` returning a NumPy array of scores (one per vertex).
At each step the greedy framework colors the UNCOLORED vertex with the highest score using the smallest feasible non-conflicting color.

Parameters:
  adj:           (n, n) bool ndarray, symmetric, no self-loops (read-only)
  colors:        (n,) int64 ndarray, -1 = uncolored, >=0 color id (read-only)
  sat:           (n,) int64 ndarray, number of distinct colors among colored neighbors (read-only)
  deg:           (n,) int64 ndarray, total degree in full graph (read-only)
  uncolored_deg: (n,) int64 ndarray, number of uncolored neighbors (read-only)

Rules:
- Pure, deterministic function returning (n,) float ndarray.
- No imports. Only whitelisted NumPy functions and array methods are allowed.
- No globals, no internal state between calls, no random generation, no file I/O.

Graph family description:
{card}

Return only the function code.
[ASSISTANT]
{code}"""


def filter_and_build_sft_dataset(
    records_path: str = "runs/teacher/dataset_v1/records.jsonl",
    out_dir: str = "data/sft",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Filter teacher records, partition by card, and export SFT JSONL datasets."""
    p_rec = Path(records_path)
    if not p_rec.exists():
        raise FileNotFoundError(f"Teacher records file not found at: {records_path}")

    records: List[Dict[str, Any]] = []
    with open(p_rec, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Separate into win, tie, contrast, loss
    wins = []
    ties = []
    contrasts = []
    losses = []

    for r in records:
        role = r.get("role", "loss")
        if role == "win":
            wins.append(r)
        elif role == "tie":
            ties.append(r)
        elif role == "contrast":
            contrasts.append(r)
        else:
            losses.append(r)

    # Primary SFT set is win + tie (D-9)
    primary_records = wins + ties

    # Unique cards to split by card
    unique_cards = sorted(list(set(r.get("card", "") for r in primary_records if r.get("card"))))
    rng = np.random.default_rng(seed)
    shuffled_cards = list(unique_cards)
    rng.shuffle(shuffled_cards)

    n_val_cards = max(1, int(len(shuffled_cards) * val_ratio)) if len(shuffled_cards) > 1 else 0
    val_card_set = set(shuffled_cards[:n_val_cards])
    train_card_set = set(shuffled_cards[n_val_cards:])

    train_samples = []
    val_samples = []

    for r in primary_records:
        card = r.get("card", "")
        code = r.get("code", "")
        rank = r.get("rank_in_card", 1)
        # Sample weight = 1.0 / rank (rank 1 has weight 1.0, rank 2 has 0.5, etc.)
        weight = 1.0 / float(max(1, rank))

        sample = {
            "id": r.get("id"),
            "family_id": r.get("family_id"),
            "card": card,
            "rank": rank,
            "weight": weight,
            "prompt_text": SFT_PROMPT_TEMPLATE.format(card=card, code=code),
            "input_text": SFT_PROMPT_TEMPLATE.split("[ASSISTANT]")[0].format(card=card).strip(),
            "target_code": code,
            "role": r.get("role"),
            "gap_vs_rf_pct": r.get("score", {}).get("gap_vs_RF_pct", 0.0),
        }

        if card in val_card_set:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    contrast_samples = []
    for r in contrasts:
        card = r.get("card", "")
        code = r.get("code", "")
        sample = {
            "id": r.get("id"),
            "family_id": r.get("family_id"),
            "card": card,
            "rank": r.get("rank_in_card", 99),
            "weight": 0.5,
            "prompt_text": SFT_PROMPT_TEMPLATE.format(card=card, code=code),
            "target_code": code,
            "role": "contrast",
        }
        contrast_samples.append(sample)

    # Write files
    p_out = Path(out_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    with open(p_out / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(p_out / "val.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    with open(p_out / "contrast.jsonl", "w", encoding="utf-8") as f:
        for s in contrast_samples:
            f.write(json.dumps(s) + "\n")

    summary = {
        "total_records": len(records),
        "wins": len(wins),
        "ties": len(ties),
        "contrasts": len(contrasts),
        "losses": len(losses),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "contrast_samples": len(contrast_samples),
        "train_cards_count": len(train_card_set),
        "val_cards_count": len(val_card_set),
    }

    with open(p_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Filter and partition teacher data into SFT format")
    parser.add_argument("--records-path", "--input", dest="records_path", default="runs/teacher/dataset_v1/records.jsonl", help="Source records.jsonl")
    parser.add_argument("--out-dir", default="data/sft", help="Output SFT directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio by card")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    args = parser.parse_args()

    filter_and_build_sft_dataset(
        records_path=args.records_path,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
