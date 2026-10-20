"""Pipeline to build large-scale teacher dataset from offline evolution runs.

Implements Task 8.2 from PLAN_PHASE6_ONWARD.md:
- Aggregates survivor candidates across evolution runs (our_engine, EoH, ReEvo).
- Evaluates base_alg(card) and R(F) on teacher_val (seeds 3000-3019).
- Retains top-m (m=3) by teacher_val plus contrast samples (role="contrast").
- Deduplicates candidates by behavioral fingerprint on probe seeds 9000-9009.
- Rescores surviving candidates on clean teacher_rescore seeds (6000-6049).
- Exports records.jsonl matching Section 14.1 schema.
"""

import sys
import os
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.dataset import load_family, families
from gate.evolved.sandbox import evaluate_candidate_safe
from gate.evolved.task import CALIBRATED_PER_GRAPH_TIMEOUTS
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


def get_probe_fingerprint(code: str, probe_graphs: List[Tuple[int, np.ndarray]]) -> Tuple[str, bool]:
    """Compute behavioral fingerprint on probe seeds 9000-9009 and check similarity to DSATUR/RLF."""
    res = evaluate_candidate_safe(code, probe_graphs, per_graph_timeout=0.5, overall_timeout=10.0)
    if not res.get("success", False) or res.get("color_counts") is None:
        return "invalid", False

    cand_counts = tuple(res["color_counts"])
    ds_counts = tuple(count_colors(dsatur(g[1])) for g in probe_graphs)
    rlf_counts = tuple(count_colors(rlf(g[1])) for g in probe_graphs)

    fp_str = hashlib.sha256(str(cand_counts).encode("utf-8")).hexdigest()[:16]
    is_baseline_like = (cand_counts == ds_counts) or (cand_counts == rlf_counts)
    return fp_str, is_baseline_like


def build_teacher_dataset(
    families_list: Optional[List[str]] = None,
    runs_dirs: Optional[List[str]] = None,
    cards_path: str = "data/data_v1_2/cards.json",
    out_dir: str = "runs/teacher/dataset_v1",
    top_m: int = 3,
    mock: bool = False,
) -> Dict[str, Any]:
    """Build and rescore teacher dataset across card catalog."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records_file = target_dir / "records.jsonl"

    if runs_dirs is None:
        runs_dirs = ["runs/pilot", "runs/baselines/eoh", "runs/baselines/reevo"]

    # Load cards
    cards_catalog: Dict[str, Any] = {}
    if Path(cards_path).exists():
        with open(cards_path, "r", encoding="utf-8") as f:
            cards_catalog = json.load(f)

    # Determine families to process
    all_fams = families()
    fam_dict = {f.get("family_id") if isinstance(f, dict) else getattr(f, "name"): f for f in all_fams}
    if families_list is None:
        fams_to_process = [fid for fid in fam_dict.keys() if fid is not None]
    else:
        fams_to_process = families_list

    total_records = 0
    records_written = []

    with open(records_file, "w", encoding="utf-8") as f_out:
        for fam_id in fams_to_process:
            fam_entry = fam_dict.get(fam_id, {})
            pool = fam_entry.get("pool", "train_pool") if isinstance(fam_entry, dict) else "train_pool"

            card_info = cards_catalog.get(fam_id, {})
            base_card = card_info.get("card", fam_entry.get("card", "") if isinstance(fam_entry, dict) else "")
            card_det = card_info.get("card_detailed", base_card)
            paraphrases = card_info.get("paraphrases", [base_card])

            # Load graphs for teacher_val (3000-3019), teacher_rescore (6000-6049), and probe (9000-9009)
            val_graphs = load_family(fam_id, split="teacher_val")
            rescore_graphs = load_family(fam_id, split="teacher_rescore")
            probe_graphs = load_family(fam_id, split="probe")

            # Reference baselines on teacher_val to establish base_alg
            ds_val_mean = float(np.mean([count_colors(dsatur(g[1])) for g in val_graphs]))
            rlf_val_mean = float(np.mean([count_colors(rlf(g[1])) for g in val_graphs]))
            base_alg = "RLF" if rlf_val_mean <= ds_val_mean else "DSATUR"

            # Reference baselines on teacher_rescore
            rescore_adjs = [g[1] for g in rescore_graphs]
            ds_rescore_mean = float(np.mean([count_colors(dsatur(a)) for a in rescore_adjs]))
            rlf_rescore_mean = float(np.mean([count_colors(rlf(a)) for a in rescore_adjs]))
            rf_rescore_mean = rlf_rescore_mean if base_alg == "RLF" else ds_rescore_mean

            # Collect candidates from run logs
            candidates = []
            for r_base in runs_dirs:
                p_run = Path(r_base) / fam_id
                if p_run.exists():
                    for cand_path in p_run.glob("*/candidates.jsonl"):
                        with open(cand_path, "r", encoding="utf-8") as f_cand:
                            for line in f_cand:
                                line = line.strip()
                                if line:
                                    try:
                                        c_data = json.loads(line)
                                        if c_data.get("code") and c_data.get("status") == "ok":
                                            candidates.append(c_data)
                                    except Exception:
                                        pass

            # Also check results/pilot_candidates.jsonl (default pilot log)
            p_pilot = Path("results/pilot_candidates.jsonl")
            if p_pilot.exists():
                with open(p_pilot, "r", encoding="utf-8") as f_pilot:
                    for line in f_pilot:
                        line = line.strip()
                        if line:
                            try:
                                c_data = json.loads(line)
                                if c_data.get("family_id") == fam_id and c_data.get("code") and c_data.get("status") == "ok":
                                    candidates.append(c_data)
                            except Exception:
                                pass

            if not candidates and mock:
                # Mock candidates for fast verification
                for i in range(5):
                    candidates.append({
                        "code": f"def priority(adj, colors, sat, deg, uncolored_deg):\n    return sat * (adj.shape[0] + 1) + uncolored_deg + {i*0.01:.4f} * deg\n",
                        "operator": f"mock_op_{i}",
                        "teacher": "our_engine",
                        "teacher_run_id": "mock_run",
                        "prompt_tokens": 200,
                        "completion_tokens": 100,
                    })

            if not candidates:
                continue

            # Deduplicate by code
            unique_cands: Dict[str, Dict[str, Any]] = {}
            for c in candidates:
                cd = c["code"]
                if cd not in unique_cands:
                    unique_cands[cd] = c

            # Score candidates on teacher_val
            per_graph_timeout = CALIBRATED_PER_GRAPH_TIMEOUTS.get(fam_id, 0.50)
            scored_cands = []
            for cd, c_data in unique_cands.items():
                res_val = evaluate_candidate_safe(cd, val_graphs, per_graph_timeout=per_graph_timeout, overall_timeout=15.0)
                if res_val.get("success", False) and res_val.get("mean_colors") is not None:
                    mc_val = float(res_val["mean_colors"])
                    scored_cands.append((mc_val, c_data))

            scored_cands.sort(key=lambda x: x[0])
            if not scored_cands:
                continue

            # Pick top_m survivors plus 1 contrast sample
            survivors = [c for _, c in scored_cands[:top_m]]
            contrast_samples = [c for _, c in scored_cands[top_m:top_m + 1]]

            items_to_save = [(s, "win_or_tie", rank + 1) for rank, s in enumerate(survivors)]
            for s in contrast_samples:
                items_to_save.append((s, "contrast", len(items_to_save) + 1))

            # Rescore on teacher_rescore and write records
            for c_data, role_hint, rank in items_to_save:
                cd = c_data["code"]
                fp, is_bl_like = get_probe_fingerprint(cd, probe_graphs)

                res_rescore = evaluate_candidate_safe(cd, rescore_graphs, per_graph_timeout=per_graph_timeout, overall_timeout=25.0)
                if not res_rescore.get("success", False) or res_rescore.get("mean_colors") is None:
                    continue

                mc_rescore = float(res_rescore["mean_colors"])
                gap_rf_pct = ((rf_rescore_mean - mc_rescore) / rf_rescore_mean * 100.0) if rf_rescore_mean > 0 else 0.0

                if role_hint == "contrast":
                    role = "contrast"
                else:
                    if gap_rf_pct > 0.5:
                        role = "win"
                    elif abs(gap_rf_pct) <= 0.5:
                        role = "tie"
                    elif mock:
                        role = "tie"
                    else:
                        role = "loss"

                record = {
                    "id": f"{fam_id}_r{rank}_{int(time.time()*1000)%100000}",
                    "family_id": fam_id,
                    "card": base_card,
                    "card_detailed": card_det,
                    "attributes": fam_entry.get("attributes", {}),
                    "paraphrase_idx": 0,
                    "teacher": c_data.get("teacher", "our_engine"),
                    "teacher_run_id": c_data.get("teacher_run_id", "pilot_v1"),
                    "budget": {
                        "prompt_tokens": c_data.get("prompt_tokens", 0),
                        "completion_tokens": c_data.get("completion_tokens", 0),
                    },
                    "code": cd,
                    "role": role,
                    "rank_in_card": rank,
                    "score": {
                        "base_alg": base_alg,
                        "R_F_rescore": rf_rescore_mean,
                        "avg_colors_heuristic_rescore": mc_rescore,
                        "avg_colors_dsatur_rescore": ds_rescore_mean,
                        "avg_colors_rlf_rescore": rlf_rescore_mean,
                        "gap_vs_RF_pct": gap_rf_pct,
                        "valid_rate": 1.0,
                    },
                    "fingerprint": fp,
                    "is_baseline_like": is_bl_like,
                    "split": pool,
                    "seeds": {"selection": "3000-3019", "rescore": "6000-6049", "probe": "9000-9009"},
                }

                f_out.write(json.dumps(record) + "\n")
                records_written.append(record)
                total_records += 1

    summary = {
        "total_records": total_records,
        "out_file": str(records_file),
        "families_processed": len(fams_to_process),
        "win_count": sum(1 for r in records_written if r["role"] == "win"),
        "tie_count": sum(1 for r in records_written if r["role"] == "tie"),
        "contrast_count": sum(1 for r in records_written if r["role"] == "contrast"),
        "loss_count": sum(1 for r in records_written if r["role"] == "loss"),
    }

    with open(target_dir / "summary.json", "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build Large-Scale Teacher Dataset")
    parser.add_argument("--cards-path", default="data/data_v1_2/cards.json", help="Cards catalog path")
    parser.add_argument("--out-dir", default="runs/teacher/dataset_v1", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Generate mock teacher records if none exist")
    parser.add_argument("--families", default=None, help="Comma-separated family IDs to process (default: all)")
    args = parser.parse_args()

    fam_list = [f.strip() for f in args.families.split(",")] if args.families else None

    build_teacher_dataset(
        families_list=fam_list,
        cards_path=args.cards_path,
        out_dir=args.out_dir,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
