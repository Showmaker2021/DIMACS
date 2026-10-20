"""End-to-End Dataset Generation Pipeline Runner.

Automates the complete dataset generation workflow:
1. Splits & Cards generation (Task 8.1, Task 8.4)
2. Teacher evolution on selected families (Task 6.1, Task 8.2)
3. Teacher data aggregation, probe fingerprint dedup & rescoring (Task 8.2)
4. SFT dataset filtering, inverse-rank weighting, and train/val partitioning (Task 8.3, Task 10.1)

Usage:
    # 1. Quick test run with mock LLM (100% offline, takes ~15 seconds):
    python scripts/run_dataset_pipeline.py --mock

    # 2. Production run with Google Gemini Free Tier:
    # Windows: $env:GEMINI_API_KEY = "AIzaSy..."
    # Linux:   export GEMINI_API_KEY="AIzaSy..."
    python scripts/run_dataset_pipeline.py --max-calls-per-family 50
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FAMILIES = [
    "kpart_k10_p0.3_n100-200",
    "gnp_d0.30_n100-200",
    "sbm_k10_in0.5_out0.05_n200-300",
    "ws_k10_p0.1_n100-200",
    "interval_L0.10_n100-200",
    "ba_m12_n200-300",
]


def run_cmd(cmd: List[str], desc: str) -> None:
    print(f"\n{'='*70}\n[STEP] {desc}\nCMD: {' '.join(cmd)}\n{'='*70}")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print(f"\n[ERROR] Step failed with returncode {res.returncode}: {desc}", file=sys.stderr)
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Automated SFT Dataset Generation Pipeline")
    parser.add_argument("--mock", action="store_true", help="Run with mock LLM generator (zero API calls, fast verification)")
    parser.add_argument("--max-calls-per-family", type=int, default=50, help="Max LLM calls per family (default: 50)")
    parser.add_argument("--families", type=str, default=None, help="Comma-separated family IDs to evolve (default: 6 pilot families)")
    parser.add_argument("--out-dir", type=str, default="data/sft", help="Final SFT output directory (default: data/sft)")
    parser.add_argument("--pacing-delay", type=float, default=4.0, help="Pacing delay (seconds) between calls (default: 4.0s for Gemini 15 RPM)")

    args = parser.parse_args()

    fams = [f.strip() for f in args.families.split(",")] if args.families else DEFAULT_FAMILIES
    python_exe = sys.executable

    print("\n" + "#"*70)
    print("# GRAPH COLORING HEURISTICS - SFT DATASET GENERATION PIPELINE")
    print(f"# Mode: {'MOCK (Offline)' if args.mock else 'LIVE API (Gemini / Configured LLM)'}")
    print(f"# Target Families ({len(fams)}): {', '.join(fams)}")
    print(f"# Calls per family: {args.max_calls_per_family}")
    print(f"# Output Directory: {args.out_dir}")
    print("#"*70)

    # 1. Generate splits
    run_cmd(
        [python_exe, "scripts/build_splits.py"],
        "1/5 Generating Dataset v1.2 Splits (data/data_v1_2/splits.json)"
    )

    # 2. Generate cards & paraphrases
    run_cmd(
        [python_exe, "scripts/make_cards.py"],
        "2/5 Generating Graph Cards & Paraphrases (data/data_v1_2/cards.json)"
    )

    # 3. Evolve heuristics for selected families
    print(f"\n{'='*70}\n[STEP] 3/5 Evolving Teacher Heuristics across {len(fams)} families\n{'='*70}")
    for idx, fam in enumerate(fams, start=1):
        print(f"\n>>> [{idx}/{len(fams)}] Evolving family: {fam} ...")
        cmd = [
            python_exe, "scripts/pilot_evolve.py",
            "--family", fam,
            "--max-calls", str(args.max_calls_per_family),
            "--pacing-delay", str(args.pacing_delay),
            "--run-id", f"gen_{fam}",
        ]
        if args.mock:
            cmd.append("--mock")
        run_cmd(cmd, f"Evolution for {fam}")

    # 4. Harvest Teacher Data & Rescore
    cmd_teacher = [
        python_exe, "scripts/build_teacher_data.py",
        "--cards-path", "data/data_v1_2/cards.json",
        "--out-dir", "runs/teacher/dataset_v1",
        "--families", ",".join(fams),
    ]
    if args.mock:
        cmd_teacher.append("--mock")
    run_cmd(cmd_teacher, "4/5 Harvesting Teacher Data, Fingerprint Dedup & Clean Rescoring")

    # 5. Filter & Build Final SFT Datasets
    run_cmd(
        [
            python_exe, "scripts/filter_sft_data.py",
            "--records-path", "runs/teacher/dataset_v1/records.jsonl",
            "--out-dir", args.out_dir,
            "--val-ratio", "0.15",
        ],
        f"5/5 Filtering & Formatting SFT Datasets ({args.out_dir})"
    )

    print("\n" + "#"*70)
    print("# DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print(f"# Output Files in '{args.out_dir}':")
    for fname in ["train.jsonl", "val.jsonl", "contrast.jsonl"]:
        fpath = Path(args.out_dir) / fname
        if fpath.exists():
            line_count = sum(1 for _ in open(fpath, "r", encoding="utf-8"))
            size_kb = fpath.stat().st_size / 1024
            print(f"#   - {fname:<15}: {line_count:>5} samples ({size_kb:>6.1f} KB)")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
