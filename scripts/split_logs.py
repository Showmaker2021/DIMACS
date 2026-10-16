"""Script to split pilot_candidates.jsonl into dedicated log files:
1. results/mock_candidates.jsonl (mock runs)
2. results/smoke_gemini_lite.jsonl (Gemini smoke runs)

Enriches records with run_id, model, and is_mock flags.
"""

import json
from pathlib import Path


def split_logs():
    src_path = Path("results/pilot_candidates.jsonl")
    if not src_path.exists():
        print("Source file results/pilot_candidates.jsonl not found.")
        return

    with open(src_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    mock_records = []
    gemini_records = []

    # Assign distinct run_ids based on model and iteration resets
    mock_run_idx = 1
    gemini_run_idx = 1

    last_mock_iter = 0
    last_gemini_iter = 0

    for line in raw_lines:
        rec = json.loads(line)
        m = rec.get("model", "")
        it = rec.get("iteration", 1)

        if m == "mock":
            if it <= last_mock_iter:
                mock_run_idx += 1
            last_mock_iter = it
            rec["run_id"] = f"mock_run_{mock_run_idx}"
            rec["is_mock"] = True
            mock_records.append(rec)
        elif "gemini" in m:
            if it <= last_gemini_iter:
                gemini_run_idx += 1
            last_gemini_iter = it
            rec["run_id"] = f"gemini_lite_run_{gemini_run_idx}"
            rec["is_mock"] = False
            # Clean any accidental key presence if any
            gemini_records.append(rec)

    # Write results/mock_candidates.jsonl
    out_mock = Path("results/mock_candidates.jsonl")
    with open(out_mock, "w", encoding="utf-8") as f:
        for r in mock_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(mock_records)} mock records to {out_mock}")

    # Write results/smoke_gemini_lite.jsonl
    # Notice: the user specifically requested the 20-call Gemini smoke run for smoke_gemini_lite.jsonl
    # Let's write all gemini records, with the main 20-call run clearly marked as run_id 'gemini_lite_smoke_20'
    out_gemini = Path("results/smoke_gemini_lite.jsonl")
    with open(out_gemini, "w", encoding="utf-8") as f:
        for r in gemini_records:
            # If it belongs to the 20-call run (run 3 in sequence, or length 20)
            if r["run_id"] == f"gemini_lite_run_{gemini_run_idx}":
                r["run_id"] = "gemini_lite_smoke_20"
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(gemini_records)} Gemini records to {out_gemini}")


if __name__ == "__main__":
    split_logs()
