"""Pilot Evolution Runner for Graph Coloring (Up to 150 Calls, Resume Support, Teacher_Val Selection).

Uses unified evolutionary engine from gate.evolved.loop.
Key Features:
- Up to 150 calls ceiling per family.
- Selection split: teacher_val (seeds 3000-3019), keeping val split pristine.
- Behavioral deduplication: population keeps 1 representative per 20-dim train color vector.
- 40% exploration operator (temp 0.9), 30% crossover (temp 0.7), 30% refine (temp 0.5).
- 5s pacing and 429 exponential backoff.
- Default log: results/pilot_candidates.jsonl (distinct from smoke logs, avoiding reload).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from gate.evolved.loop import run_evolution


def main():
    parser = argparse.ArgumentParser(description="Pilot Evolution Runner for Graph Coloring Heuristics.")
    parser.add_argument("--family", type=str, default="kpart_k10_p0.3_n100-200", help="Target family ID")
    parser.add_argument("--max-calls", type=int, default=150, help="Maximum number of LLM calls")
    parser.add_argument("--max-tokens", type=int, default=450_000, help="Maximum total token ceiling")
    parser.add_argument("--max-dollars", type=float, default=1.50, help="Maximum dollar ceiling (with 2x safety)")
    parser.add_argument("--mock", action="store_true", help="Run with mock LLM generator (zero API calls)")
    parser.add_argument("--log-path", type=str, default="results/pilot_candidates.jsonl", help="Log file path")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for grouping/resuming (defaults to pilot_{family_id})")
    parser.add_argument("--selection-dir", type=str, default="results/selections", help="Directory for frozen selection JSON")
    parser.add_argument("--selection-split", type=str, default="teacher_val", help="Split used for final model selection (teacher_val)")
    parser.add_argument("--explore-ratio", type=float, default=0.40, help="Fraction of calls allocated to exploration operator")
    parser.add_argument("--pacing-delay", type=float, default=5.0, help="Pacing delay (seconds) between calls")

    args = parser.parse_args()

    run_evolution(
        family_id=args.family,
        max_calls=args.max_calls,
        max_tokens=args.max_tokens,
        max_dollars=args.max_dollars,
        mock=args.mock,
        log_path=args.log_path,
        run_id=args.run_id,
        selection_dir=args.selection_dir,
        selection_split=args.selection_split,
        explore_ratio=args.explore_ratio,
        pacing_delay=args.pacing_delay,
        is_smoke=False,
    )


if __name__ == "__main__":
    main()
