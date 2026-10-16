"""Smoke Test Script for LLM Graph Coloring Evolution.

Features:
- Exactly 20 calls ceiling (default).
- ONLY VAL evaluation (NO test graphs loaded/evaluated).
- NO LOCK: Designed for fast repeated testing and connectivity/sandbox validation.
- Two-level seed clone detection (absolute & equivalent).
- Uses shared evolutionary engine from gate.evolved.loop.
- Billing dashboard reconciliation box and 3 sample candidate codes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import argparse
import numpy as np

from gate.dataset import load_family, families
from gate.evolved.loop import (
    load_config,
    call_llm_with_retry,
    extract_code_block,
    deduplicate_population_by_behavior,
    extract_behavior_summary,
    run_evolution,
)


def run_smoke(
    family_id: str = "kpart_k10_p0.3_n100-200",
    max_calls: int = 20,
    max_tokens: int = 60_000,
    max_dollars: float = 0.20,
    mock: bool = False,
    log_path: str = "results/smoke_gemini_lite.jsonl",
    run_id: str = "gemini_lite_smoke_v2",
):
    # Execute through shared evolution engine
    summary = run_evolution(
        family_id=family_id,
        max_calls=max_calls,
        max_tokens=max_tokens,
        max_dollars=max_dollars,
        mock=mock,
        log_path=log_path,
        run_id=run_id,
        selection_dir=None,  # smoke does not write frozen selection
        selection_split="val",
        explore_ratio=0.40,
        pacing_delay=5.0,
        is_smoke=True,
    )

    # Load pricing for reporting
    cfg_model, pricing = load_config()
    model = "mock" if mock else cfg_model
    p_info = pricing.get(model, {"input_per_million": 0.14, "output_per_million": 0.28})
    p_in = p_info["input_per_million"]
    p_out = p_info["output_per_million"]

    print("\n" + "=" * 80)
    print("SMOKE RUN SUMMARY & RECONCILIATION")
    print("=" * 80)
    print(f"  * Configured Model:      {cfg_model} (Active: {model})")
    print(f"  * Effective Pricing:     Input ${p_in:.4f}/1M | Output ${p_out:.4f}/1M")
    print(f"  * Total Calls Made:      {summary['total_calls']} (Ceiling: {max_calls})")
    print(f"  * Valid Candidates:      {summary['valid_count']}")
    print(f"  * Rejected Candidates:   {summary['rejected_count']}")
    print(f"  * Absolute Seed Clones:  {summary['abs_clone_count']}")
    print(f"  * Equivalent Clones:     {summary['equiv_clone_count']}")
    print(f"  * Distinct Behaviors:    {summary['distinct_behavior_count']}")
    print(f"  * Prompt Tokens:         {summary['total_prompt_tokens']}")
    print(f"  * Output Tokens:         {summary['total_completion_tokens']}")
    print(f"  * Total Tokens:          {summary['total_prompt_tokens'] + summary['total_completion_tokens']} (Ceiling: {max_tokens})")
    print(f"  * Real Dollar Cost:      ${summary['total_cost']:.5f}")
    print(f"  * Safety 2x Cost:        ${summary['total_cost'] * 2.0:.5f} (Ceiling: ${max_dollars:.2f})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Smoke Test Runner (20 Calls, VAL Only, No Lock)")
    parser.add_argument("--family", default="kpart_k10_p0.3_n100-200", help="Target family ID")
    parser.add_argument("--max-calls", type=int, default=20, help="Max API calls ceiling (default: 20)")
    parser.add_argument("--max-tokens", type=int, default=60_000, help="Max tokens ceiling (default: 60,000)")
    parser.add_argument("--max-dollars", type=float, default=0.20, help="Max dollar budget ceiling (default: $0.20)")
    parser.add_argument("--mock", action="store_true", help="Run with mock LLM (no real API call)")
    parser.add_argument("--log-path", default="results/smoke_gemini_lite.jsonl", help="Log path (default: results/smoke_gemini_lite.jsonl)")
    parser.add_argument("--run-id", default="gemini_lite_smoke_v2", help="Run identifier (default: gemini_lite_smoke_v2)")
    args = parser.parse_args()

    run_smoke(
        family_id=args.family,
        max_calls=args.max_calls,
        max_tokens=args.max_tokens,
        max_dollars=args.max_dollars,
        mock=args.mock,
        log_path=args.log_path,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
