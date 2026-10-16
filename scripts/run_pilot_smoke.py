"""Smoke test script for real LLM evolution.

Runs a single pilot family with a strict ceiling of 20 API calls and hard dollar ceiling.
Extracts token usage directly from API response `usage` fields.
Does NOT proceed to Phase 2.
Requires explicit API key in environment or runs with --mock flag for verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import argparse
import numpy as np

from gate.dataset import load_family, families
from gate.evolved.sandbox import evaluate_candidate_safe
from gate.evolved.task import (
    CandidateLogger,
    CALIBRATED_PER_GRAPH_TIMEOUTS,
    SEED_CODE_DSATUR,
    SEED_CODE_RLF,
)
from gate.evolved.prompts import SYSTEM_PROMPT, build_user_prompt
from gate.tune import compute_paired_bootstrap_ci
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


# Pricing per 1M tokens ($)
PRICING = {
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "mock": {"input": 0.14, "output": 0.28},
}


def call_llm_api(
    model: str,
    system_prompt: str,
    user_prompt: str,
    mock: bool = False,
) -> tuple[str, dict]:
    """Call LLM API and return (extracted_code, usage_dict)."""
    if mock:
        # Realistic mock program variant
        code = """def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    unc = (colors == -1)
    if not np.any(unc):
        return np.zeros(n, dtype=float)
    cur_color = int(np.max(colors))
    scores = np.zeros(n, dtype=float)
    if cur_color >= 0:
        in_cur_class = (colors == cur_color)
        cand = unc & (~np.any(adj[:, in_cur_class], axis=1))
        excl = unc & (~cand)
        if np.any(cand):
            for u in np.flatnonzero(cand):
                adj_u = adj[u]
                ne = np.sum(adj_u & excl)
                nc = np.sum(adj_u & cand)
                scores[u] = float(ne * (n + 1) - nc)
            return scores
    scores[unc] = uncolored_deg[unc].astype(float)
    return scores
"""
        usage = {"prompt_tokens": 1850, "completion_tokens": 320}
        return code, usage

    # Real API dispatch
    if "deepseek" in model.lower():
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is missing. Set it before running real smoke test.")
        
        import urllib.request
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        raw_text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
        code = extract_code_block(raw_text)
        return code, usage

    elif "claude" in model.lower():
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is missing. Set it before running real smoke test.")
        
        import urllib.request
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        raw_text = data["content"][0]["text"]
        usage = {
            "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
        }
        code = extract_code_block(raw_text)
        return code, usage

    else:
        raise ValueError(f"Unsupported model: {model}")


def extract_code_block(text: str) -> str:
    """Extract code from ```python ... ``` block or raw text."""
    if "```python" in text:
        parts = text.split("```python")
        code = parts[1].split("```")[0]
        return code.strip()
    elif "```" in text:
        parts = text.split("```")
        code = parts[1].split("```")[0]
        return code.strip()
    return text.strip()


def run_smoke_test(
    family_id: str = "kpart_k10_p0.3_n100-200",
    max_calls: int = 20,
    max_dollars: float = 0.20,
    model: str = "deepseek-chat",
    mock: bool = False,
    log_path: str = "results/pilot_candidates.jsonl",
):
    print("=" * 80)
    print(f"PILOT SMOKE TEST: 1 Family ({family_id}), Max {max_calls} Calls, Ceiling ${max_dollars:.2f}")
    print(f"Model: {model} | Mock mode: {mock}")
    print("=" * 80)

    # 1. Load family metadata & card
    fam_lookup = {f["family_id"]: f for f in families()}
    if family_id not in fam_lookup:
        raise ValueError(f"Family '{family_id}' not found in families.json")
    family_meta = fam_lookup[family_id]
    card = family_meta.get("cards", {}).get("card", "")
    print(f"Family Card: \"{card}\"")

    # 2. Load splits
    train_graphs = load_family(family_id, "baseline_train")
    val_graphs = load_family(family_id, "val")
    test_graphs = load_family(family_id, "test")

    # Baselines on train
    c_rlf_train = float(np.mean([count_colors(rlf(g[1])) for g in train_graphs]))
    c_ds_train = float(np.mean([count_colors(dsatur(g[1])) for g in train_graphs]))
    r_train = min(c_rlf_train, c_ds_train)
    print(f"Train Baseline: R(F) = {r_train:.3f} (DSATUR = {c_ds_train:.3f}, RLF = {c_rlf_train:.3f})")

    c_rlf_val = float(np.mean([count_colors(rlf(g[1])) for g in val_graphs]))
    c_ds_val = float(np.mean([count_colors(dsatur(g[1])) for g in val_graphs]))
    r_val = min(c_rlf_val, c_ds_val)

    test_adjs = [g[1] for g in test_graphs]
    if c_rlf_val <= c_ds_val:
        r_test_arr = np.array([count_colors(rlf(a)) for a in test_adjs], dtype=float)
    else:
        r_test_arr = np.array([count_colors(dsatur(a)) for a in test_adjs], dtype=float)

    per_graph_timeout = CALIBRATED_PER_GRAPH_TIMEOUTS.get(family_id, 0.50)
    logger = CandidateLogger(log_path)

    # 3. Initialize population with DSATUR and RLF
    price_info = PRICING.get(model if not mock else "mock", PRICING["deepseek-chat"])

    # Initial population
    population = [
        {"program": SEED_CODE_DSATUR, "fitness": 0.0, "mean_colors": c_ds_train},
        {"program": SEED_CODE_RLF, "fitness": (r_train - c_rlf_train), "mean_colors": c_rlf_train},
    ]

    total_calls = 0
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    print(f"\nStarting smoke loop (Hard call ceiling: {max_calls}, Hard budget ceiling: ${max_dollars:.2f})...")

    while total_calls < max_calls and total_cost < max_dollars:
        total_calls += 1
        operator = "crossover" if total_calls % 2 == 1 else "refine"

        # Select parents
        valid_pop = [p for p in population if p.get("fitness", -100) > -50]
        valid_pop.sort(key=lambda x: x["fitness"], reverse=True)
        parents = valid_pop[:2] if len(valid_pop) >= 2 else valid_pop

        user_prompt = build_user_prompt(family_id, card, r_train, parents, operator=operator)

        try:
            code, usage = call_llm_api(model, SYSTEM_PROMPT, user_prompt, mock=mock)
        except Exception as e:
            print(f"  [Call {total_calls}] API Error: {e}")
            break

        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        cost = (p_tok * price_info["input"] + c_tok * price_info["output"]) / 1_000_000.0

        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        total_cost += cost

        # Evaluate candidate in hardened sandbox
        res = evaluate_candidate_safe(
            code,
            train_graphs,
            per_graph_timeout=per_graph_timeout,
            overall_timeout=15.0,
        )

        if res["success"]:
            mean_c = res["mean_colors"]
            fitness = (r_train - mean_c) + 1e-4 * res.get("mean_tie_break", 0.0)
            err = None
            population.append({"program": code, "fitness": fitness, "mean_colors": mean_c})
        else:
            mean_c = None
            fitness = -100.0
            err = res["error"]

        rec = {
            "family_id": family_id,
            "card": card,
            "iteration": total_calls,
            "model": model,
            "operator": operator,
            "program": code,
            "fitness": fitness,
            "mean_colors": mean_c,
            "r_train": r_train,
            "error": err,
            "tokens": {"prompt": p_tok, "completion": c_tok},
            "call_cost": cost,
            "total_cost": total_cost,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        logger.log(rec)

        status_str = f"Mean C={mean_c:.2f}, Fit={fitness:+.4f}" if res["success"] else f"REJECTED: {err[:40]}"
        print(f"  [Call {total_calls:02d}/{max_calls}] {status_str} | Toks=({p_tok}+{c_tok}) | Step=${cost:.5f} | Cumul=${total_cost:.4f}")

    print("\n" + "-" * 60)
    print("SMOKE TEST CALL LOOP COMPLETED")
    print(f"Total API Calls: {total_calls} (Ceiling: {max_calls})")
    print(f"Total Tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion = {total_prompt_tokens + total_completion_tokens}")
    print(f"Total Expenditure: ${total_cost:.5f} (Ceiling: ${max_dollars:.2f})")
    print("-" * 60)

    # 4. Final heuristic selection protocol
    valid_candidates = [p for p in population if p["fitness"] > -50]
    valid_candidates.sort(key=lambda x: x["fitness"], reverse=True)
    top_cands = valid_candidates[:10]

    print(f"\nEvaluating top-{len(top_cands)} candidates on VAL graphs (20 graphs)...")
    best_cand = None
    best_val_mean = float("inf")
    for cand in top_cands:
        val_res = evaluate_candidate_safe(cand["program"], val_graphs, per_graph_timeout=per_graph_timeout)
        if val_res["success"] and val_res["mean_colors"] < best_val_mean:
            best_val_mean = val_res["mean_colors"]
            best_cand = cand

    if best_cand is None:
        print(f"[{family_id}] No valid candidate succeeded on VAL. Fallback to R(F).")
        test_counts = r_test_arr
    else:
        print(f"Best on VAL: Mean colors = {best_val_mean:.3f} (vs R(F) Val = {r_val:.3f})")
        # 5. Measure selected candidate on TEST (50 graphs)
        test_res = evaluate_candidate_safe(best_cand["program"], test_graphs, per_graph_timeout=per_graph_timeout)
        test_counts = np.array(test_res["color_counts"], dtype=float)
    gap_c, c_lo, c_hi, gap_p, p_lo, p_hi = compute_paired_bootstrap_ci(r_test_arr, test_counts)

    print("\n" + "=" * 80)
    print(f"FINAL TEST RESULT: Selected Heuristic on 50 TEST Graphs")
    print(f"Selected Mean Colors: {np.mean(test_counts):.3f} vs R(F) = {np.mean(r_test_arr):.3f}")
    print(f"Paired Gap in Colors: {gap_c:+.3f} colors [95% CI: {c_lo:+.3f}, {c_hi:+.3f}]")
    print(f"Paired Gap in %:      {gap_p:+.2f}% [95% CI: {p_lo:+.2f}%, {p_hi:+.2f}%]")
    print("=" * 80)
    print("SMOKE TEST COMPLETE. Stopped before Phase 2 as required.")


def main():
    parser = argparse.ArgumentParser(description="Pilot Smoke Test for LLM Graph Coloring Evolution")
    parser.add_argument("--family", default="kpart_k10_p0.3_n100-200", help="Target family ID")
    parser.add_argument("--max-calls", type=int, default=20, help="Max API calls ceiling")
    parser.add_argument("--max-dollars", type=float, default=0.20, help="Max dollar budget ceiling")
    parser.add_argument("--model", default="deepseek-chat", help="Model name")
    parser.add_argument("--mock", action="store_true", help="Run with mock LLM responses (no real API call)")
    parser.add_argument("--log-path", default="results/pilot_candidates.jsonl", help="Log path")
    args = parser.parse_args()

    run_smoke_test(
        family_id=args.family,
        max_calls=args.max_calls,
        max_dollars=args.max_dollars,
        model=args.model,
        mock=args.mock,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
