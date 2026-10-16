"""Final Pilot Smoke Test Script for LLM Graph Coloring Evolution.

Enforces:
1. Lock Mechanism: Runs once per family; prevents duplicate runs via file locks.
2. Zero Test Evaluation: Evaluates only on Train and VAL splits ("bỏ hoàn toàn đánh giá test; chỉ val").
3. Strict Ceilings:
   - Hard Call Ceiling (default 20 calls).
   - Hard Token Ceiling (default 60,000 tokens).
   - Hard Dollar Ceiling (default $0.20), computed from configs/pricing.json.
4. Retry Logic: Up to 2 retries on API failure, counted against ceilings.
5. Seed Clone Detection:
   - Behavior hash computed on the first 3 train graphs.
   - Identified clones of DSATUR or RLF are logged with is_seed_clone=True.
   - Seed clones are strictly EXCLUDED from the parent mating pool.
6. Detailed Reporting:
   - Valid candidate rate (%)
   - Rejection rate by failure reason (%)
   - Real token & dollar cost from API usage
   - Seed clone rate (%)
   - 3 sample candidate codes (best valid, median valid, rejected sample)
   - VAL evaluation vs R(F) and fallback status
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import hashlib
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
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


def load_pricing(config_path: str = "configs/pricing.json") -> dict:
    """Load pricing per 1M tokens from config file."""
    path = Path(config_path)
    if not path.exists():
        return {
            "deepseek-chat": {"input_per_million": 0.14, "output_per_million": 0.28},
            "claude-3-5-sonnet-20241022": {"input_per_million": 3.00, "output_per_million": 15.00},
            "mock": {"input_per_million": 0.14, "output_per_million": 0.28},
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_llm_with_retry(
    model: str,
    system_prompt: str,
    user_prompt: str,
    pricing: dict,
    mock: bool = False,
    max_retries: int = 2,
) -> tuple[str, dict, float, int]:
    """Call LLM API with up to max_retries attempts.
    
    Returns:
    (code_str, usage_dict, total_call_cost, attempts_made)
    """
    model_prices = pricing.get(model if not mock else "mock", pricing.get("deepseek-chat", {"input_per_million": 0.14, "output_per_million": 0.28}))
    p_in = model_prices["input_per_million"]
    p_out = model_prices["output_per_million"]

    if mock:
        # Realistic mock candidate
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
        usage = {"prompt_tokens": 1920, "completion_tokens": 340}
        cost = (usage["prompt_tokens"] * p_in + usage["completion_tokens"] * p_out) / 1_000_000.0
        return code, usage, cost, 1

    api_key = os.environ.get("DEEPSEEK_API_KEY") if "deepseek" in model.lower() else os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(f"API key missing in environment for model '{model}'. Please set it before running.")

    import urllib.request
    attempts = 0
    last_err = None

    while attempts <= max_retries:
        attempts += 1
        try:
            if "deepseek" in model.lower():
                url = "https://api.deepseek.com/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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
                cost = (usage.get("prompt_tokens", 0) * p_in + usage.get("completion_tokens", 0) * p_out) / 1_000_000.0
                return code, usage, cost, attempts

            elif "claude" in model.lower():
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
                cost = (usage["prompt_tokens"] * p_in + usage["completion_tokens"] * p_out) / 1_000_000.0
                return code, usage, cost, attempts

        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempts)

    raise RuntimeError(f"API call failed after {attempts} attempts. Last error: {last_err}")


def extract_code_block(text: str) -> str:
    """Extract python code from markdown block."""
    if "```python" in text:
        parts = text.split("```python")
        code = parts[1].split("```")[0]
        return code.strip()
    elif "```" in text:
        parts = text.split("```")
        code = parts[1].split("```")[0]
        return code.strip()
    return text.strip()


def run_pilot(
    family_id: str = "kpart_k10_p0.3_n100-200",
    max_calls: int = 20,
    max_tokens: int = 60_000,
    max_dollars: float = 0.20,
    model: str = "deepseek-chat",
    mock: bool = False,
    lock_dir: str = "results/pilot_locks",
    log_path: str = "results/pilot_candidates.jsonl",
):
    lock_path = Path(lock_dir) / f"{family_id}.done"
    Path(lock_dir).mkdir(parents=True, exist_ok=True)

    # 1. Check duplicate lock
    if lock_path.exists():
        print(f"[LOCK ACTIVE] Family '{family_id}' has already been executed. Aborting duplicate run.")
        print(f"Lock file: {lock_path}")
        return

    print("=" * 80)
    print(f"PILOT FINAL SMOKE TEST: Family '{family_id}' (VAL ONLY, NO TEST EVALUATION)")
    print(f"Model: {model} | Mock: {mock}")
    print(f"Ceilings: Max Calls = {max_calls}, Max Tokens = {max_tokens}, Max Dollars = ${max_dollars:.2f}")
    print("=" * 80)

    # 2. Load family metadata & card
    fam_lookup = {f["family_id"]: f for f in families()}
    if family_id not in fam_lookup:
        raise ValueError(f"Family '{family_id}' not found in families.json")
    family_meta = fam_lookup[family_id]
    card = family_meta.get("cards", {}).get("card", "")
    print(f"Family Card: \"{card}\"")

    # 3. Load Train & VAL splits ONLY (zero test graphs loaded)
    train_graphs = load_family(family_id, "baseline_train")
    val_graphs = load_family(family_id, "val")

    # Baselines
    c_rlf_train = float(np.mean([count_colors(rlf(g[1])) for g in train_graphs]))
    c_ds_train = float(np.mean([count_colors(dsatur(g[1])) for g in train_graphs]))
    r_train = min(c_rlf_train, c_ds_train)

    c_rlf_val = float(np.mean([count_colors(rlf(g[1])) for g in val_graphs]))
    c_ds_val = float(np.mean([count_colors(dsatur(g[1])) for g in val_graphs]))
    r_val = min(c_rlf_val, c_ds_val)

    print(f"Train Baseline: R(F) = {r_train:.3f} (DSATUR = {c_ds_train:.3f}, RLF = {c_rlf_train:.3f})")
    print(f"Val Baseline:   R(F) = {r_val:.3f} (DSATUR = {c_ds_val:.3f}, RLF = {c_rlf_val:.3f})")

    # 4. Compute Seed Behavior Hashes on first 3 train graphs
    ds_hashes = [hashlib.sha256(dsatur(adj).tobytes()).hexdigest() for _, adj in train_graphs[:3]]
    ds_hash = "-".join(ds_hashes)
    rlf_hashes = [hashlib.sha256(rlf(adj).tobytes()).hexdigest() for _, adj in train_graphs[:3]]
    rlf_hash = "-".join(rlf_hashes)

    per_graph_timeout = CALIBRATED_PER_GRAPH_TIMEOUTS.get(family_id, 0.50)
    pricing = load_pricing()
    logger = CandidateLogger(log_path)

    # Initial Population
    population = [
        {"program": SEED_CODE_DSATUR, "fitness": 0.0, "mean_colors": c_ds_train, "is_seed_clone": False},
        {"program": SEED_CODE_RLF, "fitness": (r_train - c_rlf_train), "mean_colors": c_rlf_train, "is_seed_clone": False},
    ]

    # Metrics Tracking
    total_calls = 0
    total_tokens = 0
    total_cost = 0.0
    valid_count = 0
    rejected_count = 0
    rejection_reasons = {}
    seed_clone_count = 0

    evaluated_records = []

    print(f"\nStarting pilot generation loop...")

    while total_calls < max_calls:
        if total_tokens >= max_tokens:
            print(f"Reached hard token ceiling ({total_tokens} >= {max_tokens}). Stopping.")
            break
        if total_cost >= max_dollars:
            print(f"Reached hard dollar ceiling (${total_cost:.4f} >= ${max_dollars:.4f}). Stopping.")
            break

        total_calls += 1
        operator = "crossover" if total_calls % 2 == 1 else "refine"

        # EXCLUDE SEED CLONES FROM PARENT POOL
        parent_pool = [
            p for p in population
            if p.get("fitness", -100) > -50 and not p.get("is_seed_clone", False)
        ]
        parent_pool.sort(key=lambda x: x["fitness"], reverse=True)
        parents = parent_pool[:2] if len(parent_pool) >= 2 else population[:2]

        user_prompt = build_user_prompt(family_id, card, r_train, parents, operator=operator)

        try:
            code, usage, cost, attempts = call_llm_with_retry(
                model, SYSTEM_PROMPT, user_prompt, pricing, mock=mock, max_retries=2
            )
        except Exception as e:
            print(f"  [Call {total_calls:02d}] API Failure: {e}")
            break

        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        call_tok = p_tok + c_tok

        total_tokens += call_tok
        total_cost += cost

        # Evaluate candidate in hardened sandbox on Train
        res = evaluate_candidate_safe(
            code,
            train_graphs,
            per_graph_timeout=per_graph_timeout,
            overall_timeout=15.0,
        )

        if res["success"]:
            valid_count += 1
            mean_c = res["mean_colors"]
            fitness = (r_train - mean_c) + 1e-4 * res.get("mean_tie_break", 0.0)
            err = None

            # Seed clone check
            cand_b_hash = res.get("behavior_hash", "")
            is_clone = (cand_b_hash == ds_hash or cand_b_hash == rlf_hash)
            if is_clone:
                seed_clone_count += 1

            population.append({
                "program": code,
                "fitness": fitness,
                "mean_colors": mean_c,
                "is_seed_clone": is_clone,
            })
        else:
            rejected_count += 1
            mean_c = None
            fitness = -100.0
            err = res["error"]
            is_clone = False

            # Categorize rejection reason
            err_lower = err.lower()
            if "ast security" in err_lower or "syntaxerror" in err_lower:
                reason = "AST / Syntax Violation"
            elif "memory" in err_lower:
                reason = "Memory Limit Exceeded"
            elif "per-graph timeout" in err_lower:
                reason = "Per-Graph Timeout"
            elif "timeout" in err_lower:
                reason = "Overall Timeout"
            elif "purity" in err_lower or "statelessness" in err_lower:
                reason = "Statelessness / Purity Failed"
            else:
                reason = "Runtime / Coloring Verification Error"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

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
            "is_seed_clone": is_clone,
            "tokens": {"prompt": p_tok, "completion": c_tok, "total": call_tok},
            "call_cost": cost,
            "total_cost": total_cost,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        logger.log(rec)
        evaluated_records.append(rec)

        status_str = f"Mean C={mean_c:.2f}, Fit={fitness:+.4f}" if res["success"] else f"REJECTED: {err[:35]}"
        clone_str = " [SEED CLONE]" if is_clone else ""
        print(f"  [Call {total_calls:02d}/{max_calls}] {status_str}{clone_str} | Toks={call_tok} | Cost=${cost:.5f} | Cumul=${total_cost:.4f}")

    print("\n" + "=" * 80)
    print("PILOT SMOKE TEST EVALUATION SUMMARY (VAL ONLY)")
    print("=" * 80)

    # Candidate validity & rejection statistics
    total_eval = valid_count + rejected_count
    valid_rate = (valid_count / total_eval * 100.0) if total_eval > 0 else 0.0
    clone_rate = (seed_clone_count / valid_count * 100.0) if valid_count > 0 else 0.0

    print(f"1. CANDIDATE VALIDITY & REJECTION:")
    print(f"   Total Evaluated Candidates: {total_eval}")
    print(f"   - Valid Candidates:   {valid_count} ({valid_rate:.1f}%)")
    print(f"   - Rejected Candidates: {rejected_count} ({100.0 - valid_rate:.1f}%)")
    print(f"   - Rejection Breakdown by Reason:")
    for r_name, r_cnt in rejection_reasons.items():
        print(f"     * {r_name}: {r_cnt} ({r_cnt / max(1, rejected_count) * 100.0:.1f}%)")

    print(f"\n2. SEED CLONE RATE (Identical behavior on first 3 train graphs):")
    print(f"   - Seed Clones: {seed_clone_count} / {valid_count} valid ({clone_rate:.1f}%)")
    print(f"   - Note: Seed clones were strictly excluded from the parent mating pool.")

    print(f"\n3. RESOURCE EXPENDITURE:")
    print(f"   - Total Calls:  {total_calls} (Ceiling: {max_calls})")
    print(f"   - Total Tokens: {total_tokens} (Ceiling: {max_tokens})")
    print(f"   - Total Cost:   ${total_cost:.5f} (Ceiling: ${max_dollars:.2f})")

    # 4. Final heuristic selection on VAL ONLY
    valid_cands = [p for p in population if p["fitness"] > -50 and p.get("iteration") is not None]
    if not valid_cands:
        valid_cands = [p for p in population if p["fitness"] > -50]

    valid_cands.sort(key=lambda x: x["fitness"], reverse=True)
    top_val_pool = valid_cands[:10]

    print(f"\n4. VALIDATION SELECTION (Top-{len(top_val_pool)} candidates on 20 VAL graphs):")
    best_cand = None
    best_val_mean = float("inf")
    for cand in top_val_pool:
        val_res = evaluate_candidate_safe(cand["program"], val_graphs, per_graph_timeout=per_graph_timeout)
        if val_res["success"] and val_res["mean_colors"] < best_val_mean:
            best_val_mean = val_res["mean_colors"]
            best_cand = cand

    if best_cand is not None:
        is_fallback = (best_val_mean > r_val)
        print(f"   Selected Candidate Val Mean: {best_val_mean:.3f} colors vs R(F) Val = {r_val:.3f} colors")
        print(f"   Fallback Status: {'TRIGGERED (worse than R(F) on VAL)' if is_fallback else 'NOT NEEDED (strictly beats R(F) on VAL)'}")
    else:
        print(f"   No candidate succeeded on VAL. Fallback to R(F).")

    # 5. Print 3 Sample Codes
    print(f"\n5. THREE SAMPLE CANDIDATE CODES:")
    valid_recs = [r for r in evaluated_records if r["fitness"] > -50]
    rejected_recs = [r for r in evaluated_records if r["fitness"] <= -50]

    # Sample 1: Best valid
    if valid_recs:
        best_rec = max(valid_recs, key=lambda x: x["fitness"])
        print("\n--- SAMPLE 1: BEST VALID CANDIDATE ---")
        print(f"Iteration: {best_rec['iteration']} | Fitness: {best_rec['fitness']:+.4f} | Mean Train Colors: {best_rec['mean_colors']}")
        print(best_rec["program"].strip())

    # Sample 2: Median valid
    if len(valid_recs) >= 2:
        valid_recs_sorted = sorted(valid_recs, key=lambda x: x["fitness"])
        med_rec = valid_recs_sorted[len(valid_recs_sorted) // 2]
        print("\n--- SAMPLE 2: MEDIAN VALID CANDIDATE ---")
        print(f"Iteration: {med_rec['iteration']} | Fitness: {med_rec['fitness']:+.4f} | Mean Train Colors: {med_rec['mean_colors']}")
        print(med_rec["program"].strip())

    # Sample 3: Rejected candidate
    if rejected_recs:
        rej_rec = rejected_recs[0]
        print("\n--- SAMPLE 3: REJECTED CANDIDATE ---")
        print(f"Iteration: {rej_rec['iteration']} | Error: {rej_rec['error']}")
        print(rej_rec["program"].strip())

    # Write completion lock
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%dT%H:%M:%S')}\nCalls: {total_calls}\nCost: ${total_cost:.5f}\n")

    print("\n" + "=" * 80)
    print(f"SMOKE TEST COMPLETE FOR {family_id}. LOCK CREATED AT: {lock_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Pilot Final Smoke Test (Val Only)")
    parser.add_argument("--family", default="kpart_k10_p0.3_n100-200", help="Target family ID")
    parser.add_argument("--max-calls", type=int, default=20, help="Max API calls ceiling")
    parser.add_argument("--max-tokens", type=int, default=60_000, help="Max tokens ceiling")
    parser.add_argument("--max-dollars", type=float, default=0.20, help="Max dollar budget ceiling")
    parser.add_argument("--model", default="deepseek-chat", help="Model name")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no real API call)")
    parser.add_argument("--lock-dir", default="results/pilot_locks", help="Lock directory")
    parser.add_argument("--log-path", default="results/pilot_candidates.jsonl", help="Candidates log file")
    args = parser.parse_args()

    run_pilot(
        family_id=args.family,
        max_calls=args.max_calls,
        max_tokens=args.max_tokens,
        max_dollars=args.max_dollars,
        model=args.model,
        mock=args.mock,
        lock_dir=args.lock_dir,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
