"""Unified Evolutionary Heuristic Search Engine for Graph Coloring.

Shared core engine for smoke testing (smoke.py) and full pilot evolution (pilot_evolve.py).
Features:
- Behavioral deduplication: population and selection keep only ONE representative per 20-dim train color vector.
- Operator schedule: ~40% explore (temp 0.9), 30% crossover (temp 0.7), 30% refine (temp 0.5).
- Past behaviors tracking: explore prompts receive concise 1-line summaries of past tried mechanisms.
- Pacing & Rate Limit: 5-second pacing delay between LLM calls; exponential backoff for HTTP 429.
- Safe API handling: Key passed exclusively via HTTP headers; never logged or printed.
- Resume filtering: Resuming strictly filters by family_id, model, is_mock, and run_id.
- Frozen Selection: Final candidate selection evaluated on teacher_val (seeds 3000-3019), with val split as secondary diagnostic only.
"""

import sys
import os
import json
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

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


def load_config(config_path: str = "configs/pricing.json") -> Tuple[str, dict]:
    """Load model name and pricing structure from config."""
    p = Path(config_path)
    if not p.exists():
        default_cfg = {
            "model": "deepseek-chat",
            "pricing": {
                "deepseek-chat": {"input_per_million": 0.14, "output_per_million": 0.28},
                "mock": {"input_per_million": 0.14, "output_per_million": 0.28},
            }
        }
        return default_cfg["model"], default_cfg["pricing"]
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = data.get("model", "deepseek-chat")
    pricing = data.get("pricing", {})
    return model, pricing


def extract_code_block(text: str) -> str:
    """Extract python code from markdown block and strip redundant import lines."""
    if "```python" in text:
        parts = text.split("```python")
        code = parts[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        code = parts[1].split("```")[0]
    else:
        code = text
    # Strip harmless import lines since np is already provided in sandbox scope
    filtered_lines = []
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("import numpy") or s == "import numpy as np" or s.startswith("from numpy import"):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def call_llm_with_retry(
    model: str,
    system_prompt: str,
    user_prompt: str,
    pricing: dict,
    mock: bool = False,
    max_retries: int = 4,
    temperature: float = 0.7,
    pacing_delay: float = 5.0,
) -> Tuple[str, dict, float, int]:
    """Call LLM API with up to max_retries attempts, 5s pacing, and 429 exponential backoff.
    
    Returns:
    (code_str, usage_dict, call_cost, attempts)
    """
    p_info = pricing.get(model if not mock else "mock", {"input_per_million": 0.14, "output_per_million": 0.28})
    p_in = p_info["input_per_million"]
    p_out = p_info["output_per_million"]

    if mock:
        code = """def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    unc = (colors == -1)
    if not np.any(unc):
        return np.zeros(n, dtype=np.float64)
    cur_color = int(np.max(colors))
    scores = np.zeros(n, dtype=np.float64)
    if cur_color >= 0:
        in_cur_class = (colors == cur_color)
        cand = unc & (~np.any(adj[:, in_cur_class], axis=1))
        excl = unc & (~cand)
        if np.any(cand):
            ne = adj.dot(excl.astype(np.float64))
            nc = adj.dot(cand.astype(np.float64))
            scores[cand] = ne[cand] * (n + 1.0) - nc[cand]
            return scores
    scores[unc] = sat[unc].astype(np.float64) * (n + 1.0) + uncolored_deg[unc].astype(np.float64)
    return scores
"""
        usage = {"prompt_tokens": 1950, "completion_tokens": 350}
        cost = (usage["prompt_tokens"] * p_in + usage["completion_tokens"] * p_out) / 1_000_000.0
        return code, usage, cost, 1

    if "gemini" in model.lower():
        api_key = os.environ.get("GEMINI_API_KEY")
    elif "deepseek" in model.lower():
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    elif "gpt" in model.lower() or "openai" in model.lower():
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(f"API key missing in environment for model '{model}'. Set GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.")

    attempts = 0
    last_err = None

    while attempts <= max_retries:
        attempts += 1
        try:
            if "gemini" in model.lower():
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                }
                payload = {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": {
                        "temperature": float(temperature),
                        "maxOutputTokens": 2048,
                    },
                }
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                meta = data.get("usageMetadata", {})
                finish_reason = data["candidates"][0].get("finishReason", "UNKNOWN")
                usage = {
                    "prompt_tokens": meta.get("promptTokenCount", 0),
                    "completion_tokens": meta.get("candidatesTokenCount", 0),
                    "finish_reason": finish_reason,
                }
                code = extract_code_block(raw_text)
                cost = (usage["prompt_tokens"] * p_in + usage["completion_tokens"] * p_out) / 1_000_000.0
                return code, usage, cost, attempts

            elif "deepseek" in model.lower() or "gpt" in model.lower() or "openai" in model.lower():
                url = "https://api.deepseek.com/chat/completions" if "deepseek" in model.lower() else "https://api.openai.com/v1/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": float(temperature),
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
                    "temperature": float(temperature),
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

        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 503):
                wait_time = min(60.0, 10.0 * (2 ** (attempts - 1)))
                print(f"    [HTTP {e.code}] Temporary congestion/rate limit: waiting {wait_time:.1f}s before retry (attempt {attempts}/{max_retries + 1})...")
                time.sleep(wait_time)
            else:
                time.sleep(2.0 * attempts)
        except Exception as e:
            last_err = e
            time.sleep(2.0 * attempts)

    raise RuntimeError(f"API call failed after {attempts} attempts. Last error: {last_err}")


def deduplicate_population_by_behavior(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate candidates by train color counts vector.
    
    For candidates sharing the same train color counts vector (behavioral signature),
    keep only ONE representative:
    - Highest fitness
    - Tie-break: Shortest program length (len(program.strip()))
    
    Returns deduplicated candidates sorted by fitness descending.
    """
    groups = {}
    for c in candidates:
        counts = c.get("train_counts")
        if counts is not None:
            key = tuple(counts)
        else:
            key = hashlib.sha256(c["program"].strip().encode()).hexdigest()
            
        fit = c.get("fitness", -1e9)
        prog_len = len(c.get("program", "").strip())
        
        if key not in groups:
            groups[key] = c
        else:
            existing = groups[key]
            ex_fit = existing.get("fitness", -1e9)
            ex_len = len(existing.get("program", "").strip())
            if (fit > ex_fit) or (np.isclose(fit, ex_fit) and prog_len < ex_len):
                groups[key] = c
                
    deduped = list(groups.values())
    deduped.sort(key=lambda x: (x.get("fitness", -1e9), -len(x.get("program", "").strip())), reverse=True)
    return deduped


def extract_behavior_summary(code: str, mean_colors: float) -> str:
    """Extract a concise 1-line description of the candidate's mechanism."""
    c_lower = code.lower()
    features = []
    if "in_cur_class" in c_lower or "cur_color" in c_lower:
        features.append("RLF color class candidate set extension")
    if "adj.dot(adj" in c_lower or "np.dot(adj, adj" in c_lower or "adj @ adj" in c_lower:
        features.append("common neighbors / 2-hop matrix multiplication")
    elif "adj.dot" in c_lower or "@" in c_lower:
        features.append("vectorized neighbor dot products")
    if "sat" in c_lower and "n + 1" in c_lower:
        features.append("DSATUR saturation primary score")
    if "deg" in c_lower and ("-" in c_lower or "/" in c_lower):
        features.append("degree ratio/penalty")
    if not features:
        features.append("custom polynomial / tie-breaking scoring")
    feat_str = " + ".join(features)
    return f"{feat_str} (mean train colors: {mean_colors:.2f})"


def run_evolution(
    family_id: str = "kpart_k10_p0.3_n100-200",
    max_calls: int = 150,
    max_tokens: int = 450_000,
    max_dollars: float = 1.50,
    mock: bool = False,
    log_path: str = "results/pilot_candidates.jsonl",
    run_id: Optional[str] = None,
    selection_dir: Optional[str] = "results/selections",
    selection_split: str = "teacher_val",
    explore_ratio: float = 0.40,
    pacing_delay: float = 5.0,
    is_smoke: bool = False,
) -> Dict[str, Any]:
    """Execute evolutionary heuristic optimization loop.
    
    Shared engine between smoke.py and pilot_evolve.py.
    """
    if run_id is None:
        run_id = f"smoke_{family_id}" if is_smoke else f"pilot_{family_id}"

    # Snapshot directory before execution
    files_before = set(os.listdir("."))

    # 1. Load config & pricing
    cfg_model, pricing = load_config()
    model = "mock" if mock else cfg_model
    p_info = pricing.get(model, {"input_per_million": 0.14, "output_per_million": 0.28})
    p_in = p_info["input_per_million"]
    p_out = p_info["output_per_million"]

    print("=" * 80)
    title = "SMOKE TEST RUNNER (NO LOCK, VAL ONLY)" if is_smoke else "PILOT EVOLUTION RUNNER (RESUME SUPPORT, TEACHER_VAL SELECTION)"
    print(f"{title}: Family '{family_id}'")
    print(f"Run ID: {run_id} | Log Path: {log_path}")
    print(f"Configured Model: {cfg_model} (Active: {model}) | Effective Pricing: ${p_in:.4f}/1M In, ${p_out:.4f}/1M Out")
    print(f"Ceilings: Max Calls = {max_calls}, Max Tokens = {max_tokens}, Max Dollars = ${max_dollars:.2f} (2x safety)")
    print(f"Operator Schedule: {explore_ratio*100:.0f}% explore (temp 0.9), {(1-explore_ratio)*50:.0f}% crossover (temp 0.7), {(1-explore_ratio)*50:.0f}% refine (temp 0.5)")
    print("=" * 80)

    # 2. Load family card & metadata
    fam_lookup = {f["family_id"]: f for f in families()}
    if family_id not in fam_lookup:
        raise ValueError(f"Family '{family_id}' not found in families.json")
    card = fam_lookup[family_id].get("cards", {}).get("card", "")
    print(f"Family Card: \"{card}\"")

    # 3. Load baseline train graphs (20 graphs, seeds 0-19)
    train_graphs = load_family(family_id, "baseline_train")
    ds_train_counts = [count_colors(dsatur(g[1])) for g in train_graphs]
    rlf_train_counts = [count_colors(rlf(g[1])) for g in train_graphs]

    c_ds_train = float(np.mean(ds_train_counts))
    c_rlf_train = float(np.mean(rlf_train_counts))
    r_train = min(c_ds_train, c_rlf_train)

    # Precompute seed behavior hashes on first 3 train graphs for absolute clone check
    ds_hashes = [hashlib.sha256(dsatur(adj).tobytes()).hexdigest() for _, adj in train_graphs[:3]]
    ds_hash = "-".join(ds_hashes)
    rlf_hashes = [hashlib.sha256(rlf(adj).tobytes()).hexdigest() for _, adj in train_graphs[:3]]
    rlf_hash = "-".join(rlf_hashes)

    per_graph_timeout = CALIBRATED_PER_GRAPH_TIMEOUTS.get(family_id, 0.50)
    logger = CandidateLogger(log_path)

    # Initial Population
    population = [
        {"program": SEED_CODE_DSATUR, "fitness": 0.0, "mean_colors": c_ds_train, "is_absolute_clone": False, "is_equivalent_clone": False, "train_counts": tuple(ds_train_counts)},
        {"program": SEED_CODE_RLF, "fitness": (r_train - c_rlf_train), "mean_colors": c_rlf_train, "is_absolute_clone": False, "is_equivalent_clone": False, "train_counts": tuple(rlf_train_counts)},
    ]

    past_behaviors = [
        "DSATUR baseline: sat * (n + 1) + uncolored_deg",
        "RLF baseline: independent set candidate set extension (ne * (n + 1) - nc)",
    ]

    # Metrics
    total_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    valid_count = 0
    rejected_count = 0
    rejection_reasons = {}
    abs_clone_count = 0
    equiv_clone_count = 0
    evaluated_records = []

    # 4. Resume check: strictly match family_id, model, mock, AND run_id!
    if Path(log_path).exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    model_matches = True if mock else (rec.get("model") in (model, cfg_model))
                    if (
                        rec.get("family_id") == family_id
                        and rec.get("run_id") == run_id
                        and model_matches
                        and bool(rec.get("mock", False)) == bool(mock)
                    ):
                        evaluated_records.append(rec)
                        total_calls += 1
                        toks = rec.get("tokens") or {}
                        total_prompt_tokens += toks.get("prompt_tokens", 0) if isinstance(toks, dict) else 0
                        total_completion_tokens += toks.get("completion_tokens", 0) if isinstance(toks, dict) else 0
                        total_cost += (rec.get("cost_dollars") or 0.0)

                        if rec.get("valid", False):
                            valid_count += 1
                            if rec.get("is_absolute_clone", False):
                                abs_clone_count += 1
                            if rec.get("is_equivalent_clone", False):
                                equiv_clone_count += 1

                            tr_counts = rec.get("train_counts")
                            if tr_counts is not None and not rec.get("is_absolute_clone", False):
                                population.append({
                                    "program": rec["code"],
                                    "fitness": rec.get("gain_train", 0.0),
                                    "mean_colors": rec.get("train_mean", 0.0),
                                    "is_absolute_clone": False,
                                    "is_equivalent_clone": rec.get("is_equivalent_clone", False),
                                    "train_counts": tuple(tr_counts),
                                })
                                past_behaviors.append(extract_behavior_summary(rec["code"], rec.get("train_mean", 0.0)))
                        else:
                            rejected_count += 1
                            reason = rec.get("rejection_reason", "Unknown")
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                except Exception:
                    continue

    print(f"[{family_id}] Resumed {total_calls} calls matching run_id='{run_id}', model='{model}', mock={mock}.")
    print(f"Current Population: {len(population)} candidates ({len(deduplicate_population_by_behavior(population))} behaviorally distinct).")

    # 5. Evolutionary Search Loop
    rng = np.random.default_rng(2026 + total_calls)

    while total_calls < max_calls:
        # Check ceilings
        if total_prompt_tokens + total_completion_tokens >= max_tokens:
            print(f"[CEILING REACHED] Max token ceiling ({max_tokens}) reached: {total_prompt_tokens + total_completion_tokens} tokens.")
            break
        if total_cost >= max_dollars:
            print(f"[CEILING REACHED] Max dollar ceiling (${max_dollars:.2f}) reached: ${total_cost:.4f}.")
            break

        total_calls += 1
        iteration = total_calls

        # Operator Selection: explore_ratio explore, remainder split evenly between crossover and refine
        rand_val = rng.random()
        if rand_val < explore_ratio:
            operator = "explore"
            op_temp = 0.9
        elif rand_val < explore_ratio + (1.0 - explore_ratio) / 2.0:
            operator = "crossover"
            op_temp = 0.7
        else:
            operator = "refine"
            op_temp = 0.5

        # Behavioral deduplication of parents
        eligible_pool = deduplicate_population_by_behavior(population)
        if not eligible_pool:
            eligible_pool = population

        # Parent Selection
        if operator == "refine":
            parents = [rng.choice(eligible_pool[:max(1, len(eligible_pool) // 2)])]
        elif operator == "crossover":
            if len(eligible_pool) >= 2:
                idx = rng.choice(len(eligible_pool), size=2, replace=False)
                parents = [eligible_pool[idx[0]], eligible_pool[idx[1]]]
            else:
                parents = [eligible_pool[0]]
        else:  # explore
            parents = [eligible_pool[0]]

        # Build prompt
        parent_codes = [p["program"] for p in parents]
        parent_fitnesses = [p.get("fitness", 0.0) for p in parents]

        extra_explore_guidance = None
        if operator == "explore":
            recent_behaviors = past_behaviors[-5:] if len(past_behaviors) > 5 else past_behaviors
            behavior_list_str = "\n".join([f"- {b}" for b in recent_behaviors])
            extra_explore_guidance = (
                f"EXPLORATION MANDATE: Propose a structurally novel priority heuristic fundamentally distinct from past attempts.\n"
                f"Past tried mechanisms to AVOID duplicating:\n{behavior_list_str}\n\n"
                f"Consider unexplored structural ideas suitable for '{family_id}':\n"
                f"- Counting shared common neighbors between candidate vertices via 2-hop matrix multiplication (adj @ adj)\n"
                f"- Estimating local density ratios (e.g. deg / (1 + uncolored_deg))\n"
                f"- Distinct community tie-breaking penalties\n"
                f"- Secondary structural signals using numpy functions (e.g., minimum, maximum, count_nonzero, sign, bincount)"
            )

        user_prompt = build_user_prompt(
            family_id=family_id,
            card=card,
            r_train=r_train,
            parent_candidates=parents,
            operator=operator,
            past_behaviors=past_behaviors[-5:] if past_behaviors else None,
        )

        # Call LLM with pacing delay (5s) and backoff
        if not mock and iteration > 1:
            time.sleep(pacing_delay)

        t_llm_start = time.perf_counter()
        try:
            code, usage, call_cost, attempts = call_llm_with_retry(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                pricing=pricing,
                mock=mock,
                temperature=op_temp,
                pacing_delay=pacing_delay,
            )
            latency_s = float(round(time.perf_counter() - t_llm_start, 4))
        except Exception as e:
            latency_s = float(round(time.perf_counter() - t_llm_start, 4))
            print(f"[{family_id} #{iteration}] API Call Failed: {e}")
            log_entry = {
                "family_id": family_id,
                "run_id": run_id,
                "model": model,
                "mock": mock,
                "iteration": iteration,
                "operator": operator,
                "temperature": op_temp,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_s": latency_s,
                "status": f"llm_error: {e}",
                "code": None,
                "valid": False,
                "rejection_reason": f"llm_error: {e}",
                "train_mean": None,
                "gain_train": None,
                "train_counts": None,
                "is_absolute_clone": False,
                "is_equivalent_clone": False,
                "tokens": {},
                "cost_dollars": None,
                "cumulative_cost": total_cost,
                "timestamp": time.time(),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            break

        if isinstance(usage, dict):
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
        if call_cost is not None:
            total_cost += call_cost

        # Evaluate candidate safely on 20 baseline train graphs
        res = evaluate_candidate_safe(
            code_str=code,
            graphs=train_graphs,
            per_graph_timeout=per_graph_timeout,
            overall_timeout=per_graph_timeout * 25.0,
        )

        valid = res["success"]
        rejection_reason = None
        train_mean = None
        gain_train = None
        is_abs_clone = False
        is_equiv_clone = False
        train_counts_list = None

        if valid:
            valid_count += 1
            train_counts = res["color_counts"]
            train_counts_list = [int(c) for c in train_counts]
            train_mean = float(np.mean(train_counts))
            gain_train = r_train - train_mean

            # Two-level seed clone checks
            cand_hashes = [hashlib.sha256(c_arr.tobytes()).hexdigest() for c_arr in res.get("first_3_colorings", [])]
            cand_hash = "-".join(cand_hashes)
            if cand_hash in (ds_hash, rlf_hash):
                is_abs_clone = True
                abs_clone_count += 1

            if tuple(train_counts_list) in (tuple(ds_train_counts), tuple(rlf_train_counts)):
                is_equiv_clone = True
                equiv_clone_count += 1

            # Absolute clones excluded from mating pool
            if not is_abs_clone:
                population.append({
                    "program": code,
                    "fitness": gain_train,
                    "mean_colors": train_mean,
                    "is_absolute_clone": False,
                    "is_equivalent_clone": is_equiv_clone,
                    "train_counts": tuple(train_counts_list),
                })
                past_behaviors.append(extract_behavior_summary(code, train_mean))

            status_str = f"VALID (Mean={train_mean:.2f}, Gain={gain_train:+.2f})"
            if is_abs_clone:
                status_str += " [ABSOLUTE SEED CLONE]"
            elif is_equiv_clone:
                status_str += " [EQUIVALENT SEED CLONE]"
        else:
            rejected_count += 1
            rejection_reason = res.get("error", "Unknown validation failure")
            rejection_reasons[rejection_reason] = rejection_reasons.get(rejection_reason, 0) + 1
            status_str = f"REJECTED: {rejection_reason}"

        cost_str = f"${call_cost:.4f}" if call_cost is not None else "None"
        print(f"[{family_id} #{iteration:03d}] Op={operator:<9} (T={op_temp:.1f}) | {status_str} | Cost={cost_str} (Total=${total_cost:.4f})")

        prompt_tokens = int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) else 0
        completion_tokens = int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
        status = "ok" if valid else str(rejection_reason or "rejected")
        cost_dollars = float(call_cost) if (call_cost is not None and call_cost >= 0) else None

        # Log to JSONL
        log_entry = {
            "family_id": family_id,
            "run_id": run_id,
            "model": model,
            "mock": mock,
            "iteration": iteration,
            "operator": operator,
            "temperature": op_temp,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_s": latency_s,
            "status": status,
            "code": code,
            "valid": valid,
            "rejection_reason": rejection_reason,
            "train_mean": train_mean,
            "gain_train": gain_train,
            "train_counts": train_counts_list,
            "is_absolute_clone": is_abs_clone,
            "is_equivalent_clone": is_equiv_clone,
            "tokens": usage if isinstance(usage, dict) else {},
            "cost_dollars": cost_dollars,
            "cumulative_cost": total_cost,
            "timestamp": time.time(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    # 6. Post-Run Deduplication & Frozen Selection
    deduped_pop = deduplicate_population_by_behavior(population)
    top10_candidates = [c for c in deduped_pop if c.get("fitness", -1e9) > -1e8][:10]

    selection_record = None
    if selection_dir and top10_candidates:
        Path(selection_dir).mkdir(parents=True, exist_ok=True)
        selection_file = Path(selection_dir) / f"{family_id}.json"

        # Load selection split (teacher_val by default, seeds 3000-3019)
        sel_graphs = load_family(family_id, selection_split)
        ds_sel_mean = float(np.mean([count_colors(dsatur(g[1])) for g in sel_graphs]))
        rlf_sel_mean = float(np.mean([count_colors(rlf(g[1])) for g in sel_graphs]))
        r_sel = min(ds_sel_mean, rlf_sel_mean)

        # Secondary val evaluation for reporting
        val_graphs = load_family(family_id, "val")
        ds_val_mean = float(np.mean([count_colors(dsatur(g[1])) for g in val_graphs]))
        rlf_val_mean = float(np.mean([count_colors(rlf(g[1])) for g in val_graphs]))
        r_val = min(ds_val_mean, rlf_val_mean)

        best_cand = None
        best_sel_mean = 1e9

        for cand in top10_candidates:
            res_sel = evaluate_candidate_safe(cand["program"], sel_graphs, per_graph_timeout=per_graph_timeout)
            if res_sel["success"]:
                m_sel = float(np.mean(res_sel["color_counts"]))
                cand[f"{selection_split}_mean"] = m_sel
                if m_sel < best_sel_mean:
                    best_sel_mean = m_sel
                    best_cand = cand

        if best_cand is not None:
            # Secondary val score of best candidate
            res_val = evaluate_candidate_safe(best_cand["program"], val_graphs, per_graph_timeout=per_graph_timeout)
            val_mean = float(np.mean(res_val["color_counts"])) if res_val["success"] else 1e9

            is_fallback = (best_sel_mean >= r_sel)
            selection_record = {
                "family_id": family_id,
                "card": card,
                "selection_split": selection_split,
                "selected_candidate": {
                    "program": best_cand["program"],
                    "fitness_train": best_cand.get("fitness", 0.0),
                    "mean_train": best_cand.get("mean_colors", 0.0),
                },
                "teacher_val_mean_colors": best_sel_mean,
                "r_teacher_val": r_sel,
                "val_mean_colors_secondary": val_mean,
                "r_val_secondary": r_val,
                "is_fallback": is_fallback,
                "fallback_reason": "Worse or equal to R(F) on teacher_val" if is_fallback else "None (strictly improved)",
            }
            with open(selection_file, "w", encoding="utf-8") as f:
                json.dump(selection_record, f, indent=2)
            print(f"[{family_id}] Frozen selection saved to {selection_file} (Selected on {selection_split}: {best_sel_mean:.3f} vs R(F)={r_sel:.3f}, fallback={is_fallback}).")

    # Directory snapshot after execution
    files_after = set(os.listdir("."))
    new_files = files_after - files_before
    if new_files:
        print(f"[DIRECTORY INTEGRITY] New files created during run: {new_files}")
    else:
        print("[DIRECTORY INTEGRITY] No unexpected root files created. Working directory clean.")

    summary = {
        "family_id": family_id,
        "run_id": run_id,
        "model": model,
        "total_calls": total_calls,
        "valid_count": valid_count,
        "rejected_count": rejected_count,
        "abs_clone_count": abs_clone_count,
        "equiv_clone_count": equiv_clone_count,
        "distinct_behavior_count": len(deduped_pop),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cost": total_cost,
        "selection_record": selection_record,
    }
    return summary
