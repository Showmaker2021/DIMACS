"""Feasibility probe for stronger model in the Gemini family.

Checks:
- Model: gemini-2.5-flash (or gemini-flash-latest)
- finishReason == 'STOP'
- Code complete & parseable (no truncation)
- Sufficient maxOutputTokens (4096)
- Small thinking budget (512 tokens)
- Inspects response headers for rate limits (RPM / remaining requests)
- Max 5 calls ceiling (default: 2 calls)
"""

import os
import sys
import json
import time
import ast
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.dataset import load_family, families
from gate.evolved.task import SEED_CODE_DSATUR, SEED_CODE_RLF
from gate.evolved.prompts import SYSTEM_PROMPT, build_user_prompt
from gate.evolved.sandbox import evaluate_candidate_safe
from scripts.smoke import extract_code_block


def probe_model(model_name: str = "gemini-2.5-flash", num_calls: int = 2):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment.")
        return

    print("=" * 80)
    print(f"PROBING STRONGER MODEL: {model_name} (Max calls: {num_calls})")
    print("=" * 80)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    # Setup dummy prompt on kpart family
    fam_id = "kpart_k10_p0.3_n100-200"
    fam_lookup = {f["family_id"]: f for f in families()}
    card = fam_lookup[fam_id]["cards"]["card"]
    parents = [
        {"program": SEED_CODE_DSATUR, "mean_colors": 14.35, "fitness": 0.0},
        {"program": SEED_CODE_RLF, "mean_colors": 13.70, "fitness": 0.65},
    ]
    user_prompt = build_user_prompt(fam_id, card, 13.70, parents, operator="crossover")

    # Payload with large maxOutputTokens and small thinkingBudget
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
            "thinkingConfig": {
                "thinkingBudget": 512
            }
        }
    }

    train_graphs = load_family(fam_id, "baseline_train")

    for i in range(1, num_calls + 1):
        print(f"\n--- CALL {i}/{num_calls} to {model_name} ---")
        t0 = time.time()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                elapsed = time.time() - t0
                status_code = resp.status
                resp_headers = dict(resp.headers)
                body = json.loads(resp.read().decode("utf-8"))

            cand_meta = body.get("candidates", [{}])[0]
            finish_reason = cand_meta.get("finishReason", "UNKNOWN")
            usage = body.get("usageMetadata", {})
            raw_text = cand_meta.get("content", {}).get("parts", [{}])[0].get("text", "")

            # Rate limit headers check
            rate_limit_info = {}
            for k, v in resp_headers.items():
                k_lower = k.lower()
                if "ratelimit" in k_lower or "quota" in k_lower:
                    rate_limit_info[k] = v

            code = extract_code_block(raw_text)
            
            # AST verification
            try:
                ast.parse(code)
                is_ast_valid = True
                ast_err = None
            except Exception as e:
                is_ast_valid = False
                ast_err = str(e)

            # Sandbox evaluation on 5 graphs
            eval_res = evaluate_candidate_safe(code, train_graphs[:5], per_graph_timeout=0.5)

            print(f"  * Status: HTTP {status_code} ({elapsed:.2f}s)")
            print(f"  * Finish Reason: {finish_reason} (Target: STOP)")
            print(f"  * Token Breakdown: Prompt={usage.get('promptTokenCount')} | Completion={usage.get('candidatesTokenCount')} | Total={usage.get('totalTokenCount')}")
            print(f"  * AST Parse: {'VALID' if is_ast_valid else f'FAIL ({ast_err})'}")
            print(f"  * Code Length: {len(code)} chars ({len(code.splitlines())} lines)")
            print(f"  * Sandbox 5-graph Eval: {'SUCCESS (mean colors: ' + str(eval_res.get('mean_colors')) + ')' if eval_res['success'] else 'FAIL (' + str(eval_res.get('error')) + ')'}")
            
            if rate_limit_info:
                print(f"  * Rate Limit Headers: {rate_limit_info}")
            else:
                print("  * Rate Limit Headers: None returned in HTTP response")

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            print(f"  * HTTP Error {e.code}: {e.reason}")
            print(f"  * Details: {err_body[:200]}")
        except Exception as e:
            print(f"  * Exception: {e}")

        time.sleep(2)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash"
    calls = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    probe_model(model, calls)
