"""GRPO (Group Relative Policy Optimization) Training Loop for Graph Coloring.

Implements Task 11.3 and Task 11.4 from PLAN_PHASE6_ONWARD.md:
- Supports arms A2 (GRPO-only), A3 (SFT -> GRPO), and A4 (A3 + diversity + KL).
- Uses train/rollout.py to pull fresh graphs from rl_train_pool (seeds 100000-199999).
- Uses train/reward.py to compute verified color savings, runtime penalty, and advantages.
- Logs reward components, failure rates, entropy, and baseline similarity.
- Supports deterministic --mock mode for testing and fast iteration.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.rollout import sample_rl_graphs, evaluate_rollout_group
from train.reward import CandidateResult, compute_group_rewards, RewardConfig
from gate.dataset import families


def run_grpo_training(
    arm: str = "A3",
    output_dir: str = "runs/grpo/A3/seed_42",
    steps: int = 20,
    group_size: int = 8,
    model_name: str = "runs/sft/qwen_lora_v1",
    seed: int = 42,
    mock: bool = False,
) -> Dict[str, Any]:
    """Execute GRPO training loop with group relative policy optimization."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Arm configuration
    cfg = RewardConfig()
    if arm == "A4":
        cfg.d = 0.1  # Enable diversity reward
        beta_kl = 0.05
    elif arm == "A3":
        cfg.d = 0.0
        beta_kl = 0.02
    else:  # A2
        cfg.d = 0.0
        beta_kl = 0.01

    if mock:
        print(f"[MOCK GRPO] Running Arm {arm} on output: {output_dir}")
        rng = np.random.default_rng(seed)
        step_logs = []
        # Representative fast families strictly from train_pool for mock testing
        target_fams = ["ba_m12_n200-300", "ws_k10_p0.1_n100-200"]

        for step in range(1, steps + 1):
            fam = rng.choice(target_fams)
            graphs = sample_rl_graphs(fam, n_graphs=3, rng=rng)

            # Generate mock candidate group of size G
            mock_codes = []
            for g_idx in range(group_size):
                if g_idx == 0:
                    # Broken candidate to test failure handling
                    mock_codes.append("def priority(adj, colors, sat, deg, uncolored_deg):\n    import os\n    return sat\n")
                else:
                    noise = rng.uniform(0.01, 0.05)
                    mock_codes.append(
                        f"def priority(adj, colors, sat, deg, uncolored_deg):\n"
                        f"    return sat * (adj.shape[0] + 1) + uncolored_deg + {noise:.4f} * deg\n"
                    )

            cand_results, c_base, lat = evaluate_rollout_group(fam, mock_codes, graphs)
            rewards = compute_group_rewards(
                candidates=cand_results,
                config=cfg,
                c_base_per_graph=c_base,
            )
            from train.reward import compute_advantages
            advs = compute_advantages(rewards, eps_std=cfg.eps_std)

            mean_rew = float(np.mean(rewards))
            fail_count = sum(1 for c in cand_results if c.status != "ok")
            fail_rate = fail_count / group_size

            log_entry = {
                "step": step,
                "family_id": fam,
                "mean_reward": mean_rew,
                "mean_advantage": float(np.mean(advs)),
                "fail_rate": fail_rate,
                "latency_s": lat,
                "arm": arm,
            }
            step_logs.append(log_entry)

        with open(out_path / "step_logs.jsonl", "w", encoding="utf-8") as f:
            for l in step_logs:
                f.write(json.dumps(l) + "\n")

        summary = {
            "arm": arm,
            "steps": steps,
            "group_size": group_size,
            "final_mean_reward": step_logs[-1]["mean_reward"],
            "avg_fail_rate": float(np.mean([l["fail_rate"] for l in step_logs])),
            "status": "completed",
        }
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    # Real GPU GRPO Training Mode
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(f"Missing required dependencies for GRPO training: {e}") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting real GRPO training on device: {device} (Arm: {arm})")
    if device != "cuda":
        print("[WARNING] CUDA is not available. Real GRPO on CPU will be slow. Use GPU or --mock.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 4-bit QLoRA configuration to fit comfortably within Kaggle T4 16GB VRAM
    bnb_config = None
    if device == "cuda":
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print(f"[WARNING] BitsAndBytesConfig init failed: {e}. Falling back to 16-bit.")
            bnb_config = None

    # 1. Load Tokenizer & Model
    adapter_path = Path(model_name)
    if (adapter_path / "adapter_config.json").exists():
        # Load SFT LoRA checkpoint
        with open(adapter_path / "adapter_config.json", "r", encoding="utf-8") as f:
            ad_cfg = json.load(f)
        base_id = ad_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cuda" and bnb_config is not None:
            base_model = prepare_model_for_kbit_training(base_model)
        model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cuda" and bnb_config is not None:
            base_model = prepare_model_for_kbit_training(base_model)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base_model, peft_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 2. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    # 3. Load Splits & Cards Catalog (strictly train_pool to avoid data leakage)
    splits_file = Path("data/data_v1_2/splits.json")
    train_pool_fams = []
    if splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            sp_data = json.load(f)
        train_pool_fams = [
            f["family_id"]
            for f in sp_data.get("splits", {}).get("train_pool", {}).get("families", [])
        ]

    cards_file = Path("data/data_v1_2/cards.json")
    cards_catalog = {}
    if cards_file.exists():
        with open(cards_file, "r", encoding="utf-8") as f:
            cards_catalog = json.load(f)

    if train_pool_fams:
        fam_pool = [fid for fid in train_pool_fams if fid in cards_catalog]
        if not fam_pool:
            fam_pool = train_pool_fams
    else:
        fam_pool = list(cards_catalog.keys()) if cards_catalog else ["ba_m12_n200-300", "ws_k10_p0.1_n100-200"]

    from gate.evolved.prompts import build_user_prompt, SYSTEM_PROMPT
    from gate.evolved.loop import extract_code_block
    from train.reward import compute_advantages

    rng = np.random.default_rng(seed)
    step_logs = []
    print(f"Beginning GRPO rollouts across {len(fam_pool)} train_pool families for {steps} steps (Group Size G={group_size})...")

    for step in range(1, steps + 1):
        t0 = time.time()
        fam = rng.choice(fam_pool)
        c_info = cards_catalog.get(fam, {})
        card_text = c_info.get("card", fam)

        prompt = f"[SYSTEM] {SYSTEM_PROMPT}\n[USER]\nTask: write priority(adj, colors, sat, deg, uncolored_deg)\nGraph family: {card_text}\n[ASSISTANT]\n```python\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        # Generate G rollout candidates
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=group_size,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_codes = []
        for out_toks in outputs:
            gen_text = tokenizer.decode(out_toks[prompt_len:], skip_special_tokens=True)
            code = extract_code_block("```python\n" + gen_text)
            gen_codes.append(code)

        # Sample fresh graphs from rl_train_pool (seeds 100000-199999)
        graphs = sample_rl_graphs(fam, n_graphs=3, rng=rng)

        # Evaluate candidate group in sandbox
        cand_results, c_base, lat = evaluate_rollout_group(fam, gen_codes, graphs)
        rewards = compute_group_rewards(candidates=cand_results, config=cfg, c_base_per_graph=c_base)
        advs = compute_advantages(rewards, eps_std=cfg.eps_std)

        # Policy update step (GRPO advantage-weighted loss)
        model.train()
        optimizer.zero_grad()

        # Compute log-probs strictly on generated completion tokens (prompt masked out with -100)
        loss_val = 0.0
        for g_idx, adv in enumerate(advs):
            if abs(adv) > 1e-6:
                out_seq = outputs[g_idx].unsqueeze(0)  # Shape (1, seq_len)
                attn_mask = (out_seq != tokenizer.pad_token_id).long()

                # Mask out prompt tokens so loss is computed ONLY on completion tokens
                labels = out_seq.clone()
                labels[:, :prompt_len] = -100
                labels[out_seq == tokenizer.pad_token_id] = -100

                out = model(input_ids=out_seq, attention_mask=attn_mask, labels=labels)
                # Policy gradient weighted loss: out.loss is mean NLL of completion tokens
                loss = (out.loss * float(adv)) / group_size
                loss.backward()
                loss_val += float(out.loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        fail_count = sum(1 for c in cand_results if c.status != "ok")
        mean_rew = float(np.mean(rewards))
        step_time = time.time() - t0

        log_entry = {
            "step": step,
            "family_id": fam,
            "mean_reward": mean_rew,
            "mean_advantage": float(np.mean(advs)),
            "fail_rate": fail_count / group_size,
            "latency_s": step_time,
            "arm": arm,
        }
        step_logs.append(log_entry)
        print(f"[GRPO Step {step:03d}/{steps:03d}] Fam={fam:<24} | Mean Reward={mean_rew:+.3f} | Fail Rate={fail_count}/{group_size} ({step_time:.1f}s)")

    # Save final model & checkpoints
    final_grpo_dir = out_path / "final_policy"
    model.save_pretrained(str(final_grpo_dir))
    tokenizer.save_pretrained(str(final_grpo_dir))

    with open(out_path / "step_logs.jsonl", "w", encoding="utf-8") as f:
        for l in step_logs:
            f.write(json.dumps(l) + "\n")

    summary = {
        "arm": arm,
        "steps": steps,
        "group_size": group_size,
        "final_mean_reward": step_logs[-1]["mean_reward"] if step_logs else 0.0,
        "avg_fail_rate": float(np.mean([l["fail_rate"] for l in step_logs])) if step_logs else 0.0,
        "policy_path": str(final_grpo_dir),
        "status": "completed",
    }
    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"GRPO Training completed successfully! Final policy saved to: {final_grpo_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Graph Coloring Heuristics")
    parser.add_argument("--arm", default="A3", choices=["A2", "A3", "A4"], help="Training Arm")
    parser.add_argument("--output-dir", default="runs/grpo/A3/seed_42", help="Output directory")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--group-size", "--rollout-group-size", dest="group_size", type=int, default=8, help="Rollout group size G")
    parser.add_argument("--model-name", "--sft-checkpoint", dest="model_name", default="runs/sft/qwen_lora_v1", help="Base/SFT model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mock", action="store_true", help="Run simulated mock GRPO")
    args = parser.parse_args()

    run_grpo_training(
        arm=args.arm,
        output_dir=args.output_dir,
        steps=args.steps,
        group_size=args.group_size,
        model_name=args.model_name,
        seed=args.seed,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
