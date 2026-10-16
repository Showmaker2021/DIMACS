"""Amortized vs Per-Family Evolutionary Search Cost Model.

Implements Task 12.4 from PLAN_PHASE6_ONWARD.md:
- Computes per-family cost of from-scratch evolution (our_engine, EoH, ReEvo).
- Computes one-off amortized training cost: Teacher generation + SFT + GRPO.
- Computes deployment inference cost (1 forward pass per card).
- Calculates the break-even number of graph families where amortization dominates.
- Performs sensitivity analysis across various API token pricing and GPU rental rate tiers.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


class CostModel:
    """Analytical cost model for combinatorial heuristic generation."""

    def __init__(
        self,
        token_input_price_per_m: float = 0.14,
        token_output_price_per_m: float = 0.28,
        gpu_hourly_rate: float = 2.00,
        evolution_budget_calls: int = 200,
        evolution_avg_prompt_tokens: int = 400,
        evolution_avg_comp_tokens: int = 150,
        teacher_cards_count: int = 100,
        sft_gpu_hours: float = 4.0,
        grpo_gpu_hours: float = 12.0,
        inference_prompt_tokens: int = 350,
        inference_comp_tokens: int = 180,
    ):
        self.p_in = token_input_price_per_m / 1e6
        self.p_out = token_output_price_per_m / 1e6
        self.p_gpu = gpu_hourly_rate
        self.calls_per_fam = evolution_budget_calls
        self.prompt_tok_per_call = evolution_avg_prompt_tokens
        self.comp_tok_per_call = evolution_avg_comp_tokens
        self.teacher_cards = teacher_cards_count
        self.sft_gpu_h = sft_gpu_hours
        self.grpo_gpu_h = grpo_gpu_hours
        self.inf_prompt_tok = inference_prompt_tokens
        self.inf_comp_tok = inference_comp_tokens

    def cost_evolution_per_family(self) -> float:
        """Cost of running per-family evolutionary search."""
        cost_prompt = self.calls_per_fam * self.prompt_tok_per_call * self.p_in
        cost_comp = self.calls_per_fam * self.comp_tok_per_call * self.p_out
        return cost_prompt + cost_comp

    def cost_amortized_training(self) -> Dict[str, float]:
        """One-off amortized cost of teacher data collection, SFT, and GRPO."""
        # Teacher data collection (offline evolution across teacher_cards)
        c_teacher = self.teacher_cards * self.cost_evolution_per_family()
        c_sft = self.sft_gpu_h * self.p_gpu
        c_grpo = self.grpo_gpu_h * self.p_gpu
        c_total = c_teacher + c_sft + c_grpo
        return {
            "teacher_data_collection": c_teacher,
            "sft_training": c_sft,
            "grpo_training": c_grpo,
            "total_amortized_cost": c_total,
        }

    def cost_inference_per_card(self) -> float:
        """Cost of deploying distilled/RL-tuned generator (1 call per card)."""
        return self.inf_prompt_tok * self.p_in + self.inf_comp_tok * self.p_out

    def compute_break_even_families(self) -> float:
        """Break-even number of families N where total amortized cost < evolutionary cost."""
        c_evo = self.cost_evolution_per_family()
        c_inf = self.cost_inference_per_card()
        c_amort = self.cost_amortized_training()["total_amortized_cost"]

        net_savings_per_fam = c_evo - c_inf
        if net_savings_per_fam <= 0:
            return float("inf")
        return c_amort / net_savings_per_fam

    def run_sensitivity_grid(self) -> List[Dict[str, Any]]:
        """Sensitivity analysis over API price multipliers and GPU costs."""
        grid = []
        for api_mult in [0.5, 1.0, 2.0, 5.0]:
            for gpu_h in [1.0, 2.0, 4.0]:
                orig_in, orig_out, orig_gpu = self.p_in, self.p_out, self.p_gpu
                self.p_in = orig_in * api_mult
                self.p_out = orig_out * api_mult
                self.p_gpu = gpu_h

                n_be = self.compute_break_even_families()
                c_evo = self.cost_evolution_per_family()
                c_inf = self.cost_inference_per_card()
                c_total = self.cost_amortized_training()["total_amortized_cost"]

                grid.append({
                    "api_price_multiplier": api_mult,
                    "gpu_hourly_rate": gpu_h,
                    "cost_evolution_per_family": c_evo,
                    "cost_inference_per_card": c_inf,
                    "total_amortized_cost": c_total,
                    "break_even_families": round(n_be, 1),
                })

                self.p_in, self.p_out, self.p_gpu = orig_in, orig_out, orig_gpu
        return grid


def main():
    parser = argparse.ArgumentParser(description="Cost and Break-Even Model")
    parser.add_argument("--out-dir", default="runs/analysis/cost_model", help="Output directory")
    args = parser.parse_args()

    model = CostModel()
    c_evo = model.cost_evolution_per_family()
    c_inf = model.cost_inference_per_card()
    amort = model.cost_amortized_training()
    n_be = model.compute_break_even_families()
    grid = model.run_sensitivity_grid()

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    res = {
        "cost_evolution_per_family": c_evo,
        "cost_inference_per_card": c_inf,
        "amortized_breakdown": amort,
        "break_even_families": n_be,
        "sensitivity_analysis": grid,
    }

    with open(out_p / "cost_report.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    # Render Markdown report
    lines = [
        "# Cost Model and Amortization Analysis",
        "",
        f"- **Per-Family Evolutionary Search Cost**: ${c_evo:.4f}",
        f"- **Amortized Generator Deployment Inference Cost**: ${c_inf:.6f} per card",
        f"- **Total Amortized Training Investment**: ${amort['total_amortized_cost']:.2f}",
        f"  * Teacher Data Collection: ${amort['teacher_data_collection']:.2f}",
        f"  * SFT LoRA Training: ${amort['sft_training']:.2f}",
        f"  * GRPO Training: ${amort['grpo_training']:.2f}",
        f"- **Break-Even Threshold**: **{n_be:.1f} families**",
        "",
        "## Sensitivity Analysis",
        "| API Multiplier | GPU Rate ($/h) | Evo Cost / Fam | Amortized Cost | Break-Even Families |",
        "|---|---|---|---|---|",
    ]
    for row in grid:
        lines.append(
            f"| {row['api_price_multiplier']}x | ${row['gpu_hourly_rate']:.2f} | "
            f"${row['cost_evolution_per_family']:.4f} | ${row['total_amortized_cost']:.2f} | "
            f"**{row['break_even_families']}** |"
        )

    with open(out_p / "cost_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Cost report generated at: {out_p / 'cost_report.md'}")


if __name__ == "__main__":
    main()
