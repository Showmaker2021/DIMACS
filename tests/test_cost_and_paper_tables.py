import json
import pytest
from pathlib import Path

from analysis.cost_model import CostModel
from scripts.make_all_figures import generate_master_paper_tables


def test_cost_model_break_even():
    model = CostModel()
    c_evo = model.cost_evolution_per_family()
    c_inf = model.cost_inference_per_card()
    amort = model.cost_amortized_training()
    n_be = model.compute_break_even_families()

    assert c_evo > 0.0
    assert c_inf > 0.0
    assert c_evo > c_inf * 100  # Evolution is orders of magnitude more expensive per family
    assert amort["total_amortized_cost"] > 0.0
    assert 1.0 < n_be < 10000.0

    grid = model.run_sensitivity_grid()
    assert len(grid) == 12  # 4 API multipliers x 3 GPU rates


def test_master_paper_tables_generation(tmp_path):
    out_dir = tmp_path / "paper"
    res = generate_master_paper_tables(
        runs_dir="runs",
        out_dir=str(out_dir),
    )

    assert res["rows_count"] >= 12
    md_file = out_dir / "table_main.md"
    tex_file = out_dir / "table_main.tex"

    assert md_file.exists()
    assert tex_file.exists()

    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Check that all required arms appear in table
    for arm_name in ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "D7 Classifier", "DSATUR", "RLF", "R(F)", "EoH", "ReEvo"]:
        assert arm_name in md_text, f"Missing arm {arm_name} in paper table"

    with open(tex_file, "r", encoding="utf-8") as f:
        tex_text = f.read()
    assert r"\begin{table*}" in tex_text
    assert r"\end{table*}" in tex_text
