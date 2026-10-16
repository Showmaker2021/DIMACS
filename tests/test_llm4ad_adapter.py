import pytest
from pathlib import Path

from baselines.llm4ad_task import GraphColoringEvaluation, TEMPLATE_PROGRAM
from gate.evolved.sandbox import evaluate_candidate_safe
from gate.dataset import load_family


VALID_CODE = """def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    return sat * (n + 1) + uncolored_deg
"""

FORBIDDEN_AST_CODE = """def priority(adj, colors, sat, deg, uncolored_deg):
    import sys
    return sat.astype(float)
"""

TIMEOUT_CODE = """def priority(adj, colors, sat, deg, uncolored_deg):
    x = 0
    while True:
        x += 1
    return sat.astype(float)
"""

STATEFUL_CODE = """def priority(adj, colors, sat, deg, uncolored_deg):
    adj[0, 1] = not adj[0, 1]
    return sat.astype(float)
"""


@pytest.fixture(scope="module")
def eval_adapter():
    # Use ba_m12_n200-300 with 2 graphs for fast test
    return GraphColoringEvaluation(
        family_name="ba_m12_n200-300",
        split="train",
        seed_range=(0, 1),
        per_graph_timeout=0.5,
        overall_timeout=5.0,
    )


def test_valid_candidate_matches_sandbox(eval_adapter):
    score = eval_adapter.evaluate_program(VALID_CODE)
    assert score is not None
    assert score < 0  # Should be negative mean colors

    # Direct sandbox evaluation
    direct_res = evaluate_candidate_safe(
        code_str=VALID_CODE,
        graphs=eval_adapter.graphs,
        per_graph_timeout=0.5,
        overall_timeout=5.0,
    )
    assert direct_res["success"] is True
    expected_score = -float(direct_res["mean_colors"])
    assert score == pytest.approx(expected_score, abs=1e-9)


def test_ast_violation_rejected(eval_adapter):
    score = eval_adapter.evaluate_program(FORBIDDEN_AST_CODE)
    assert score is None
    assert eval_adapter.last_result is not None
    assert "AST" in eval_adapter.last_result["error"]


def test_timeout_rejected(eval_adapter):
    score = eval_adapter.evaluate_program(TIMEOUT_CODE)
    assert score is None
    assert eval_adapter.last_result is not None
    assert eval_adapter.last_result["success"] is False


def test_stateful_purity_rejected(eval_adapter):
    score = eval_adapter.evaluate_program(STATEFUL_CODE)
    assert score is None
    assert eval_adapter.last_result is not None
    assert eval_adapter.last_result["success"] is False


def test_penalty_option():
    adapter_with_penalty = GraphColoringEvaluation(
        family_name="ba_m12_n200-300",
        split="train",
        seed_range=(0, 1),
        return_penalty_on_fail=True,
    )
    score = adapter_with_penalty.evaluate_program(FORBIDDEN_AST_CODE)
    assert score == -1e6
