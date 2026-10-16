"""Unit and integration tests for offline evaluation harness (Task 9.2).

Verifies:
1. Metrics on synthetic data (beats_rf, pass@k, coverage@k, best_of_k, numerical stability).
2. Mock backend end-to-end on tiny split (valid artifacts, graceful handling of broken & slow code).
3. Disk caching (100% cache hits on identical re-runs, zero new evaluations).
4. Test-lock enforcement on test* splits, refusal on repeat, and --force-unlock behavior.
5. Non-test splits are never locked.
6. Run-id overwrite protection.
7. Lazy imports and error handling for HF/vLLM/API backend stubs.
8. Zero network access and zero torch import during tests.
"""

import json
from pathlib import Path
import re
import socket
import sys
import pytest
import numpy as np

from eval.coverage import (
    beats_rf,
    pass_at_k,
    coverage_at_k,
    empirical_coverage_at_k,
    best_of_k,
    compute_rf,
)
from eval.eval_model import (
    build_eval_prompt,
    extract_code,
    MockBackend,
    HFBackend,
    VLLMBackend,
    APIBackend,
    get_backend,
    run_evaluation,
    EvaluationCache,
    TestLockManager,
    classify_sandbox_error,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# 0. INVARIANT FIXTURES: ZERO NETWORK & ZERO TORCH IMPORT
# ==============================================================================

@pytest.fixture(autouse=True)
def forbid_network_access(monkeypatch):
    """Ensure tests never open real network connections."""
    def guarded_connect(self, *args, **kwargs):
        raise RuntimeError("Network access is strictly forbidden during tests!")
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def test_zero_torch_import():
    """Assert torch is not imported in test environment."""
    assert "torch" not in sys.modules, "torch was unexpectedly imported!"


# ==============================================================================
# 1. PURE METRICS UNIT TESTS (eval/coverage.py)
# ==============================================================================

class TestBeatsRF:
    def test_canonical_beats_rf_strict_win(self):
        # Candidate strictly better by >= eps
        assert beats_rf(10.0, 10.10, eps=0.05) is True
        assert beats_rf(10.0, 10.05, eps=0.05) is True

    def test_canonical_beats_rf_ties_do_not_count(self):
        # Exact tie is not a win
        assert beats_rf(10.0, 10.0, eps=0.05) is False
        assert beats_rf(10.0, 10.0, eps=0.0) is True  # eps=0 allows tie/win, but default eps=0.05 does not

    def test_canonical_beats_rf_smaller_margin_fails(self):
        # Better by 0.04 with eps=0.05 is not enough
        assert beats_rf(10.0, 10.04, eps=0.05) is False

    def test_canonical_beats_rf_worse_candidate(self):
        assert beats_rf(11.0, 10.0, eps=0.05) is False

    def test_canonical_beats_rf_edge_cases(self):
        assert beats_rf(None, 10.0, eps=0.05) is False
        assert beats_rf(float("nan"), 10.0, eps=0.05) is False
        assert beats_rf(float("inf"), 10.0, eps=0.05) is False
        assert beats_rf(10.0, float("nan"), eps=0.05) is False

    def test_compute_rf(self):
        assert compute_rf(12.5, 11.2) == 11.2
        assert compute_rf(9.0, 10.5) == 9.0


class TestPassAtK:
    def test_pass_at_k_k_equals_1(self):
        # pass@1 == c / n
        assert pass_at_k(10, 4, 1) == pytest.approx(0.4)
        assert pass_at_k(100, 25, 1) == pytest.approx(0.25)

    def test_pass_at_k_n_equals_k(self):
        # if c >= 1 and n == k, probability of at least 1 winner is 1.0
        assert pass_at_k(5, 1, 5) == 1.0
        assert pass_at_k(5, 5, 5) == 1.0
        # if c == 0 and n == k, probability is 0.0
        assert pass_at_k(5, 0, 5) == 0.0

    def test_pass_at_k_zero_winners(self):
        assert pass_at_k(10, 0, 3) == 0.0

    def test_pass_at_k_all_winners(self):
        assert pass_at_k(10, 10, 3) == 1.0

    def test_pass_at_k_n_minus_c_less_than_k(self):
        # Non-winners count is 2 (10 - 8 = 2). If k=3, any sample of 3 must have at least 1 winner.
        assert pass_at_k(10, 8, 3) == 1.0

    def test_pass_at_k_known_combination_value(self):
        # n=10, c=3, k=2: binom(7,2) / binom(10,2) = 21 / 45 = 7 / 15. pass@2 = 1 - 7/15 = 8/15
        assert pass_at_k(10, 3, 2) == pytest.approx(8.0 / 15.0)

    def test_pass_at_k_numerical_stability_large_numbers(self):
        # n=1000, c=500, k=64
        val = pass_at_k(1000, 500, 64)
        assert 0.0 <= val <= 1.0
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_pass_at_k_invalid_arguments_raise_errors(self):
        with pytest.raises(ValueError, match="cannot exceed n"):
            pass_at_k(5, 2, 6)
        with pytest.raises(ValueError, match="positive integer"):
            pass_at_k(0, 0, 1)
        with pytest.raises(ValueError, match="positive integer"):
            pass_at_k(5, 2, 0)
        with pytest.raises(ValueError, match="0 <= c <= n"):
            pass_at_k(5, -1, 2)
        with pytest.raises(ValueError, match="0 <= c <= n"):
            pass_at_k(5, 6, 2)
        with pytest.raises(TypeError):
            pass_at_k("10", 2, 1)  # type: ignore


class TestCoverageAtK:
    def test_coverage_at_k_all_cards_win(self):
        # 3 cards, all have winners
        res = coverage_at_k([2, 3, 1], n_samples=4, k=2)
        assert res > 0.5

    def test_coverage_at_k_no_cards_win(self):
        res = coverage_at_k([0, 0, 0], n_samples=4, k=2)
        assert res == 0.0

    def test_coverage_at_k_zero_cards(self):
        assert coverage_at_k([], n_samples=4, k=2) == 0.0

    def test_empirical_coverage_at_k(self):
        assert empirical_coverage_at_k([]) == 0.0
        assert empirical_coverage_at_k([[True, False], [False, False]]) == 0.5
        assert empirical_coverage_at_k([[False], [False]]) == 0.0
        assert empirical_coverage_at_k([[True], [True]]) == 1.0


class TestBestOfK:
    def test_best_of_k_normal(self):
        card_colors = [
            [12.0, 10.5, 11.0],  # card 1: best is 10.5
            [8.0, 7.5, 9.0],     # card 2: best is 7.5
        ]
        rf_colors = [11.0, 8.0]
        res = best_of_k(card_colors, rf_colors, k=2)

        # For card 1: best among first 2 is 10.5. Savings vs rf (11.0) is 0.5
        # For card 2: best among first 2 is 7.5. Savings vs rf (8.0) is 0.5
        assert res["mean_best_colors"] == pytest.approx(9.0)  # (10.5 + 7.5) / 2
        assert res["mean_savings_rf"] == pytest.approx(0.5)
        assert res["card_success_rate"] == 1.0

    def test_best_of_k_card_with_all_failed(self):
        card_colors = [
            [10.0, None],
            [None, None],  # all failed
        ]
        rf_colors = [10.0, 8.0]
        res = best_of_k(card_colors, rf_colors, k=2, failed_savings_fallback=0.0)

        assert res["mean_best_colors"] == pytest.approx(10.0)
        assert res["per_card_best_colors"] == [10.0, None]
        assert res["card_success_rate"] == 0.5
        assert res["per_card_savings_rf"] == [0.0, 0.0]

    def test_best_of_k_zero_cards(self):
        res = best_of_k([], [], k=2)
        assert res["mean_best_colors"] is None
        assert res["mean_savings_rf"] == 0.0
        assert res["card_success_rate"] == 0.0

    def test_best_of_k_invalid_args(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            best_of_k([[10.0]], [10.0, 11.0], k=1)
        with pytest.raises(ValueError, match="positive integer"):
            best_of_k([[10.0]], [10.0], k=0)


# ==============================================================================
# 2. PROMPT & BACKEND TESTS
# ==============================================================================

def test_build_eval_prompt_contains_required_sections():
    card = "random (Erdos-Renyi), nodes 100-200, density 0.10"
    prompt = build_eval_prompt(card, prompt_version="v1")
    assert "priority(adj, colors, sat, deg, uncolored_deg)" in prompt
    assert "Graph family description:" in prompt
    assert card in prompt
    assert "Pure function" in prompt


def test_extract_code_from_markdown():
    wrapped = "Here is the code:\n```python\ndef priority(adj, colors, sat, deg, uncolored_deg):\n    return sat\n```"
    extracted = extract_code(wrapped)
    assert extracted == "def priority(adj, colors, sat, deg, uncolored_deg):\n    return sat"

    plain = "def priority(adj, colors, sat, deg, uncolored_deg):\n    return sat"
    assert extract_code(plain) == plain


def test_mock_backend_emits_expected_archetypes():
    backend = MockBackend()
    completions = backend.generate("dummy prompt", n=4, temperature=0.7, seed=42)
    assert len(completions) == 4

    c0 = extract_code(completions[0])
    c1 = extract_code(completions[1])
    c2 = extract_code(completions[2])
    c3 = extract_code(completions[3])

    # Sample 0 is DSATUR-like
    assert "sat.astype(np.float64) * (n + 1)" in c0
    # Sample 1 is RLF-like
    assert "k = int(np.max(colors))" in c1
    # Sample 2 is broken AST (import)
    assert "import os" in c2
    # Sample 3 is slow timeout
    assert "for i in range" in c3


def test_lazy_backends_stubs():
    # HFBackend requires transformers/torch
    hf = HFBackend("fake-model")
    with pytest.raises((ImportError, NotImplementedError)):
        hf.generate("prompt", n=1, temperature=0.7, seed=1)

    # VLLMBackend requires vllm
    vllm = VLLMBackend("fake-model")
    with pytest.raises((ImportError, NotImplementedError)):
        vllm.generate("prompt", n=1, temperature=0.7, seed=1)

    # APIBackend requires non-empty env var
    with pytest.raises(ValueError, match="not set or empty"):
        APIBackend("fake-model", api_key_env="NONEXISTENT_KEY_12345")


# ==============================================================================
# 3. END-TO-END MOCK EVALUATION RUN (eval/eval_model.py)
# ==============================================================================

def test_mock_eval_end_to_end_on_tiny_split(tmp_path):
    """Run mock backend end-to-end on 2 families with 2 graphs and 4 samples."""
    out_dir = tmp_path / "runs_eval"
    cache_dir = tmp_path / "cache"
    lock_dir = tmp_path / "locks"
    run_id = "test_mock_run_01"

    res = run_evaluation(
        backend_name="mock",
        split_name="val",
        k=2,
        n_samples=4,
        temperature=0.7,
        n_graphs=2,
        run_id=run_id,
        seed=42,
        output_dir=out_dir,
        cache_dir=cache_dir,
        lock_dir=lock_dir,
        workers=2,
        families_filter=["gnp_d0.10_n50-100", "sbm_k4_in0.7_out0.10_n50-100"],
        per_graph_timeout=0.5,
        overall_timeout=5.0,
    )

    run_path = out_dir / run_id
    assert run_path.is_dir()

    # 1. Check generated files
    per_card_file = run_path / "per_card.jsonl"
    summary_file = run_path / "summary.json"
    config_file = run_path / "config.json"

    assert per_card_file.is_file()
    assert summary_file.is_file()
    assert config_file.is_file()

    # 2. Check config.json contents
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["run_id"] == run_id
    assert cfg["backend"] == "mock"
    assert cfg["k"] == 2
    assert cfg["n_samples"] == 4
    assert cfg["force_unlock_used"] is False
    assert "git_commit" in cfg
    assert "library_versions" in cfg
    assert cfg["prompt_version"] == "v1"

    # 3. Check per_card.jsonl
    with open(per_card_file, "r", encoding="utf-8") as f:
        cards = [json.loads(line) for line in f if line.strip()]
    assert len(cards) == 2

    for card in cards:
        assert card["n_samples"] == 4
        assert card["k"] == 2
        # 2 samples succeed (DSATUR, RLF) and 2 fail (AST fail, timeout)
        assert card["valid_candidates"] == 2
        assert card["valid_rate"] == 0.5
        assert card["failure_breakdown"]["ast_fail"] == 1
        assert card["failure_breakdown"]["timeout"] == 1
        assert card["dsatur_mean"] > 0
        assert card["rlf_mean"] > 0
        assert card["r_f"] > 0
        # DSATUR/RLF candidates do not strictly beat R(F)
        assert card["coverage_at_k"] == 0.0

        candidates = card["candidates"]
        assert len(candidates) == 4
        assert candidates[0]["status"] == "ok"
        assert candidates[1]["status"] == "ok"
        assert candidates[2]["status"] == "ast_fail"
        assert candidates[3]["status"] == "timeout"

    # 4. Check summary.json
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["total_cards"] == 2
    assert summary["valid_rate"] == 0.5
    assert summary["coverage_at_k"] == 0.0
    assert summary["failure_counts"]["ast_fail"] == 2
    assert summary["failure_counts"]["timeout"] == 2
    assert summary["frac_baseline_like"] == 0.5  # 2 out of 4 samples are DSATUR/RLF
    assert summary["frac_dsatur_like"] == 0.25
    assert summary["frac_rlf_like"] == 0.25


# ==============================================================================
# 4. DISK CACHE TESTS
# ==============================================================================

def test_disk_cache_hits_on_rerun(tmp_path):
    """Verify second identical run performs 0 new sandbox evaluations."""
    out_dir = tmp_path / "runs_eval"
    cache_dir = tmp_path / "cache"
    lock_dir = tmp_path / "locks"

    # Run 1: initial run, misses cache
    res1 = run_evaluation(
        backend_name="mock",
        split_name="val",
        k=1,
        n_samples=2,
        n_graphs=2,
        run_id="run_cache_1",
        seed=42,
        output_dir=out_dir,
        cache_dir=cache_dir,
        lock_dir=lock_dir,
        families_filter=["gnp_d0.10_n50-100"],
    )
    assert res1["summary"]["cache_hits"] == 0
    assert res1["summary"]["cache_misses"] == 2
    assert res1["summary"]["cache_hit_rate"] == 0.0

    # Run 2: identical candidates and graphs, must hit cache 100%
    res2 = run_evaluation(
        backend_name="mock",
        split_name="val",
        k=1,
        n_samples=2,
        n_graphs=2,
        run_id="run_cache_2",
        seed=42,
        output_dir=out_dir,
        cache_dir=cache_dir,
        lock_dir=lock_dir,
        families_filter=["gnp_d0.10_n50-100"],
    )
    assert res2["summary"]["cache_hits"] == 2
    assert res2["summary"]["cache_misses"] == 0
    assert res2["summary"]["cache_hit_rate"] == 1.0


# ==============================================================================
# 5. TEST-LOCK AND --FORCE-UNLOCK TESTS
# ==============================================================================

def test_test_lock_refusal_and_force_unlock(tmp_path):
    """Test that split starting with 'test' creates .done lock and blocks re-eval."""
    out_dir = tmp_path / "runs_eval"
    cache_dir = tmp_path / "cache"
    lock_dir = tmp_path / "locks"

    # Run 1 on test_pilot: creates lock
    res1 = run_evaluation(
        backend_name="mock",
        split_name="test_pilot",
        k=1,
        n_samples=1,
        n_graphs=2,
        run_id="run_lock_1",
        output_dir=out_dir,
        cache_dir=cache_dir,
        lock_dir=lock_dir,
        families_filter=["gnp_d0.10_n50-100"],
    )
    assert res1["summary"]["force_unlock_used"] is False

    # Check that .done lock file was created
    done_files = list(lock_dir.glob("*.done"))
    assert len(done_files) == 1

    # Run 2 with identical config and new run_id without --force-unlock: must be refused!
    with pytest.raises(RuntimeError, match="Second evaluation is blocked to prevent data snooping"):
        run_evaluation(
            backend_name="mock",
            split_name="test_pilot",
            k=1,
            n_samples=1,
            n_graphs=2,
            run_id="run_lock_2",
            output_dir=out_dir,
            cache_dir=cache_dir,
            lock_dir=lock_dir,
            force_unlock=False,
            families_filter=["gnp_d0.10_n50-100"],
        )

    # Run 3 with --force-unlock: must succeed and log force_unlock_used = True
    res3 = run_evaluation(
        backend_name="mock",
        split_name="test_pilot",
        k=1,
        n_samples=1,
        n_graphs=2,
        run_id="run_lock_3",
        output_dir=out_dir,
        cache_dir=cache_dir,
        lock_dir=lock_dir,
        force_unlock=True,
        families_filter=["gnp_d0.10_n50-100"],
    )
    assert res3["summary"]["force_unlock_used"] is True
    assert res3["config"]["force_unlock_used"] is True


def test_val_split_not_locked(tmp_path):
    """Non-test splits (like val) should never create .done lock files or be blocked."""
    out_dir = tmp_path / "runs_eval"
    cache_dir = tmp_path / "cache"
    lock_dir = tmp_path / "locks"

    # Run 1
    run_evaluation(
        backend_name="mock",
        split_name="val",
        k=1,
        n_samples=1,
        n_graphs=2,
        run_id="run_val_1",
        output_dir=out_dir,
        cache_dir=cache_dir,
        lock_dir=lock_dir,
        families_filter=["gnp_d0.10_n50-100"],
    )

    # Assert no .done files in lock_dir
    assert len(list(lock_dir.glob("*.done"))) == 0

    # Run 2 on same config succeeds without error
    res2 = run_evaluation(
        backend_name="mock",
        split_name="val",
        k=1,
        n_samples=1,
        n_graphs=2,
        run_id="run_val_2",
        output_dir=out_dir,
        cache_dir=cache_dir,
        lock_dir=lock_dir,
        force_unlock=False,
        families_filter=["gnp_d0.10_n50-100"],
    )
    assert res2["summary"]["force_unlock_used"] is False


# ==============================================================================
# 6. RUN-ID PROTECTION & VALIDATION TESTS
# ==============================================================================

def test_refuse_overwriting_existing_run_id(tmp_path):
    """Harness must refuse to overwrite an existing run_id."""
    out_dir = tmp_path / "runs_eval"
    run_id = "test_collision"
    (out_dir / run_id).mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileExistsError, match="already exists"):
        run_evaluation(
            backend_name="mock",
            split_name="val",
            run_id=run_id,
            output_dir=out_dir,
        )


def test_k_greater_than_n_samples_raises_error(tmp_path):
    """Assert k > n_samples raises ValueError."""
    with pytest.raises(ValueError, match="k .* cannot exceed n_samples"):
        run_evaluation(
            backend_name="mock",
            split_name="val",
            k=5,
            n_samples=2,
            run_id="test_invalid_k",
            output_dir=tmp_path / "runs",
        )
