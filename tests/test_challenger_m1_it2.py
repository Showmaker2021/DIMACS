"""Comprehensive empirical re-verification and stress tests for M1 fixes.

Targeting all 4 failure scenarios and edge combinations:
1. call_cost is None (and negative / mixed / cumulative)
2. usage is None (and non-dict / missing keys / edge types)
3. LLM exception during call (on iter 1, iter 2, various exceptions)
4. Resume with null / missing cost_dollars (and mixed records, population integrity)
5. End-to-end multi-step lifecycle combining all 4 conditions
"""

import json
import tempfile
from pathlib import Path
import pytest
import gate.evolved.loop as loop_mod


VALID_CODE = (
    "def priority(adj, colors, sat, deg, uncolored_deg):\n"
    "    return sat.astype(float) * 100.0 + uncolored_deg.astype(float)\n"
)


# =========================================================================
# Scenario 1: call_cost is None / edge cases
# =========================================================================

def test_scenario1_mixed_none_and_valid_costs(monkeypatch):
    """Test sequential iterations with mixed None and positive costs."""
    call_idx = 0
    costs = [0.005, None, 0.003]

    def mock_call(*args, **kwargs):
        nonlocal call_idx
        c = costs[call_idx % len(costs)]
        call_idx += 1
        return VALID_CODE, {"prompt_tokens": 100, "completion_tokens": 50}, c, 1

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_call)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "mixed_cost.jsonl"
        summary = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=3,
            mock=True,
            log_path=str(log_path),
            run_id="mixed_cost_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert log_path.is_file()
        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3

        assert lines[0]["cost_dollars"] == pytest.approx(0.005)
        assert lines[1]["cost_dollars"] is None
        assert lines[2]["cost_dollars"] == pytest.approx(0.003)

        # Cumulative cost checks
        assert lines[0]["cumulative_cost"] == pytest.approx(0.005)
        assert lines[1]["cumulative_cost"] == pytest.approx(0.005)
        assert lines[2]["cumulative_cost"] == pytest.approx(0.008)
        assert summary["total_cost"] == pytest.approx(0.008)


def test_scenario1_negative_cost_logged_as_none(monkeypatch):
    """Verify negative cost (e.g. calculation artifact) is logged as None."""
    def mock_call(*args, **kwargs):
        return VALID_CODE, {"prompt_tokens": 50, "completion_tokens": 20}, -0.05, 1

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_call)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "neg_cost.jsonl"
        summary = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=1,
            mock=True,
            log_path=str(log_path),
            run_id="neg_cost_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert log_path.is_file()
        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["cost_dollars"] is None


# =========================================================================
# Scenario 2: usage is None / non-dict structures
# =========================================================================

@pytest.mark.parametrize("bad_usage", [None, "invalid_usage_string", [100, 50], False])
def test_scenario2_non_dict_usage_handled_safely(monkeypatch, bad_usage):
    """Verify usage that is None, string, list, or bool does not crash and logs zero tokens."""
    def mock_call(*args, **kwargs):
        return VALID_CODE, bad_usage, 0.001, 1

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_call)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / f"bad_usage.jsonl"
        summary = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=1,
            mock=True,
            log_path=str(log_path),
            run_id="bad_usage_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert log_path.is_file()
        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["prompt_tokens"] == 0
        assert lines[0]["completion_tokens"] == 0
        assert lines[0]["tokens"] == {}


def test_scenario2_usage_missing_keys(monkeypatch):
    """Verify usage with missing keys defaults to 0 tokens."""
    def mock_call(*args, **kwargs):
        # usage dict without completion_tokens
        return VALID_CODE, {"prompt_tokens": 75}, 0.001, 1

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_call)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "missing_keys_usage.jsonl"
        summary = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=1,
            mock=True,
            log_path=str(log_path),
            run_id="missing_keys_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert log_path.is_file()
        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["prompt_tokens"] == 75
        assert lines[0]["completion_tokens"] == 0


# =========================================================================
# Scenario 3: LLM exception handling
# =========================================================================

def test_scenario3_exception_on_second_iteration(monkeypatch):
    """Verify exception on iteration 2 logs correctly and preserves iteration 1 state."""
    call_count = 0

    def mock_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return VALID_CODE, {"prompt_tokens": 100, "completion_tokens": 50}, 0.002, 1
        else:
            raise TimeoutError("LLM API endpoint request timed out after 45s")

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_call)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "iter2_fail.jsonl"
        summary = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=3,
            mock=True,
            log_path=str(log_path),
            run_id="iter2_fail_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert log_path.is_file()
        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

        # Iteration 1 succeeded
        assert lines[0]["iteration"] == 1
        assert lines[0]["status"] == "ok"
        assert lines[0]["valid"] is True
        assert lines[0]["cost_dollars"] == pytest.approx(0.002)

        # Iteration 2 logged error before terminating
        assert lines[1]["iteration"] == 2
        assert "llm_error" in lines[1]["status"]
        assert "timed out" in lines[1]["status"]
        assert lines[1]["valid"] is False
        assert lines[1]["cost_dollars"] is None
        assert lines[1]["cumulative_cost"] == pytest.approx(0.002)
        assert lines[1]["prompt_tokens"] == 0
        assert lines[1]["completion_tokens"] == 0
        assert isinstance(lines[1]["latency_s"], float)


@pytest.mark.parametrize("exc", [
    RuntimeError("Exhausted all retries"),
    ConnectionResetError("Connection reset by peer"),
    ValueError("Malformed HTTP response"),
])
def test_scenario3_various_exceptions_logged(monkeypatch, exc):
    """Verify different exception types are captured in telemetry without unhandled crash."""
    def mock_call(*args, **kwargs):
        raise exc

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_call)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "exc_test.jsonl"
        summary = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=1,
            mock=True,
            log_path=str(log_path),
            run_id="exc_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert log_path.is_file()
        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["valid"] is False
        assert "llm_error" in lines[0]["status"]
        assert str(exc) in lines[0]["status"]


# =========================================================================
# Scenario 4: Resume with null / missing cost_dollars and population integrity
# =========================================================================

def test_scenario4_resume_mixed_records_and_population_integrity():
    """Verify resuming multiple records with null, float, and missing cost_dollars."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "multi_resume.jsonl"
        records = [
            # Record 1: cost is null
            {
                "family_id": "kpart_k10_p0.3_n100-200",
                "run_id": "multi_resume_run",
                "model": "gemini-flash-lite-latest",
                "mock": True,
                "iteration": 1,
                "operator": "explore",
                "temperature": 0.9,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "latency_s": 0.1,
                "status": "ok",
                "code": "def priority(adj, colors, sat, deg, uncolored_deg):\n    return sat.astype(float)\n",
                "valid": True,
                "cost_dollars": None,
                "tokens": {"prompt_tokens": 100, "completion_tokens": 50},
                "train_counts": [15, 15],
            },
            # Record 2: cost is 0.004
            {
                "family_id": "kpart_k10_p0.3_n100-200",
                "run_id": "multi_resume_run",
                "model": "gemini-flash-lite-latest",
                "mock": True,
                "iteration": 2,
                "operator": "refine",
                "temperature": 0.3,
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "latency_s": 0.15,
                "status": "ok",
                "code": "def priority(adj, colors, sat, deg, uncolored_deg):\n    return deg.astype(float)\n",
                "valid": True,
                "cost_dollars": 0.004,
                "tokens": {"prompt_tokens": 120, "completion_tokens": 60},
                "train_counts": [16, 16],
            },
            # Record 3: cost_dollars missing from dict entirely
            {
                "family_id": "kpart_k10_p0.3_n100-200",
                "run_id": "multi_resume_run",
                "model": "gemini-flash-lite-latest",
                "mock": True,
                "iteration": 3,
                "operator": "explore",
                "temperature": 0.9,
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "latency_s": 0.08,
                "status": "rejected",
                "code": None,
                "valid": False,
                "rejection_reason": "SyntaxError",
            },
        ]
        with open(log_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Now resume and execute 1 more call (max_calls=4)
        summary = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=4,
            mock=True,
            log_path=str(log_path),
            run_id="multi_resume_run",
            selection_dir=None,
            is_smoke=True,
        )

        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        # 3 resumed + 1 new = 4 lines total
        assert len(lines) == 4, f"Expected 4 lines total, got {len(lines)}"
        assert lines[3]["iteration"] == 4
        # Total cost should include record 2 ($0.004) + whatever call 4 cost
        assert summary["total_cost"] >= 0.004


# =========================================================================
# Scenario 5: Full combined lifecycle stress test
# =========================================================================

def test_full_lifecycle_combined_stress(monkeypatch):
    """End-to-end multi-step lifecycle combining all 4 conditions.
    
    Step 1: Run 2 calls where call 1 has usage+cost, call 2 has None cost & None usage.
    Step 2: Resume log file, run call 3 which fails with an LLM exception.
    Step 3: Assert log file contains 3 records with exact expected telemetry attributes.
    """
    call_step = 0

    def mock_step_call(*args, **kwargs):
        nonlocal call_step
        call_step += 1
        if call_step == 1:
            return VALID_CODE, {"prompt_tokens": 150, "completion_tokens": 75}, 0.0035, 1
        elif call_step == 2:
            return VALID_CODE, None, None, 1
        else:
            raise ConnectionRefusedError("Simulated LLM outage")

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_step_call)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "lifecycle.jsonl"

        # Phase 1: Calls 1 and 2
        summary1 = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=2,
            mock=True,
            log_path=str(log_path),
            run_id="lifecycle_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert summary1["total_cost"] == pytest.approx(0.0035)

        # Phase 2: Resume for call 3 (which will raise exception)
        summary2 = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=3,
            mock=True,
            log_path=str(log_path),
            run_id="lifecycle_run",
            selection_dir=None,
            is_smoke=True,
        )

        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3, f"Expected 3 records in log, found {len(lines)}"

        # Verify record 1 (standard)
        assert lines[0]["iteration"] == 1
        assert lines[0]["status"] == "ok"
        assert lines[0]["cost_dollars"] == pytest.approx(0.0035)
        assert lines[0]["prompt_tokens"] == 150

        # Verify record 2 (none cost & none usage)
        assert lines[1]["iteration"] == 2
        assert lines[1]["status"] == "ok"
        assert lines[1]["cost_dollars"] is None
        assert lines[1]["prompt_tokens"] == 0
        assert lines[1]["completion_tokens"] == 0
        assert lines[1]["cumulative_cost"] == pytest.approx(0.0035)

        # Verify record 3 (LLM exception after resume)
        assert lines[2]["iteration"] == 3
        assert lines[2]["valid"] is False
        assert "llm_error" in lines[2]["status"]
        assert "Simulated LLM outage" in lines[2]["status"]
        assert lines[2]["cost_dollars"] is None
        assert lines[2]["cumulative_cost"] == pytest.approx(0.0035)
        assert lines[2]["prompt_tokens"] == 0
        assert lines[2]["completion_tokens"] == 0
        assert isinstance(lines[2]["latency_s"], float)
