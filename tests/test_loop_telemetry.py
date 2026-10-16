"""Unit tests for telemetry and cost logging in gate/evolved/loop.py."""

import json
import os
import tempfile
from pathlib import Path
import pytest
import numpy as np

from gate.evolved.loop import run_evolution, call_llm_with_retry


def test_call_llm_with_retry_mock_return():
    """Verify call_llm_with_retry with mock=True returns code, usage, cost, and attempts."""
    pricing = {"mock": {"input_per_million": 0.14, "output_per_million": 0.28}}
    code, usage, cost, attempts = call_llm_with_retry(
        model="mock",
        system_prompt="system",
        user_prompt="user",
        pricing=pricing,
        mock=True,
    )
    assert isinstance(code, str) and "def priority" in code
    assert isinstance(usage, dict)
    assert usage.get("prompt_tokens") == 1950
    assert usage.get("completion_tokens") == 350
    assert isinstance(cost, float) and cost > 0.0
    assert attempts == 1


def test_loop_telemetry_fields_logged():
    """Verify run_evolution logs prompt_tokens, completion_tokens, latency_s, operator, temperature, status, cost_dollars."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "test_telemetry.jsonl"

        # Run 2 mock iterations
        summary = run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=2,
            mock=True,
            log_path=str(log_path),
            run_id="test_telemetry_run",
            selection_dir=None,
            is_smoke=True,
        )

        assert log_path.is_file(), f"Log file not created at {log_path}"
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2, f"Expected 2 log entries, got {len(lines)}"

        for i, line in enumerate(lines):
            entry = json.loads(line)

            # 1. Telemetry fields presence & types
            assert "prompt_tokens" in entry, f"Missing prompt_tokens in entry {i}"
            assert isinstance(entry["prompt_tokens"], int)
            assert entry["prompt_tokens"] > 0

            assert "completion_tokens" in entry, f"Missing completion_tokens in entry {i}"
            assert isinstance(entry["completion_tokens"], int)
            assert entry["completion_tokens"] > 0

            assert "latency_s" in entry, f"Missing latency_s in entry {i}"
            assert isinstance(entry["latency_s"], float)
            assert entry["latency_s"] >= 0.0

            assert "operator" in entry, f"Missing operator in entry {i}"
            assert isinstance(entry["operator"], str) and entry["operator"] in {"explore", "crossover", "refine"}

            assert "temperature" in entry, f"Missing temperature in entry {i}"
            assert isinstance(entry["temperature"], (int, float))
            assert 0.0 <= entry["temperature"] <= 2.0

            assert "status" in entry, f"Missing status in entry {i}"
            assert isinstance(entry["status"], str)
            assert entry["status"] == "ok" or "reject" in entry["status"].lower()

            assert "cost_dollars" in entry, f"Missing cost_dollars in entry {i}"
            assert entry["cost_dollars"] is None or (isinstance(entry["cost_dollars"], float) and entry["cost_dollars"] >= 0.0)

            # 2. Backward compatibility fields
            assert "valid" in entry and isinstance(entry["valid"], bool)
            assert "rejection_reason" in entry
            assert "tokens" in entry and isinstance(entry["tokens"], dict)
            assert "timestamp" in entry and isinstance(entry["timestamp"], float)


def test_loop_telemetry_status_on_rejection(monkeypatch):
    """Verify that when candidate validation fails, status correctly logs the rejection reason."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "test_rejection_telemetry.jsonl"

        # Mock call_llm_with_retry to return invalid python code
        import gate.evolved.loop as loop_mod

        def mock_bad_call(*args, **kwargs):
            code = "this is not valid python syntax !!!"
            usage = {"prompt_tokens": 100, "completion_tokens": 50}
            cost = 0.0001
            return code, usage, cost, 1

        monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_bad_call)

        run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=1,
            mock=True,
            log_path=str(log_path),
            run_id="test_rejection_run",
            selection_dir=None,
            is_smoke=True,
        )

        assert log_path.is_file()
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])

        assert entry["valid"] is False
        assert entry["status"] != "ok"
        assert "prompt_tokens" in entry and entry["prompt_tokens"] == 100
        assert "completion_tokens" in entry and entry["completion_tokens"] == 50
        assert "latency_s" in entry and isinstance(entry["latency_s"], float)
        assert entry["cost_dollars"] == 0.0001


def test_loop_telemetry_none_cost(monkeypatch):
    """Verify that when call_cost is None, loop handles it gracefully and logs cost_dollars as None."""
    import gate.evolved.loop as loop_mod

    def mock_none_cost(*args, **kwargs):
        code = "def priority(adj, colors, sat, deg, uncolored_deg):\n    return sat.astype(float)\n"
        usage = {"prompt_tokens": 80, "completion_tokens": 40}
        cost = None
        return code, usage, cost, 1

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_none_cost)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "test_none_cost.jsonl"
        summary = run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=1,
            mock=True,
            log_path=str(log_path),
            run_id="test_none_cost_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert log_path.is_file()
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["cost_dollars"] is None
        assert entry["prompt_tokens"] == 80
        assert entry["completion_tokens"] == 40
        assert summary["total_cost"] == 0.0


def test_loop_telemetry_none_usage(monkeypatch):
    """Verify that when usage is None, loop does not crash and logs zero tokens."""
    import gate.evolved.loop as loop_mod

    def mock_none_usage(*args, **kwargs):
        code = "def priority(adj, colors, sat, deg, uncolored_deg):\n    return sat.astype(float)\n"
        usage = None
        cost = 0.0005
        return code, usage, cost, 1

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_none_usage)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "test_none_usage.jsonl"
        summary = run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=1,
            mock=True,
            log_path=str(log_path),
            run_id="test_none_usage_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert log_path.is_file()
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["prompt_tokens"] == 0
        assert entry["completion_tokens"] == 0
        assert entry["cost_dollars"] == 0.0005


def test_loop_telemetry_llm_exception_logged(monkeypatch):
    """Verify that an exception raised by call_llm_with_retry is logged to telemetry before breaking."""
    import gate.evolved.loop as loop_mod

    def mock_fail(*args, **kwargs):
        raise ConnectionResetError("Remote server closed connection unexpectedly")

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", mock_fail)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "test_exception.jsonl"
        summary = run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=1,
            mock=True,
            log_path=str(log_path),
            run_id="test_exception_run",
            selection_dir=None,
            is_smoke=True,
        )
        assert log_path.is_file()
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "llm_error" in entry["status"]
        assert entry["valid"] is False
        assert entry["prompt_tokens"] == 0
        assert entry["completion_tokens"] == 0
        assert entry["cost_dollars"] is None
        assert isinstance(entry["latency_s"], float)

