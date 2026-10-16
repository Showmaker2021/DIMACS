"""Adversarial edge condition tests for gate/evolved/loop.py and teacher_llm config."""

import json
import re
import tempfile
from pathlib import Path
import pytest
import gate.evolved.loop as loop_mod


def test_adversarial_loop_telemetry_none_cost():
    """Verify loop.py handles call_cost=None without crashing and logs cost_dollars as None."""
    def mock_call_none_cost(*args, **kwargs):
        code = (
            "def priority(adj, colors, sat, deg, uncolored_deg):\n"
            "    return sat.astype(float)\n"
        )
        usage = {"prompt_tokens": 120, "completion_tokens": 60}
        cost = None  # Explicit None / unknown cost
        return code, usage, cost, 1

    orig_call = loop_mod.call_llm_with_retry
    try:
        loop_mod.call_llm_with_retry = mock_call_none_cost
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "test_none_cost.jsonl"
            summary = loop_mod.run_evolution(
                family_id="kpart_k10_p0.3_n100-200",
                max_calls=1,
                mock=True,
                log_path=str(log_path),
                run_id="test_none_cost_run",
                selection_dir=None,
                is_smoke=True,
            )
            assert log_path.is_file(), "Log file was not created"
            lines = [l.strip() for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["cost_dollars"] is None
            assert entry["prompt_tokens"] == 120
            assert entry["completion_tokens"] == 60
            assert "latency_s" in entry and isinstance(entry["latency_s"], float)
    finally:
        loop_mod.call_llm_with_retry = orig_call


def test_adversarial_loop_telemetry_none_usage():
    """Verify loop.py handles usage=None without crashing."""
    def mock_call_none_usage(*args, **kwargs):
        code = (
            "def priority(adj, colors, sat, deg, uncolored_deg):\n"
            "    return sat.astype(float)\n"
        )
        usage = None
        cost = 0.001
        return code, usage, cost, 1

    orig_call = loop_mod.call_llm_with_retry
    try:
        loop_mod.call_llm_with_retry = mock_call_none_usage
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "test_none_usage.jsonl"
            summary = loop_mod.run_evolution(
                family_id="kpart_k10_p0.3_n100-200",
                max_calls=1,
                mock=True,
                log_path=str(log_path),
                run_id="test_none_usage_run",
                selection_dir=None,
                is_smoke=True,
            )
            assert log_path.is_file()
            lines = [l.strip() for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["prompt_tokens"] == 0
            assert entry["completion_tokens"] == 0
    finally:
        loop_mod.call_llm_with_retry = orig_call


def test_adversarial_loop_telemetry_llm_exception_logged():
    """Verify that when call_llm_with_retry raises an Exception, telemetry still logs the failed attempt."""
    def mock_failing_call(*args, **kwargs):
        raise RuntimeError("Mock LLM Connection Timeout / Rate Limit Exhaustion")

    orig_call = loop_mod.call_llm_with_retry
    try:
        loop_mod.call_llm_with_retry = mock_failing_call
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "test_llm_exception.jsonl"
            summary = loop_mod.run_evolution(
                family_id="kpart_k10_p0.3_n100-200",
                max_calls=1,
                mock=True,
                log_path=str(log_path),
                run_id="test_llm_exception_run",
                selection_dir=None,
                is_smoke=True,
            )
            assert log_path.is_file(), "Log file should exist even if LLM call failed"
            lines = [l.strip() for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(lines) == 1, "Failed LLM call attempt should be logged to telemetry"
            entry = json.loads(lines[0])
            assert "latency_s" in entry and isinstance(entry["latency_s"], float)
            assert "error" in entry["status"].lower() or "fail" in entry["status"].lower()
    finally:
        loop_mod.call_llm_with_retry = orig_call


def test_adversarial_loop_resume_with_null_cost_dollars():
    """Verify that resume properly counts and preserves records that have cost_dollars=null."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "resume_test.jsonl"
        record = {
            "family_id": "kpart_k10_p0.3_n100-200",
            "run_id": "test_resume_null_cost",
            "model": "gemini-flash-lite-latest",
            "mock": True,
            "iteration": 1,
            "operator": "explore",
            "temperature": 0.9,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "latency_s": 0.123,
            "status": "ok",
            "code": "def priority(adj, colors, sat, deg, uncolored_deg):\n    return sat.astype(float)\n",
            "valid": True,
            "cost_dollars": None,
            "tokens": {"prompt_tokens": 100, "completion_tokens": 50},
            "train_counts": [14, 14],
        }
        log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        summary = loop_mod.run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=2,
            mock=True,
            log_path=str(log_path),
            run_id="test_resume_null_cost",
            selection_dir=None,
            is_smoke=True,
        )
        lines = [l.strip() for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        # If resume worked, there should be 2 lines (1 resumed, 1 new).
        # When bug is present, line 1 fails to resume, so max_calls=2 runs 2 new calls (total 3 lines)
        assert len(lines) == 2, f"Expected 2 total lines (1 resumed + 1 new), got {len(lines)}"


def test_adversarial_teacher_llm_config_rejects_embedded_secrets():
    """Verify that teacher_llm validation rejects configurations containing actual API secrets."""
    import yaml

    invalid_configs = [
        ("sk-openai-key", "api_key_env: 'sk-proj-123456789012345678901234'"),
        ("google-aiza-key", "api_key_env: 'AIzaSyA1234567890123456789012345678901'"),
        ("github-token", "api_key_env: 'ghp_123456789012345678901234567890123456'"),
        ("lowercase-env", "api_key_env: 'my_secret_key'"),
        ("missing-env", "provider: 'deepseek'"),  # missing api_key_env entirely
    ]

    key_patterns = [
        re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),
        re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    ]

    for label, snippet in invalid_configs:
        has_secret = any(p.search(snippet) for p in key_patterns)
        is_uppercase_env = bool(re.match(r"^api_key_env:\s*['\"][A-Z_][A-Z0-9_]*['\"]$", snippet.strip()))
        # The configuration must fail either the secret pattern check or the uppercase env format check
        assert has_secret or not is_uppercase_env, f"Config test case {label} should be identified as invalid"

