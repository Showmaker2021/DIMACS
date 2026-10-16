"""Unit tests for SEEDS.md registry and configs/teacher_llm.yaml."""

import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = REPO_ROOT / "SEEDS.md"
TEACHER_CONFIG_PATH = REPO_ROOT / "configs" / "teacher_llm.yaml"

REQUIRED_NAMES = {
    "train",
    "val",
    "test",
    "teacher_val",
    "test_pilot",
    "sft_val_probe",
    "teacher_rescore",
    "test_interp",
    "test_extrap",
    "test_comp",
    "test_heldout_type",
    "probe",
    "rl_train_pool",
}

EXPECTED_FROZEN_RANGES = {
    "train": (0, 99),
    "val": (1000, 1019),
    "test": (2000, 2049),
    "teacher_val": (3000, 3019),
    "test_pilot": (4000, 4199),
    "probe": (9000, 9009),
}

EXPECTED_PROPOSED_RANGES = {
    "sft_val_probe": (5000, 5019),
    "teacher_rescore": (6000, 6049),
    "test_interp": (7000, 7049),
    "test_extrap": (7100, 7149),
    "test_comp": (7200, 7249),
    "test_heldout_type": (7300, 7349),
    "rl_train_pool": (100000, 199999),
}


def parse_seeds_table(file_path: Path):
    """Parse SEEDS.md markdown table into structured records."""
    assert file_path.is_file(), f"SEEDS.md not found at {file_path}"
    content = file_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    rows = []
    in_table = False
    for line in lines:
        if line.startswith("|") and line.endswith("|"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if not parts:
                continue
            if parts[0] == "name":
                in_table = True
                continue
            if in_table and all(c in "- :" for c in "".join(parts)):
                continue
            if in_table:
                assert len(parts) == 5, f"Expected 5 columns in row: {line}"
                name, start_str, end_str, status, purpose = parts
                start = int(start_str)
                end = int(end_str)
                assert status in {"frozen", "proposed"}, f"Invalid status: {status}"
                rows.append({
                    "name": name,
                    "start": start,
                    "end": end,
                    "status": status,
                    "purpose": purpose,
                })
    return rows


def test_seeds_file_exists():
    assert SEEDS_PATH.is_file()


def test_seeds_table_every_row_parses():
    """Criterion (a): every row parses."""
    rows = parse_seeds_table(SEEDS_PATH)
    assert len(rows) == 13, f"Expected 13 rows, got {len(rows)}"
    for r in rows:
        assert isinstance(r["name"], str) and r["name"]
        assert isinstance(r["start"], int) and r["start"] >= 0
        assert isinstance(r["end"], int) and r["end"] >= 0
        assert r["status"] in {"frozen", "proposed"}
        assert isinstance(r["purpose"], str) and r["purpose"]


def test_seeds_start_less_than_equal_end():
    """Criterion (b): start <= end for every row."""
    rows = parse_seeds_table(SEEDS_PATH)
    for r in rows:
        assert r["start"] <= r["end"], f"Range inverted for {r['name']}: {r['start']} > {r['end']}"


def test_seeds_ranges_pairwise_disjoint():
    """Criterion (c): all ranges pairwise disjoint (including rl_train_pool vs everything)."""
    rows = parse_seeds_table(SEEDS_PATH)
    sorted_rows = sorted(rows, key=lambda x: x["start"])

    for i in range(len(sorted_rows) - 1):
        cur = sorted_rows[i]
        nxt = sorted_rows[i + 1]
        assert cur["end"] < nxt["start"], (
            f"Collision detected between {cur['name']} [{cur['start']}, {cur['end']}] "
            f"and {nxt['name']} [{nxt['start']}, {nxt['end']}]"
        )


def test_seeds_frozen_rows_match_spec():
    """Criterion (d): all frozen rows match the required values."""
    rows = parse_seeds_table(SEEDS_PATH)
    by_name = {r["name"]: r for r in rows}

    for name, (start, end) in EXPECTED_FROZEN_RANGES.items():
        assert name in by_name, f"Missing frozen range: {name}"
        r = by_name[name]
        assert r["status"] == "frozen", f"Expected frozen status for {name}, got {r['status']}"
        assert (r["start"], r["end"]) == (start, end), (
            f"Frozen range mismatch for {name}: expected ({start}, {end}), got ({r['start']}, {r['end']})"
        )

    for name, (start, end) in EXPECTED_PROPOSED_RANGES.items():
        assert name in by_name, f"Missing proposed range: {name}"
        r = by_name[name]
        assert r["status"] == "proposed", f"Expected proposed status for {name}, got {r['status']}"
        assert (r["start"], r["end"]) == (start, end), (
            f"Proposed range mismatch for {name}: expected ({start}, {end}), got ({r['start']}, {r['end']})"
        )


def test_seeds_all_required_names_present():
    """Criterion (e): all required names are present."""
    rows = parse_seeds_table(SEEDS_PATH)
    present_names = {r["name"] for r in rows}
    assert present_names == REQUIRED_NAMES, (
        f"Missing names: {REQUIRED_NAMES - present_names}, Unexpected names: {present_names - REQUIRED_NAMES}"
    )


SECRET_KEY_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z-_]{30,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{30,}"),
]


def validate_teacher_llm_config_content(content: str) -> dict:
    """Validate teacher_llm YAML content according to specification rules.

    Raises ValueError on missing keys, invalid types/ranges, or embedded secrets.
    """
    for pattern in SECRET_KEY_PATTERNS:
        match = pattern.search(content)
        if match:
            raise ValueError(f"Found forbidden pattern resembling an API key/secret: {match.group(0)}")

    try:
        import yaml
        config = yaml.safe_load(content)
    except ImportError:
        from ruamel.yaml import YAML
        y = YAML(typ="safe")
        config = y.load(content)

    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping/dict")

    required_keys = {"provider", "model", "base_temperature", "max_tokens", "rate_limit", "retry", "api_key_env"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Missing required keys in teacher_llm config: {missing}")

    if not (isinstance(config["provider"], str) and config["provider"].strip()):
        raise ValueError("Field 'provider' must be a non-empty string")

    if not (isinstance(config["model"], str) and config["model"].strip()):
        raise ValueError("Field 'model' must be a non-empty string")

    if not isinstance(config["base_temperature"], (int, float)) or not (0.0 <= config["base_temperature"] <= 2.0):
        raise ValueError(f"Field 'base_temperature' must be numeric in [0.0, 2.0], got {config.get('base_temperature')}")

    if not isinstance(config["max_tokens"], int) or config["max_tokens"] <= 0:
        raise ValueError(f"Field 'max_tokens' must be a positive integer, got {config.get('max_tokens')}")

    if not isinstance(config["rate_limit"], dict):
        raise ValueError("Field 'rate_limit' must be a mapping")
    if not (isinstance(config["rate_limit"].get("requests_per_minute"), int) and config["rate_limit"]["requests_per_minute"] > 0):
        raise ValueError("Field 'rate_limit.requests_per_minute' must be a positive integer")
    if not (isinstance(config["rate_limit"].get("pacing_seconds"), (int, float)) and config["rate_limit"]["pacing_seconds"] >= 0):
        raise ValueError("Field 'rate_limit.pacing_seconds' must be non-negative")

    if not isinstance(config["retry"], dict):
        raise ValueError("Field 'retry' must be a mapping")
    if not (isinstance(config["retry"].get("max_retries"), int) and config["retry"]["max_retries"] >= 0):
        raise ValueError("Field 'retry.max_retries' must be a non-negative integer")
    if not (isinstance(config["retry"].get("backoff_factor"), (int, float)) and config["retry"]["backoff_factor"] >= 1.0):
        raise ValueError("Field 'retry.backoff_factor' must be >= 1.0")

    api_key_env = config["api_key_env"]
    if not (isinstance(api_key_env, str) and re.match(r"^[A-Z_][A-Z0-9_]*$", api_key_env)):
        raise ValueError(f"Field 'api_key_env' must be an uppercase environment variable name, got '{api_key_env}'")

    return config


def test_teacher_llm_yaml_parses_and_secure():
    """Verify configs/teacher_llm.yaml parses, has required fields, and contains no secret keys."""
    assert TEACHER_CONFIG_PATH.is_file(), f"Config file not found at {TEACHER_CONFIG_PATH}"
    content = TEACHER_CONFIG_PATH.read_text(encoding="utf-8")
    config = validate_teacher_llm_config_content(content)
    assert isinstance(config, dict)


@pytest.mark.parametrize(
    "invalid_snippet, error_match",
    [
        ("provider: 'gemini'", "Missing required keys"),
        ("provider: ''\nmodel: 'gemini'\nbase_temperature: 0.7\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'KEY'", "provider"),
        ("provider: 'gemini'\nmodel: ''\nbase_temperature: 0.7\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'KEY'", "model"),
        ("provider: 'gemini'\nmodel: 'm'\nbase_temperature: 3.5\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'KEY'", "base_temperature"),
        ("provider: 'gemini'\nmodel: 'm'\nbase_temperature: -0.1\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'KEY'", "base_temperature"),
        ("provider: 'gemini'\nmodel: 'm'\nbase_temperature: 0.7\nmax_tokens: -100\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'KEY'", "max_tokens"),
        ("provider: 'gemini'\nmodel: 'm'\nbase_temperature: 0.7\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 0\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'KEY'", "requests_per_minute"),
        ("provider: 'gemini'\nmodel: 'm'\nbase_temperature: 0.7\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 0.5\napi_key_env: 'KEY'", "backoff_factor"),
        ("provider: 'gemini'\nmodel: 'm'\nbase_temperature: 0.7\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'lowercase_var'", "api_key_env"),
        ("provider: 'gemini'\nmodel: 'm'\nbase_temperature: 0.7\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'sk-proj-123456789012345678901234'", "pattern resembling an API key"),
        ("provider: 'gemini'\nmodel: 'm'\nbase_temperature: 0.7\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'AIzaSyA1234567890123456789012345678901'", "pattern resembling an API key"),
        ("provider: 'gemini'\nmodel: 'm'\nbase_temperature: 0.7\nmax_tokens: 1000\nrate_limit:\n  requests_per_minute: 10\n  pacing_seconds: 5\nretry:\n  max_retries: 3\n  backoff_factor: 1.5\napi_key_env: 'ghp_123456789012345678901234567890123456'", "pattern resembling an API key"),
    ],
)
def test_teacher_llm_validation_rejects_invalid_configs(invalid_snippet, error_match):
    """Verify that malformed configs and embedded secrets are rejected with ValueError."""
    with pytest.raises(ValueError, match=error_match):
        validate_teacher_llm_config_content(invalid_snippet)
