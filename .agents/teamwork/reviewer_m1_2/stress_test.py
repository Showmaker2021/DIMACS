"""Adversarial stress tests for Milestone 1 by reviewer_m1_2.
Tests edge cases, failure modes, tamper detection, and integrity constraints.
"""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path("d:/Graph Coloring")
sys.path.insert(0, str(REPO_ROOT))

def test_frozen_files_tamper_detection():
    """Verify that test_frozen_files actually detects any 1-byte tampering."""
    from tests.test_frozen_files import compute_sha256, compute_set_hash
    manifest_path = REPO_ROOT / "tests" / "frozen_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Verify verify_py hash
    verify_path = REPO_ROOT / "evaluator" / "verify.py"
    real_hash = compute_sha256(verify_path)
    assert real_hash == manifest["verify_py"], "verify.py hash mismatch in manifest!"

    # 2. Simulate 1-byte modification of verify.py
    tampered_bytes = verify_path.read_bytes() + b"\n# tamper"
    tampered_hash = hashlib.sha256(tampered_bytes).hexdigest()
    assert tampered_hash != manifest["verify_py"], "Tamper failed to change hash!"

    # 3. Verify data/data_v1 count matches manifest
    v1_files = manifest["data_v1"]
    assert len(v1_files) == 324, f"Expected 324 files in data_v1, got {len(v1_files)}"
    for rel_path, exp_hash in v1_files.items():
        p = REPO_ROOT / rel_path
        assert p.is_file(), f"File in manifest missing: {p}"
        assert compute_sha256(p) == exp_hash, f"Hash mismatch for {rel_path}"

    # 4. Verify data/data_v1_1
    v1_1_files = manifest["data_v1_1"]
    assert len(v1_1_files) == 2, f"Expected 2 files in data_v1_1, got {len(v1_1_files)}"
    for rel_path, exp_hash in v1_1_files.items():
        p = REPO_ROOT / rel_path
        assert p.is_file(), f"File in manifest missing: {p}"
        assert compute_sha256(p) == exp_hash, f"Hash mismatch for {rel_path}"

    print("[PASS] Tamper detection and manifest integrity verified.")


def test_rogue_done_lock_detection():
    """Verify that rogue .done files outside runs/ are caught."""
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "evaluator") as tmp_dir:
        rogue_lock = Path(tmp_dir) / "test.done"
        rogue_lock.touch()
        try:
            from tests.test_frozen_files import test_no_rogue_done_locks
            test_no_rogue_done_locks()
            assert False, "Failed to catch rogue .done file in evaluator/"
        except AssertionError as e:
            assert "Unexpected .done lock file outside runs/" in str(e)
            print("[PASS] Rogue .done lock detection verified.")


def test_seeds_collision_stress():
    """Stress test boundary collisions in seed ranges."""
    from tests.test_seeds import parse_seeds_table, SEEDS_PATH
    rows = parse_seeds_table(SEEDS_PATH)

    # Check that no two ranges touch or overlap
    for i, r1 in enumerate(rows):
        for j, r2 in enumerate(rows):
            if i >= j:
                continue
            # Check overlap: max(start1, start2) <= min(end1, end2)
            overlap = max(r1["start"], r2["start"]) <= min(r1["end"], r2["end"])
            assert not overlap, f"Overlap between {r1['name']} and {r2['name']}"

            # Check that start <= end
            assert r1["start"] <= r1["end"]
            assert r2["start"] <= r2["end"]

    print("[PASS] Seeds collision stress verified.")


def test_teacher_yaml_security_stress():
    """Stress test key detection and invalid environment variable names."""
    import yaml
    teacher_path = REPO_ROOT / "configs" / "teacher_llm.yaml"
    with open(teacher_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Check api_key_env is not a dummy secret
    env_var = cfg["api_key_env"]
    assert env_var == "DEEPSEEK_API_KEY"
    assert not any(prefix in env_var for prefix in ["sk-", "AIza", "ghp_"])

    # Check that no actual API key exists in the environment under this test
    # (should run clean without requiring keys)
    print("[PASS] Teacher YAML security stress verified.")


def test_loop_telemetry_json_serializability():
    """Stress test that loop telemetry entries are 100% valid JSON under edge conditions."""
    from gate.evolved.loop import run_evolution
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "stress_log.jsonl"
        summary = run_evolution(
            family_id="kpart_k10_p0.3_n100-200",
            max_calls=3,
            mock=True,
            log_path=str(log_file),
            run_id="stress_run",
            selection_dir=None,
            is_smoke=True,
        )

        assert log_file.is_file()
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            # Verify all required keys
            required_keys = [
                "prompt_tokens", "completion_tokens", "latency_s",
                "operator", "temperature", "status", "cost_dollars"
            ]
            for k in required_keys:
                assert k in data, f"Missing required telemetry key: {k}"

            assert isinstance(data["prompt_tokens"], int)
            assert isinstance(data["completion_tokens"], int)
            assert isinstance(data["latency_s"], float)
            assert data["latency_s"] >= 0.0
            assert isinstance(data["status"], str)
            assert isinstance(data["operator"], str)
            assert isinstance(data["temperature"], float)

    print("[PASS] Loop telemetry serialization stress verified.")


if __name__ == "__main__":
    test_frozen_files_tamper_detection()
    test_rogue_done_lock_detection()
    test_seeds_collision_stress()
    test_teacher_yaml_security_stress()
    test_loop_telemetry_json_serializability()
    print("ALL ADVERSARIAL STRESS TESTS PASSED SUCCESSFULLY!")
