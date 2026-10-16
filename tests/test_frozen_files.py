"""Integrity tests to guard frozen files and sandbox whitelist from accidental modification.

Guards:
- evaluator/verify.py
- data/data_v1/ (all 324 files)
- data/data_v1_1/ (all 2 files)
- Existing .done lock files
- gate/evolved/sandbox.py whitelist (FORBIDDEN_CALLS, ALLOWED_OBJECT_ATTRS, FORBIDDEN_ATTRS,
  ALLOWED_NP_ATTRS, SAFE_BUILTINS)
"""

import hashlib
import json
from pathlib import Path
import pytest

from gate.evolved.sandbox import (
    FORBIDDEN_CALLS,
    ALLOWED_OBJECT_ATTRS,
    FORBIDDEN_ATTRS,
    ALLOWED_NP_ATTRS,
    SAFE_BUILTINS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "tests" / "frozen_manifest.json"


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def compute_set_hash(items) -> str:
    """Compute deterministic SHA256 hex digest of a set of items."""
    h = hashlib.sha256()
    for item in sorted(items):
        h.update(str(item).encode("utf-8") + b"\n")
    return h.hexdigest()


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST_PATH.is_file(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_verify_py_unmodified(manifest):
    """Assert evaluator/verify.py exists and matches the frozen SHA256 manifest."""
    verify_path = REPO_ROOT / "evaluator" / "verify.py"
    assert verify_path.is_file(), "evaluator/verify.py is missing!"
    current_hash = compute_sha256(verify_path)
    expected_hash = manifest["verify_py"]
    assert current_hash == expected_hash, (
        f"evaluator/verify.py has been modified! "
        f"Expected {expected_hash}, got {current_hash}"
    )


def test_data_v1_1_unmodified(manifest):
    """Assert data/data_v1_1/ files exist and match their frozen SHA256 hashes."""
    data_dir = REPO_ROOT / "data" / "data_v1_1"
    assert data_dir.is_dir(), "data/data_v1_1 directory is missing!"

    expected_files = manifest["data_v1_1"]
    current_files = {
        f.relative_to(REPO_ROOT).as_posix(): compute_sha256(f)
        for f in sorted(data_dir.rglob("*"))
        if f.is_file()
    }

    assert set(current_files.keys()) == set(expected_files.keys()), (
        f"File list in data/data_v1_1 changed! "
        f"Missing: {set(expected_files.keys()) - set(current_files.keys())}, "
        f"Added: {set(current_files.keys()) - set(expected_files.keys())}"
    )

    for rel_path, expected_hash in expected_files.items():
        assert current_files[rel_path] == expected_hash, (
            f"File modified in data/data_v1_1: {rel_path}! "
            f"Expected {expected_hash}, got {current_files[rel_path]}"
        )


def test_data_v1_unmodified(manifest):
    """Assert data/data_v1/ files exist and match their frozen SHA256 hashes."""
    data_dir = REPO_ROOT / "data" / "data_v1"
    assert data_dir.is_dir(), "data/data_v1 directory is missing!"

    expected_files = manifest["data_v1"]
    current_files = {
        f.relative_to(REPO_ROOT).as_posix(): compute_sha256(f)
        for f in sorted(data_dir.rglob("*"))
        if f.is_file()
    }

    assert set(current_files.keys()) == set(expected_files.keys()), (
        f"File list in data/data_v1 changed! "
        f"Missing: {set(expected_files.keys()) - set(current_files.keys())}, "
        f"Added: {set(current_files.keys()) - set(expected_files.keys())}"
    )

    mismatches = []
    for rel_path, expected_hash in expected_files.items():
        if current_files[rel_path] != expected_hash:
            mismatches.append((rel_path, expected_hash, current_files[rel_path]))

    assert not mismatches, f"Modified files detected in data/data_v1: {mismatches}"


def test_no_rogue_done_locks():
    """Assert no unauthorized .done lock files exist in data or source directories."""
    # .done lock files are only valid inside runs/
    all_done_files = list(REPO_ROOT.rglob("*.done"))
    for done_file in all_done_files:
        rel = done_file.relative_to(REPO_ROOT).as_posix()
        assert rel.startswith("runs/"), f"Unexpected .done lock file outside runs/: {rel}"


def test_sandbox_whitelist_unmodified(manifest):
    """Assert gate/evolved/sandbox.py whitelist sets and keys are unmodified."""
    expected_whitelist = manifest["sandbox_whitelist"]
    expected_hashes = manifest["sandbox_hashes"]

    assert set(FORBIDDEN_CALLS) == set(expected_whitelist["FORBIDDEN_CALLS"])
    assert compute_set_hash(FORBIDDEN_CALLS) == expected_hashes["FORBIDDEN_CALLS"]

    assert set(ALLOWED_OBJECT_ATTRS) == set(expected_whitelist["ALLOWED_OBJECT_ATTRS"])
    assert compute_set_hash(ALLOWED_OBJECT_ATTRS) == expected_hashes["ALLOWED_OBJECT_ATTRS"]

    assert set(FORBIDDEN_ATTRS) == set(expected_whitelist["FORBIDDEN_ATTRS"])
    assert compute_set_hash(FORBIDDEN_ATTRS) == expected_hashes["FORBIDDEN_ATTRS"]

    assert set(ALLOWED_NP_ATTRS) == set(expected_whitelist["ALLOWED_NP_ATTRS"])
    assert compute_set_hash(ALLOWED_NP_ATTRS) == expected_hashes["ALLOWED_NP_ATTRS"]

    assert set(SAFE_BUILTINS.keys()) == set(expected_whitelist["SAFE_BUILTINS_KEYS"])
    assert compute_set_hash(SAFE_BUILTINS.keys()) == expected_hashes["SAFE_BUILTINS_KEYS"]
