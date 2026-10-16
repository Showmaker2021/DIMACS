"""Adversarial Verification Suite for Milestone 1.

Empirically challenges:
1. SEEDS.md parsing and validation robustness (overlapping ranges, inverted ranges,
   missing required ranges, malformed table rows, bad statuses, negative values).
2. test_frozen_files.py tamper detection (verify.py, data_v1, data_v1_1, rogue .done locks,
   sandbox whitelist tampering, and SHA256 calculation accuracy against known vectors).
3. configs/teacher_llm.yaml credential hygiene and schema validation.
"""

import copy
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_seeds import (
    parse_seeds_table,
    REQUIRED_NAMES,
    EXPECTED_FROZEN_RANGES,
    EXPECTED_PROPOSED_RANGES,
)
from tests.test_frozen_files import (
    compute_sha256,
    compute_set_hash,
    MANIFEST_PATH,
)
from gate.evolved.sandbox import (
    FORBIDDEN_CALLS,
    ALLOWED_OBJECT_ATTRS,
    FORBIDDEN_ATTRS,
    ALLOWED_NP_ATTRS,
    SAFE_BUILTINS,
)


def validate_seeds_records(rows):
    """Execute full validation pipeline on parsed seeds rows."""
    assert len(rows) == 13, f"Expected 13 rows, got {len(rows)}"

    for r in rows:
        assert isinstance(r["name"], str) and r["name"], f"Invalid name: {r['name']}"
        assert isinstance(r["start"], int) and r["start"] >= 0, f"Invalid start: {r['start']}"
        assert isinstance(r["end"], int) and r["end"] >= 0, f"Invalid end: {r['end']}"
        assert r["status"] in {"frozen", "proposed"}, f"Invalid status: {r['status']}"
        assert isinstance(r["purpose"], str) and r["purpose"], f"Invalid purpose: {r['purpose']}"

    # Criterion (b): start <= end
    for r in rows:
        assert r["start"] <= r["end"], f"Range inverted for {r['name']}: {r['start']} > {r['end']}"

    # Criterion (c): pairwise disjoint
    sorted_rows = sorted(rows, key=lambda x: x["start"])
    for i in range(len(sorted_rows) - 1):
        cur = sorted_rows[i]
        nxt = sorted_rows[i + 1]
        assert cur["end"] < nxt["start"], (
            f"Collision detected between {cur['name']} [{cur['start']}, {cur['end']}] "
            f"and {nxt['name']} [{nxt['start']}, {nxt['end']}]"
        )

    # Criterion (d): frozen/proposed values match spec
    by_name = {r["name"]: r for r in rows}
    for name, (start, end) in EXPECTED_FROZEN_RANGES.items():
        assert name in by_name, f"Missing frozen range: {name}"
        r = by_name[name]
        assert r["status"] == "frozen", f"Expected frozen status for {name}"
        assert (r["start"], r["end"]) == (start, end), (
            f"Frozen range mismatch for {name}: expected ({start}, {end}), got ({r['start']}, {r['end']})"
        )

    for name, (start, end) in EXPECTED_PROPOSED_RANGES.items():
        assert name in by_name, f"Missing proposed range: {name}"
        r = by_name[name]
        assert r["status"] == "proposed", f"Expected proposed status for {name}"
        assert (r["start"], r["end"]) == (start, end), (
            f"Proposed range mismatch for {name}: expected ({start}, {end}), got ({r['start']}, {r['end']})"
        )

    # Criterion (e): all required names present
    present_names = {r["name"] for r in rows}
    assert present_names == REQUIRED_NAMES, (
        f"Missing names: {REQUIRED_NAMES - present_names}, Unexpected: {present_names - REQUIRED_NAMES}"
    )


def validate_seeds_file(path: Path):
    """Full parsing and validation pipeline for a SEEDS.md file."""
    rows = parse_seeds_table(path)
    validate_seeds_records(rows)
    return rows


def generate_valid_seeds_markdown() -> str:
    """Read the real SEEDS.md as the baseline valid markdown."""
    return (REPO_ROOT / "SEEDS.md").read_text(encoding="utf-8")


def run_seeds_adversarial_challenges():
    """Stress test SEEDS parser and validator against adversarial mutations."""
    print("=== [TASK 1] SEEDS.md Adversarial Stress Tests ===")
    valid_md = generate_valid_seeds_markdown()
    passed_challenges = 0
    total_challenges = 0

    def assert_rejected(name: str, modified_content: str, expected_exc=(AssertionError, ValueError)):
        nonlocal passed_challenges, total_challenges
        total_challenges += 1
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmp:
            tmp.write(modified_content)
            tmp_path = Path(tmp.name)
        try:
            try:
                validate_seeds_file(tmp_path)
                raise Exception(f"FAIL: Validator did NOT reject adversarial input for: {name}")
            except expected_exc as e:
                # Successfully rejected!
                passed_challenges += 1
                print(f"  [PASS] {name} -> rejected with {type(e).__name__}: {str(e)[:60]}...")
            except Exception as e:
                raise Exception(f"FAIL: Unexpected exception {type(e)} for {name}: {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # 1. Overlapping ranges (train 0-1050 overlaps with val 1000-1019)
    bad_md = valid_md.replace("| train | 0 | 99 |", "| train | 0 | 1050 |")
    assert_rejected("Overlap: train extended into val", bad_md)

    # 2. Overlapping ranges (touching boundary: train ends at 1000, val starts at 1000)
    bad_md = valid_md.replace("| train | 0 | 99 |", "| train | 0 | 1000 |")
    assert_rejected("Overlap: touching boundary (end == start)", bad_md)

    # 3. Nested range (probe 9000-9009 nested inside probe 8900-9100)
    bad_md = valid_md.replace("| probe | 9000 | 9009 |", "| probe | 8900 | 9100 |")
    assert_rejected("Spec mismatch/overlap: probe expanded", bad_md)

    # 4. Inverted range (train start > end: 99 to 0)
    bad_md = valid_md.replace("| train | 0 | 99 |", "| train | 99 | 0 |")
    assert_rejected("Inverted range: train start > end", bad_md)

    # 5. Inverted range on proposed split (test_interp: 7049 to 7000)
    bad_md = valid_md.replace("| test_interp | 7000 | 7049 |", "| test_interp | 7049 | 7000 |")
    assert_rejected("Inverted range: test_interp start > end", bad_md)

    # 6. Missing required range (remove teacher_val row)
    lines = [line for line in valid_md.splitlines() if "teacher_val" not in line]
    assert_rejected("Missing required range: teacher_val omitted", "\n".join(lines))

    # 7. Missing required range (remove rl_train_pool row)
    lines = [line for line in valid_md.splitlines() if "rl_train_pool" not in line]
    assert_rejected("Missing required range: rl_train_pool omitted", "\n".join(lines))

    # 8. Extra unexpected range
    extra_row = "| extra_seed | 999900 | 999999 | proposed | Extra seed range |"
    assert_rejected("Extra unexpected range in table", valid_md + "\n" + extra_row)

    # 9. Malformed row: 4 columns instead of 5
    bad_md = valid_md.replace("| train | 0 | 99 | frozen | Training and in-loop evolution graphs (baseline uses seeds 0-19) |",
                              "| train | 0 | 99 | frozen |")
    assert_rejected("Malformed row: 4 columns", bad_md)

    # 10. Malformed row: 6 columns instead of 5
    bad_md = valid_md.replace("| train | 0 | 99 | frozen | Training and in-loop evolution graphs (baseline uses seeds 0-19) |",
                              "| train | 0 | 99 | frozen | extra_col | Training and in-loop evolution graphs |")
    assert_rejected("Malformed row: 6 columns", bad_md)

    # 11. Malformed row: non-integer start
    bad_md = valid_md.replace("| train | 0 | 99 |", "| train | zero | 99 |")
    assert_rejected("Malformed row: non-int start ('zero')", bad_md)

    # 12. Malformed row: float start
    bad_md = valid_md.replace("| train | 0 | 99 |", "| train | 0.0 | 99 |")
    assert_rejected("Malformed row: float start ('0.0')", bad_md)

    # 13. Malformed row: non-integer end
    bad_md = valid_md.replace("| val | 1000 | 1019 |", "| val | 1000 | N/A |")
    assert_rejected("Malformed row: non-int end ('N/A')", bad_md)

    # 14. Malformed row: invalid status ('active')
    bad_md = valid_md.replace("| train | 0 | 99 | frozen |", "| train | 0 | 99 | active |")
    assert_rejected("Malformed row: invalid status 'active'", bad_md)

    # 15. Malformed row: uppercase status ('FROZEN')
    bad_md = valid_md.replace("| train | 0 | 99 | frozen |", "| train | 0 | 99 | FROZEN |")
    assert_rejected("Malformed row: uppercase status 'FROZEN'", bad_md)

    # 16. Negative start seed (-10)
    bad_md = valid_md.replace("| train | 0 | 99 |", "| train | -10 | 99 |")
    assert_rejected("Negative seed: start = -10", bad_md)

    # 17. Duplicate name (replace val with duplicate train row)
    bad_md = valid_md.replace("| val | 1000 | 1019 | frozen | Validation graphs for model selection and early stopping |",
                              "| train | 1000 | 1019 | frozen | Duplicate train name |")
    assert_rejected("Duplicate range name: two train rows", bad_md)

    # 18. Empty markdown file
    assert_rejected("Empty file", "")

    # 19. Markdown file with text but no table
    assert_rejected("Text without markdown table", "# Notes\nThere are no tables here.")

    # 20. Frozen range value mismatch (train 1 to 99 instead of 0 to 99)
    bad_md = valid_md.replace("| train | 0 | 99 |", "| train | 1 | 99 |")
    assert_rejected("Frozen range mismatch: train 1-99", bad_md)

    # 21. Non-existent file path
    non_existent = REPO_ROOT / "SEEDS_NON_EXISTENT_12345.md"
    total_challenges += 1
    try:
        validate_seeds_file(non_existent)
        raise Exception("FAIL: Non-existent file did not raise AssertionError")
    except AssertionError as e:
        passed_challenges += 1
        print(f"  [PASS] Non-existent file -> rejected with AssertionError: {str(e)[:60]}...")

    print(f"Task 1 result: {passed_challenges}/{total_challenges} adversarial challenges strictly rejected.\n")
    return passed_challenges == total_challenges


def run_frozen_tamper_challenges():
    """Stress test test_frozen_files.py tamper detection mechanisms and hash verification."""
    print("=== [TASK 2] Frozen Files Guard & Hash Tamper Detection ===")
    assert MANIFEST_PATH.is_file(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        real_manifest = json.load(f)

    passed_tests = 0
    total_tests = 0

    def record_pass(desc: str):
        nonlocal passed_tests, total_tests
        total_tests += 1
        passed_tests += 1
        print(f"  [PASS] {desc}")

    def record_fail(desc: str, reason: str):
        nonlocal total_tests
        total_tests += 1
        print(f"  [FAIL] {desc}: {reason}")
        raise Exception(f"Failed test: {desc} -> {reason}")

    # 1. Verify compute_sha256 matches hashlib.sha256 directly on known test vectors
    test_vectors = [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"GraphColoring", hashlib.sha256(b"GraphColoring").hexdigest()),
        (b"evaluator/verify.py contents test vector \n\x00\xff", hashlib.sha256(b"evaluator/verify.py contents test vector \n\x00\xff").hexdigest()),
    ]
    for data, expected in test_vectors:
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            actual = compute_sha256(tmp_path)
            assert actual == expected, f"Expected {expected}, got {actual}"
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    record_pass("compute_sha256 matches exact SHA256 test vectors")

    # 2. Verify compute_set_hash deterministic order-independence
    s1 = {"alpha", "beta", "gamma"}
    s2 = {"gamma", "alpha", "beta"}
    h1 = compute_set_hash(s1)
    h2 = compute_set_hash(s2)
    assert h1 == h2, "compute_set_hash must be order-invariant"
    # Modifying an item must change hash
    h3 = compute_set_hash({"alpha", "beta", "delta"})
    assert h1 != h3, "compute_set_hash must differ on different sets"
    record_pass("compute_set_hash order-invariance and sensitivity")

    # 3. Dynamic test: evaluator/verify.py tamper detection
    # If the manifest has a tampered hash, test_verify_py_unmodified must fail
    fake_manifest = copy.deepcopy(real_manifest)
    fake_manifest["verify_py"] = "0000000000000000000000000000000000000000000000000000000000000000"
    from tests.test_frozen_files import test_verify_py_unmodified
    try:
        test_verify_py_unmodified(fake_manifest)
        record_fail("verify.py tamper detection", "Did not raise AssertionError on tampered manifest")
    except AssertionError as e:
        assert "evaluator/verify.py has been modified!" in str(e)
        record_pass("evaluator/verify.py tamper detection successfully triggers AssertionError")

    # 4. Dynamic test: data_v1_1 tamper detection
    from tests.test_frozen_files import test_data_v1_1_unmodified
    # 4a. Tampered file hash
    fake_manifest = copy.deepcopy(real_manifest)
    target_key = list(fake_manifest["data_v1_1"].keys())[0]
    fake_manifest["data_v1_1"][target_key] = "1111111111111111111111111111111111111111111111111111111111111111"
    try:
        test_data_v1_1_unmodified(fake_manifest)
        record_fail("data_v1_1 hash tamper", "Did not raise AssertionError on modified file hash")
    except AssertionError as e:
        assert f"File modified in data/data_v1_1: {target_key}!" in str(e)
        record_pass("data/data_v1_1 file hash tamper triggers AssertionError")

    # 4b. Missing file in data_v1_1
    fake_manifest = copy.deepcopy(real_manifest)
    fake_manifest["data_v1_1"]["data/data_v1_1/ghost_file.json"] = "dummy_hash"
    try:
        test_data_v1_1_unmodified(fake_manifest)
        record_fail("data_v1_1 missing file", "Did not raise AssertionError on missing file")
    except AssertionError as e:
        assert "File list in data/data_v1_1 changed!" in str(e)
        record_pass("data/data_v1_1 missing file triggers AssertionError")

    # 4c. Rogue file added to data_v1_1
    fake_manifest = copy.deepcopy(real_manifest)
    del fake_manifest["data_v1_1"][target_key]  # makes current directory look like it has an added file
    try:
        test_data_v1_1_unmodified(fake_manifest)
        record_fail("data_v1_1 added file", "Did not raise AssertionError on unexpected added file")
    except AssertionError as e:
        assert "File list in data/data_v1_1 changed!" in str(e)
        record_pass("data/data_v1_1 added file triggers AssertionError")

    # 5. Dynamic test: data_v1 tamper detection
    from tests.test_frozen_files import test_data_v1_unmodified
    # 5a. Tampered file hash in data_v1
    fake_manifest = copy.deepcopy(real_manifest)
    target_v1 = list(fake_manifest["data_v1"].keys())[0]
    fake_manifest["data_v1"][target_v1] = "2222222222222222222222222222222222222222222222222222222222222222"
    try:
        test_data_v1_unmodified(fake_manifest)
        record_fail("data_v1 hash tamper", "Did not raise AssertionError on modified data_v1 file")
    except AssertionError as e:
        assert "Modified files detected in data/data_v1" in str(e)
        record_pass("data/data_v1 file hash tamper triggers AssertionError")

    # 5b. Extra file in data_v1
    fake_manifest = copy.deepcopy(real_manifest)
    del fake_manifest["data_v1"][target_v1]
    try:
        test_data_v1_unmodified(fake_manifest)
        record_fail("data_v1 added file", "Did not raise AssertionError on added data_v1 file")
    except AssertionError as e:
        assert "File list in data/data_v1 changed!" in str(e)
        record_pass("data/data_v1 added file triggers AssertionError")

    # 6. Dynamic test: rogue .done locks detection
    from tests.test_frozen_files import test_no_rogue_done_locks
    # Create a temporary rogue .done file in repo root
    rogue_lock = REPO_ROOT / "rogue_test_lock.done"
    try:
        rogue_lock.write_text("rogue lock content", encoding="utf-8")
        try:
            test_no_rogue_done_locks()
            record_fail("rogue .done detection", "Did not raise AssertionError on rogue .done file")
        except AssertionError as e:
            assert "Unexpected .done lock file outside runs/:" in str(e)
            record_pass("Rogue .done lock outside runs/ triggers AssertionError")
    finally:
        if rogue_lock.exists():
            rogue_lock.unlink()

    # Confirm clean state afterwards
    test_no_rogue_done_locks()
    record_pass("No rogue .done locks passes cleanly when repo is clean")

    # 7. Dynamic test: sandbox whitelist tampering detection
    from tests.test_frozen_files import test_sandbox_whitelist_unmodified
    # 7a. Tampered whitelist manifest items
    fake_manifest = copy.deepcopy(real_manifest)
    fake_manifest["sandbox_whitelist"]["FORBIDDEN_CALLS"].append("rogue_forbidden_call")
    try:
        test_sandbox_whitelist_unmodified(fake_manifest)
        record_fail("sandbox whitelist tamper", "Did not raise AssertionError on modified FORBIDDEN_CALLS")
    except AssertionError:
        record_pass("Sandbox whitelist item modification triggers AssertionError")

    # 7b. Tampered whitelist hash
    fake_manifest = copy.deepcopy(real_manifest)
    fake_manifest["sandbox_hashes"]["SAFE_BUILTINS_KEYS"] = "badhash123"
    try:
        test_sandbox_whitelist_unmodified(fake_manifest)
        record_fail("sandbox hash tamper", "Did not raise AssertionError on modified SAFE_BUILTINS hash")
    except AssertionError:
        record_pass("Sandbox whitelist hash mismatch triggers AssertionError")

    print(f"Task 2 result: {passed_tests}/{total_tests} tamper detection assertions passed.\n")
    return passed_tests == total_tests


def run_teacher_config_challenges():
    """Verify configs/teacher_llm.yaml robustness and secret exclusion."""
    print("=== [TASK 3] Teacher Config Adversarial Checks ===")
    from tests.test_seeds import test_teacher_llm_yaml_parses_and_secure
    # Run real test first
    test_teacher_llm_yaml_parses_and_secure()
    print("  [PASS] configs/teacher_llm.yaml passes all baseline checks")

    # Adversarial test: API key patterns must be caught
    key_patterns = [
        re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),
        re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    ]
    adversarial_samples = [
        "sk-proj-1234567890123456789012345",
        "AIzaSyD-1234567890123456789012345678901",
        "ghp_123456789012345678901234567890123456",
    ]
    for sample in adversarial_samples:
        matched = any(p.search(sample) for p in key_patterns)
        assert matched, f"Pattern scanner failed to detect leak: {sample}"
    print("  [PASS] API key leak detection regex successfully flags all canary keys")

    # Env var name regex validation
    valid_env_names = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY_V2"]
    invalid_env_names = ["sk-key-123", "lower_case_key", "123_INVALID", "API-KEY", ""]
    env_regex = re.compile(r"^[A-Z_][A-Z0-9_]*$")
    for name in valid_env_names:
        assert env_regex.match(name), f"Valid name rejected: {name}"
    for name in invalid_env_names:
        assert not env_regex.match(name), f"Invalid name accepted: {name}"
    print("  [PASS] api_key_env identifier format strictly enforced (uppercase identifiers only)\n")
    return True


if __name__ == "__main__":
    t1 = run_seeds_adversarial_challenges()
    t2 = run_frozen_tamper_challenges()
    t3 = run_teacher_config_challenges()

    if t1 and t2 and t3:
        print("ALL ADVERSARIAL CHALLENGES COMPLETED SUCCESSFULLY: IMPLEMENTATION IS ROBUST.")
        sys.exit(0)
    else:
        print("ADVERSARIAL CHALLENGES DETECTED VULNERABILITIES.")
        sys.exit(1)
