import csv
import json
import pytest
from pathlib import Path
from scripts.make_cards import (
    extract_numbers,
    verify_numbers_preserved,
    verify_no_label_leak,
    generate_rule_based_paraphrases,
    FORBIDDEN_LEAK_WORDS,
)

ROOT = Path(__file__).resolve().parent.parent
CARDS_JSON = ROOT / "data" / "data_v1_2" / "cards.json"
SPOTCHECK_CSV = ROOT / "data" / "data_v1_2" / "cards_spotcheck_50.csv"


def test_cards_json_exists_and_valid():
    assert CARDS_JSON.exists(), f"cards.json not found at {CARDS_JSON}"
    with open(CARDS_JSON, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    
    assert len(catalog) == 106, f"Expected 106 families, got {len(catalog)}"
    
    for fam_id, entry in catalog.items():
        assert "card" in entry
        assert "card_detailed" in entry
        assert "paraphrases" in entry
        assert len(entry["paraphrases"]) >= 3
        
        # Test numeric preservation
        base_card = entry["card"]
        for p in entry["paraphrases"]:
            ok, missing = verify_numbers_preserved(base_card, p)
            assert ok, f"Family {fam_id} paraphrase missed numbers {missing}: {p}"
            
        # Test no label leakage
        for text in [entry["card"], entry["card_detailed"]] + entry["paraphrases"]:
            ok, leaks = verify_no_label_leak(text, fam_id)
            assert ok, f"Family {fam_id} leaked {leaks} in text: {text}"


def test_spotcheck_csv_structure():
    assert SPOTCHECK_CSV.exists()
    with open(SPOTCHECK_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 50, f"Expected 50 spot-check rows, got {len(rows)}"
    expected_cols = {"family_id", "structure_type", "original_card", "paraphrase_1", "paraphrase_2", "paraphrase_3"}
    assert set(reader.fieldnames or []).issuperset(expected_cols)
    for row in rows:
        orig = row["original_card"]
        fam_id = row["family_id"]
        for p_key in ["paraphrase_1", "paraphrase_2", "paraphrase_3"]:
            p = row[p_key]
            assert p, f"Empty {p_key} in row {fam_id}"
            ok, missing = verify_numbers_preserved(orig, p)
            assert ok, f"Missing numbers {missing} in {p_key} for {fam_id}"
            ok, leaks = verify_no_label_leak(p, fam_id)
            assert ok, f"Leaks {leaks} in {p_key} for {fam_id}"


def test_paraphrase_generator_unit():
    card = "random geometric, nodes 150-250, density 0.15, high degree variance, high clustering"
    paras = generate_rule_based_paraphrases(card, "random_geometric")
    assert len(paras) == 3
    for p in paras:
        ok, missing = verify_numbers_preserved(card, p)
        assert ok, f"Missing numbers {missing} in {p}"
        ok, leaks = verify_no_label_leak(p)
        assert ok
