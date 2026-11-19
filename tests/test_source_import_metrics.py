import pytest

from sonar_br.source_import.compare import select_code, summarize


def reading(text, confidence=0.99):
    return {"text": text, "confidence": confidence}


def test_complete_reading_is_not_replaced_by_high_scoring_partial_tokens():
    assert select_code([reading("6030-10", 0.85), reading("6030"), reading("10")]) == "6030-10"


def test_conflicting_complete_readings_abstain():
    assert select_code([reading("6030-10"), reading("6030-11")]) is None


def test_no_missing_suffix_or_O_zero_repair():
    assert select_code([reading("6030")]) == "6030"
    assert select_code([reading("O30")]) == "O30"
    assert select_code([reading("6030-10", 0.69)]) is None


def test_missing_ROI_remains_in_the_denominator_and_is_not_independent_test():
    rows = [
        {"id": "a", "truth": "6030-10", "prediction": "6030-10", "exact": True, "ms": 1},
        {"id": "b", "truth": "971D-10", "prediction": None, "exact": False, "ms": 0},
    ]
    report = summarize(rows, {"a", "b"})
    assert report["exact_match_DEV"] == 0.5
    assert report["abstentions"] == 1
    assert report["independent_test_exact_match"] is None
    with pytest.raises(ValueError, match="All frozen examples"):
        summarize(rows[:1], {"a", "b"})
