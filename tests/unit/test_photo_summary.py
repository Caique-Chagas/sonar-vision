from copy import deepcopy
from typing import Any

import pytest

from sonar_vision.evaluation.photo_summary import summarize_photos


def sample() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = {
        "id": "photo",
        "expected_code": "201",
        "code_location": "main_display",
        "city": "City",
        "confirmed_negative": False,
        "difficulty": "hard",
        "ap_eligible": True,
        "source": "licensed-source",
    }
    prediction = {
        "frame_code": "201",
        "confidence": 0.8,
        "observation": {"ocr_confidence": 0.9},
        "reference_ious": [0.4],
        "box": {},
        "readings": [{"text": "201", "confidence": 0.9}],
    }
    case = {
        "id": "photo",
        "expected_code": "201",
        "predictions": [prediction],
        "reference_boxes": [],
        "final_code": None,
        "events": [],
        "detection": {"tp": 0, "fp": 1, "fn": 1, "ignored_unassessed": 0},
        "pipeline_wall_ms": 100,
        "detector_wall_ms": 30,
        "ocr_wall_ms": 70,
        "cold_first_detector_call": False,
    }
    return [dict(deepcopy(case), model=m) for m in ["Bangkok", "BR"]], [reference]


def test_raw_candidate_joint_and_confirmation_are_distinct() -> None:
    cases, refs = sample()
    result = summarize_photos(cases, refs)["models"]["BR"]
    assert result["raw_exact"] == result["eligible_exact"] == 1
    assert result["joint_panel_code_exact"] == result["confirmed_events"] == 0


def test_confidence_rejection_keeps_raw_OCR_success() -> None:
    cases, refs = sample()
    cases[1]["predictions"][0]["observation"]["ocr_confidence"] = 0.69
    result = summarize_photos(cases, refs)["models"]["BR"]
    assert result["raw_exact"] == 1
    assert result["eligible_exact"] == 0


def test_missing_truth_does_not_expand_code_denominator() -> None:
    cases, refs = sample()
    refs[0]["expected_code"] = None
    for row in cases:
        row["expected_code"] = None
    result = summarize_photos(cases, refs)["models"]["BR"]
    assert result["images"] == 1
    assert result["coded_images"] == result["raw_exact"] == 0


def test_unpaired_or_duplicate_photos_refused() -> None:
    cases, refs = sample()
    with pytest.raises(ValueError, match="same unique"):
        summarize_photos(cases + [deepcopy(cases[1])], refs)
    with pytest.raises(ValueError, match="same unique"):
        summarize_photos(cases[:1], refs)


def test_posthoc_truth_change_and_confirmation_refused() -> None:
    cases, refs = sample()
    cases[0]["expected_code"] = "202"
    with pytest.raises(ValueError, match="reference"):
        summarize_photos(cases, refs)
    cases[0]["expected_code"] = "201"
    cases[0]["events"] = [{"status": "CONFIRMED"}]
    with pytest.raises(ValueError, match="temporal"):
        summarize_photos(cases, refs)
