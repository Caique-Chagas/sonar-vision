import copy
import json

import pytest

from sonar_vision.core.contracts import Box
from sonar_vision.evaluation.prospective import (
    match_panels,
    summarize_paired,
    verify_prospective_inputs,
)
from sonar_vision.training.dataset import sha256


def pair(identifier="one", truth="7005-51", correct=False):
    return [
        {
            "id": identifier,
            "profile": p,
            "expected_code": truth,
            "code_exact": correct,
            "eligible_code_exact": correct,
            "confirmed_events": 0,
            "panel_metrics": {"tp": 0, "fn": 1, "fp": 0, "ignored_unassessed": 0},
        }
        for p in ["A", "D"]
    ]


def test_missing_truth_not_counted_as_wrong_or_correct_code():
    rows = pair() + pair("unknown", None, False)
    summary = summarize_paired(rows)
    assert summary["A"]["images"] == 2 and summary["A"]["coded_images"] == 1
    assert summary["A"]["complete_code_accuracy"] == 0
    unknown = summarize_paired(pair(truth=None))
    assert unknown["D"]["complete_code_accuracy"] is None


def test_profile_change_does_not_change_denominator():
    rows = pair()
    rows[1]["code_exact"] = True
    rows[1]["eligible_code_exact"] = True
    summary = summarize_paired(rows)
    assert summary["D"]["exact_codes"] == 1 and summary["A"]["exact_codes"] == 0
    assert summary["D"]["coded_images"] == summary["A"]["coded_images"] == 1


@pytest.mark.parametrize("change", ["unpaired", "duplicate", "confirmation"])
def test_reject_invalid_paired_or_static_evidence(change):
    rows = pair()
    if change == "unpaired":
        rows = rows[:1]
    elif change == "duplicate":
        rows += pair()
    else:
        rows[0]["confirmed_events"] = 1
    with pytest.raises(ValueError):
        summarize_paired(rows)


def test_one_reference_cannot_match_two_predictions():
    b = Box(x1=10, y1=10, x2=110, y2=40)
    predictions = [{"box": b.model_dump(), "confidence": s} for s in [0.9, 0.8]]
    assert match_panels(predictions, [b], []) == {
        "tp": 1,
        "fn": 0,
        "fp": 1,
        "ignored_unassessed": 0,
    }


def test_unannotated_background_is_not_claimed_false_positive():
    b = Box(x1=10, y1=10, x2=110, y2=40)
    result = match_panels([{"box": b.model_dump(), "confidence": 0.9}], [], [b])
    assert result["fp"] == 0 and result["ignored_unassessed"] == 1


def test_tight_text_is_not_full_panel_even_if_it_contains_code():
    panel = Box(x1=0, y1=0, x2=200, y2=80)
    text = Box(x1=50, y1=20, x2=150, y2=50)
    result = match_panels([{"box": text.model_dump(), "confidence": 0.9}], [panel], [])
    assert result["tp"] == 0 and result["fn"] == 1


def fixture_files(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"bytes_for_hash_only_not_CV")
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fixture")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "source_hashes": {},
                "profiles": {
                    "A": {"weights": "model.pt", "sha256": sha256(weights)},
                    "D": {"weights": "model.pt", "sha256": sha256(weights)},
                },
            }
        )
    )
    record = {
        "id": "new",
        "image": "image.jpg",
        "image_sha256": sha256(image),
        "rights_reviewed": True,
        "split": "prospective_test_only",
        "expected_code": "7005-51",
        "ground_truth_method": "fixture",
    }
    review = {
        "protocol_sha256": sha256(protocol),
        "reviewed_before_model_outputs": True,
        "records": [record],
    }
    ref = tmp_path / "review.json"
    ref.write_text(json.dumps(review))
    return protocol, ref, review


@pytest.mark.parametrize(
    "change", ["rights", "protocol", "image", "truth", "training", "duplicate", "weights"]
)
def test_input_gate_rejects_unverified_or_changed_material(tmp_path, change):
    protocol, ref, review = fixture_files(tmp_path)
    if change == "rights":
        review["records"][0]["rights_reviewed"] = False
    elif change == "protocol":
        review["protocol_sha256"] = "wrong"
    elif change == "image":
        (tmp_path / "image.jpg").write_bytes(b"changed")
    elif change == "truth":
        review["records"][0]["ground_truth_method"] = ""
    elif change == "training":
        review["records"][0]["split"] = "train"
    elif change == "duplicate":
        review["records"].append(copy.deepcopy(review["records"][0]))
    else:
        (tmp_path / "model.pt").write_bytes(b"changed")
    ref.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        verify_prospective_inputs(protocol, ref, tmp_path)


def test_frozen_inputs_accept_verified_reference(tmp_path):
    protocol, ref, _ = fixture_files(tmp_path)
    assert len(verify_prospective_inputs(protocol, ref, tmp_path)["records"]) == 1
