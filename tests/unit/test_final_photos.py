import json
from copy import deepcopy

import pytest

from sonar_vision.evaluation.final_photos import read_case, validate_final_reference, write_case
from sonar_vision.training.dataset import sha256


@pytest.fixture
def frozen_inputs(tmp_path):
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"fixture")
    source = tmp_path / "source.py"
    source.write_text("# fixture")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "weights": {"model": {"path": weight.name, "sha256": sha256(weight)}},
                "source_hashes": {source.name: sha256(source)},
            }
        )
    )
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture, no real model metrics")
    reference = tmp_path / "reference.json"
    record = {
        "id": "fixture",
        "group_id": "one-day",
        "image": image.name,
        "image_sha256": sha256(image),
        "width": 100,
        "height": 50,
        "author": "fixture",
        "license": "CC0",
        "source": "https://example.org/test",
        "rights_verified": True,
        "rights_evidence": "fixture only",
        "difficulty": "easy",
        "expected_code": "1234-10",
        "code_location": "main_display",
        "ground_truth_method": "fixture truth",
        "boxes": [{"xyxy": [10, 5, 90, 25], "approved": True, "boundary_evidence": "fixture"}],
    }
    data = {
        "protocol_sha256": sha256(protocol),
        "reviewed_before_model_outputs": True,
        "records": [record],
    }
    reference.write_text(json.dumps(data))
    return reference, protocol, tmp_path, data


def test_valid_frozen_reference(frozen_inputs):
    ref, protocol, root, _ = frozen_inputs
    assert len(validate_final_reference(ref, protocol, root, set())["records"]) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("rights_verified", False),
        ("difficulty", None),
        ("ground_truth_method", None),
        ("image_sha256", "changed"),
        ("author", ""),
        ("boxes", [{"xyxy": [-1, 5, 90, 25], "approved": True, "boundary_evidence": "fixture"}]),
    ],
)
def test_invalid_reference_refused(frozen_inputs, field, value):
    ref, protocol, root, data = frozen_inputs
    data["records"][0][field] = value
    ref.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_final_reference(ref, protocol, root, set())


def test_duplicate_capture_group_refused(frozen_inputs):
    ref, protocol, root, data = frozen_inputs
    data["records"].append(deepcopy(data["records"][0]))
    ref.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="group reused"):
        validate_final_reference(ref, protocol, root, set())


def test_previously_used_day_refused(frozen_inputs):
    ref, protocol, root, _ = frozen_inputs
    with pytest.raises(ValueError, match="group reused"):
        validate_final_reference(ref, protocol, root, {"one-day"})


def test_weight_changed_refused(frozen_inputs):
    ref, protocol, root, _ = frozen_inputs
    (root / "weight.pt").write_bytes(b"other")
    with pytest.raises(ValueError, match="model changed"):
        validate_final_reference(ref, protocol, root, set())


def test_empty_negative_requires_confirmation(frozen_inputs):
    ref, protocol, root, data = frozen_inputs
    record = data["records"][0]
    record.update(boxes=[], expected_code=None, difficulty="negative")
    ref.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="verified negative"):
        validate_final_reference(ref, protocol, root, set())


def test_case_binding_and_no_overwrite(tmp_path):
    path = tmp_path / "case.json"
    assert read_case(path, "frozen") is None
    write_case(path, {"test": True}, "frozen")
    assert read_case(path, "frozen")["test"]
    with pytest.raises(ValueError, match="different frozen"):
        read_case(path, "changed")
    with pytest.raises(ValueError, match="overwritten"):
        write_case(path, {}, "frozen")


def test_body_marking_cannot_be_a_display_code(frozen_inputs):
    ref, protocol, root, data = frozen_inputs
    data["records"][0]["code_location"] = "body_line_marking"
    ref.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Body or fleet"):
        validate_final_reference(ref, protocol, root, set())
