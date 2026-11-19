from dataclasses import replace

import cv2
import numpy as np
import pytest

from sonar_vision.training.dataset import sha256
from sonar_vision.training.recognition_dataset import (
    RecognitionSample,
    export_recognition,
    verify_recognition,
)


def sample(tmp_path, index=0, split="train", text="6030-10"):
    path = tmp_path / f"fixture{index}.png"
    cv2.imwrite(str(path), np.random.default_rng(index).integers(0, 256, (32, 96, 3), np.uint8))
    return RecognitionSample(
        path,
        text,
        split,
        f"parent{index}",
        f"capture{index}",
        f"author{index}",
        "https://example.invalid/software-fixture",
        "fixture-not-real-data",
        sha256(path),
        True,
        True,
        "known_development",
    )


def test_paddle_format_empty_test_and_preservation(tmp_path):
    out = tmp_path / "dataset"
    rows = [sample(tmp_path), sample(tmp_path, 1, "val", "F04")]
    receipt = export_recognition(rows, out)
    assert receipt["counts"] == {"train": 1, "val": 1, "test": 0}
    assert receipt["independent_exact_match"] is None
    assert not receipt["training_authorized"]
    assert not (out / "rec_gt_test.txt").read_text()
    assert (out / "rec_gt_train.txt").read_text() == "images/0000.png\t6030-10\n"
    assert verify_recognition(out) == receipt
    with pytest.raises(ValueError, match="Preserve"):
        export_recognition(rows, out)


@pytest.mark.parametrize("field", ["rights_verified", "manual_verified"])
def test_unapproved_photos_or_OCR_generated_labels_rejected(tmp_path, field):
    record = replace(sample(tmp_path), **{field: False})
    with pytest.raises(ValueError, match="Rights"):
        export_recognition([record], tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


@pytest.mark.parametrize("field,value", [("parent_id", "35046766"), ("scope", "historical_test")])
def test_reserved_and_historical_sources_cannot_be_repurposed(tmp_path, field, value):
    record = replace(sample(tmp_path), **{field: value})
    with pytest.raises(ValueError, match="Reserved/historical"):
        export_recognition([record], tmp_path / "blocked")


@pytest.mark.parametrize("field", ["parent_id", "group", "author", "vehicle"])
def test_capture_author_vehicle_leakage_is_rejected(tmp_path, field):
    first = replace(sample(tmp_path), vehicle="vehicle-a")
    second = replace(sample(tmp_path, 1, "val", "050"), **{field: getattr(first, field)})
    with pytest.raises(ValueError, match="crosses splits"):
        export_recognition([first, second], tmp_path / "blocked")


def test_known_development_and_seen_codes_do_not_become_unseen_test(tmp_path):
    first = sample(tmp_path)
    second = sample(tmp_path, 1, "test", "F04")
    with pytest.raises(ValueError, match="Known development"):
        export_recognition([first, second], tmp_path / "blocked")
    second = replace(second, transcript=first.transcript, scope="new_unseen")
    with pytest.raises(ValueError, match="unseen test codes"):
        export_recognition([first, second], tmp_path / "blocked")


def test_modified_pixels_and_labels_detected(tmp_path):
    out = tmp_path / "dataset"
    export_recognition([sample(tmp_path)], out)
    (out / "rec_gt_train.txt").write_text("images/0000.png\t9999-99\n")
    with pytest.raises(ValueError, match="labels changed"):
        verify_recognition(out)


def test_reencoded_same_pixels_cannot_cross_splits(tmp_path):
    first = sample(tmp_path)
    second = sample(tmp_path, 1, "val", "F04")
    cv2.imwrite(str(second.image), cv2.imread(str(first.image)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    second = replace(second, image_sha256=sha256(second.image))
    assert second.image_sha256 != first.image_sha256
    with pytest.raises(ValueError, match="Near-duplicate"):
        export_recognition([first, second], tmp_path / "blocked")


@pytest.mark.parametrize("text", ["6030-10\n9999-99", "6030\t10", "DESTINO"])
def test_transcription_must_be_explicit_code(tmp_path, text):
    with pytest.raises(ValueError, match="Exact code transcript"):
        export_recognition([sample(tmp_path, text=text)], tmp_path / "blocked")
