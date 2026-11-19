import numpy as np
import pytest

from sonar_br.final_correction.run import eligible, evaluate
from sonar_vision.core.contracts import Box, Detection, OCRReading


class Detector:
    model_version = "fixture-not-real-CV"

    def detect(self, frame):
        return [
            Detection(
                box=Box(x1=40, y1=40, x2=160, y2=80), confidence=0.9, label="destination_sign"
            )
        ]


class OCR:
    def __init__(self):
        self.shapes = []
        self.corners = []

    def read(self, roi):
        self.shapes.append(roi.shape)
        self.corners.append(int(roi[0, 0, 0]))
        return [OCRReading(text="6030-10", confidence=0.95)]


def reference():
    return {
        "id": "fixture",
        "split": "development_only_never_new_test",
        "expected_code": "6030-10",
        "boxes": [{"xyxy": [40, 40, 160, 80]}],
    }


@pytest.mark.parametrize("margin", [0, 0.1])
def test_static_runner_retains_safety_and_detection_geometry(margin):
    ocr = OCR()
    image = np.zeros((200, 240, 3), np.uint8)
    image[:, :, 0] = np.arange(240, dtype=np.uint8)
    result = evaluate(image, reference(), Detector(), ocr, margin=margin)
    assert result["eligible_exact"]
    assert result["final_status"] == "UNCERTAIN"
    assert all(e["status"] == "UNCERTAIN" for e in result["events"])
    assert result["predictions"][0]["reference_ious"] == [1]
    assert ocr.corners == [28 if margin else 40]


@pytest.mark.parametrize("field,value", [("id", "35046766"), ("split", "prospective_test")])
def test_reserved_and_test_images_are_rejected_before_OCR(field, value):
    ref = reference()
    ref[field] = value
    ocr = OCR()
    with pytest.raises(ValueError, match="Reserved/test"):
        evaluate(np.zeros((200, 240, 3), np.uint8), ref, Detector(), ocr, margin=0)
    assert not ocr.shapes


def test_high_OCR_without_geometric_or_detection_evidence_is_not_eligible():
    prediction = {
        "confidence": 0.9,
        "observation": {"ocr_confidence": 0.99},
        "reference_ious": [],
    }
    assert not eligible(prediction)
    prediction["reference_ious"] = [0.49]
    assert not eligible(prediction)
    prediction["reference_ious"] = [0.8]
    prediction["confidence"] = 0.34
    assert not eligible(prediction)


def test_incorrect_recognized_suffix_remains_in_evidence():
    ref = reference()
    ref["expected_code"] = "6030-11"
    row = evaluate(np.zeros((200, 240, 3), np.uint8), ref, Detector(), OCR(), margin=0)
    assert not row["eligible_exact"]
    assert row["wrong_eligible_codes"] == ["6030-10"]
