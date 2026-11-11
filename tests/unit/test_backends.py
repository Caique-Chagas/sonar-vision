from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sonar_vision.detection.yolo import ModelUnavailable, YoloSignDetector
from sonar_vision.ocr.paddle import OCRFailure, PaddleEngine
from sonar_vision.processing.roi import extract_roi, validate_frame


class Array:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class Model:
    names = {0: "destination_sign", 1: "bus"}

    def predict(self, **kwargs):
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=Array([[0, 0, 100, 60], [0, 0, 100, 60], [0, 0, 0, 0], [0, 0, 50, 50]]),
                    conf=Array([0.9, 0.2, 0.9, 0.95]),
                    cls=Array([0, 0, 0, 1]),
                )
            )
        ]


def test_detector_format_filters_and_load(frame):
    detector = YoloSignDetector(Path("not-there.pt"), model=Model())
    results = detector.detect(frame)
    assert len(results) == 1
    assert results[0].box.x2 == 100 and results[0].confidence == 0.9


def test_model_missing_and_generic_rejected(tmp_path):
    with pytest.raises(ModelUnavailable, match="ausente"):
        YoloSignDetector(tmp_path / "absent.pt")
    with pytest.raises(ModelUnavailable, match="classe"):
        YoloSignDetector(tmp_path / "absent.pt", model=SimpleNamespace(names={0: "bus"}))


def test_empty_detector(frame):
    class EmptyModel(Model):
        def predict(self, **kwargs):
            return [SimpleNamespace(boxes=None)]

    assert YoloSignDetector(Path("absent.pt"), model=EmptyModel()).detect(frame) == []


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        np.zeros((10, 10), np.uint8),
        np.zeros((10, 10, 3), np.float32),
        np.zeros((0, 10, 3), np.uint8),
    ],
)
def test_invalid_frames(invalid):
    with pytest.raises(ValueError):
        validate_frame(invalid)


def test_roi_clips_and_transforms(frame, sign):
    assert extract_roi(frame, sign.box).size > 0
    assert extract_roi(frame, sign.box, "clahe").ndim == 3
    with pytest.raises(ValueError):
        extract_roi(frame, sign.box, "unknown")


@pytest.mark.parametrize("texts,scores", [(["971D-10"], [0.88]), ([], []), (["971D"], [0.52])])
def test_paddle_v3_adapter_preserves_text_and_actual_score(frame, texts, scores):
    engine = SimpleNamespace(predict=lambda image: [{"rec_texts": texts, "rec_scores": scores}])
    readings = PaddleEngine(engine).read(frame)
    assert [r.text for r in readings] == texts
    assert [r.confidence for r in readings] == scores


def test_ocr_failure_never_returns_fabricated_reading(frame):
    def failed(image):
        raise RuntimeError("backend failure")

    with pytest.raises(OCRFailure):
        PaddleEngine(SimpleNamespace(predict=failed)).read(frame)
