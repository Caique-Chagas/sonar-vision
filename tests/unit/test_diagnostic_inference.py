from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import Box, Detection, OCRReading
from sonar_vision.detection.yolo import ModelUnavailable, YoloSignDetector
from sonar_vision.processing.pipeline import Context, FrameProcessor
from sonar_vision.processing.roi import extract_roi
from sonar_vision.validation.normalizer import LineNormalizer
from sonar_vision.validation.routes import RouteMatcher


class CapturedModel:
    names = {0: "destination_sign"}

    def __init__(self):
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return []


def test_default_roi_and_operating_gates_remain_unchanged():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    region = Box(x1=20, y1=10, x2=100, y2=50)
    before = cv2.resize(image[10:50, 20:100], None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    assert np.array_equal(extract_roi(image, region), before)
    settings = Settings()
    assert settings.roi_padding == 0 and settings.min_detection == 0.35
    assert settings.min_votes == 3 and settings.min_ocr == 0.7


def test_padding_clips_at_frame_edges_and_includes_context():
    image = np.zeros((100, 200, 3), np.uint8)
    image[0, 0] = [10, 20, 30]
    region = Box(x1=1, y1=1, x2=199, y2=99)
    result = extract_roi(image, region, padding=0.1)
    assert result.shape == image.shape
    assert np.array_equal(result[0, 0], image[0, 0])


@pytest.mark.parametrize("padding", [-0.1, 0.26, float("nan"), float("inf")])
def test_invalid_margins_rejected(padding):
    with pytest.raises(ValueError):
        extract_roi(
            np.zeros((100, 200, 3), np.uint8), Box(x1=1, y1=1, x2=50, y2=50), padding=padding
        )


def test_padding_does_not_turn_outside_box_into_evidence():
    with pytest.raises(ValueError, match="fora"):
        extract_roi(
            np.zeros((100, 200, 3), np.uint8), Box(x1=201, y1=1, x2=250, y2=50), padding=0.25
        )


def test_real_pipeline_passes_margin_and_keeps_single_frame_uncertain():
    class Detector:
        model_version = "software-fixture"

        def detect(self, frame):
            return [
                Detection(
                    box=Box(x1=50, y1=20, x2=150, y2=80), confidence=0.54, label="destination_sign"
                )
            ]

    class OCR:
        shape = None

        def read(self, roi):
            self.shape = roi.shape
            return [OCRReading(text="6913-10", confidence=0.99)]

    ocr = OCR()
    processor = FrameProcessor(
        Detector(),
        ocr,
        Context("6291-10", "test", "fixture", "simulation"),
        Settings(roi_padding=0.1),
    )
    _, events = processor.process(np.zeros((100, 200, 3), np.uint8), 0, 0)
    assert ocr.shape == (72, 120, 3)
    assert len(events) == 1 and events[0].status == "UNCERTAIN"
    assert events[0].route_detected is None and events[0].simulated
    assert events[0].supporting_frames == 1


@pytest.mark.parametrize("size", [319, 641, 1952, True])
def test_invalid_inference_sizes_rejected(size):
    with pytest.raises(ValueError):
        YoloSignDetector(Path("fixture"), model=CapturedModel(), imgsz=size)


def test_resolution_is_explicit_and_legacy_default_sends_no_tile_keyword():
    model = CapturedModel()
    YoloSignDetector(Path("fixture"), model=model, imgsz=1280).detect(
        np.zeros((80, 160, 3), np.uint8)
    )
    assert model.kwargs["imgsz"] == 1280 and "tile" not in model.kwargs
    assert model.kwargs["conf"] == 0.35 and model.kwargs["max_det"] == 16


def test_tiles_opt_in_require_supported_version(monkeypatch):
    monkeypatch.setattr("sonar_vision.detection.yolo._supports_native_tiles", lambda: False)
    with pytest.raises(ModelUnavailable, match="não suporta"):
        YoloSignDetector(Path("fixture"), model=CapturedModel(), tile_size=640)
    assert YoloSignDetector(Path("fixture"), model=CapturedModel()).tile_size == 0


def test_native_tiles_and_budget(monkeypatch):
    monkeypatch.setattr("sonar_vision.detection.yolo._supports_native_tiles", lambda: True)
    monkeypatch.setattr("sonar_vision.detection.yolo._native_tile_count", lambda frame, size: 12)
    model = CapturedModel()
    detector = YoloSignDetector(Path("fixture"), model=model, tile_size=640)
    frame = np.zeros((80, 160, 3), np.uint8)
    detector.detect(frame)
    assert model.kwargs["tile"] == 640 and model.kwargs["batch"] == 1
    model.kwargs = None
    monkeypatch.setattr("sonar_vision.detection.yolo._native_tile_count", lambda frame, size: 65)
    with pytest.raises(ValueError, match="64 tiles"):
        detector.detect(frame)
    assert model.kwargs is None


def test_sticky_tiling_does_not_leak_into_default_detector():
    model = CapturedModel()
    model.predictor = SimpleNamespace(args=SimpleNamespace(tile=640))
    YoloSignDetector(Path("fixture"), model=model).detect(np.zeros((80, 160, 3), np.uint8))
    assert model.kwargs["tile"] is False


def test_wrong_ocr_suffix_is_not_invented_or_corrected_by_normalizer():
    normalized = LineNormalizer({"6913-10"}).normalize("6913-11")
    assert normalized.route == "6913-11"
    # A supplied fixture catalog can reject it; this is not a live SPTrans catalog.
    assert RouteMatcher({"6913-10"}).match(normalized.route, "6913-10").status == "UNCERTAIN"
