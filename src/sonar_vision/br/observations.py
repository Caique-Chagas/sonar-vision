"""Instrumentation adapters that preserve the production detector/OCR contracts."""

import re
import time
import unicodedata
from dataclasses import asdict
from typing import Any

from sonar_vision.core.contracts import Detection, OCRReading
from sonar_vision.detection.yolo import Detector
from sonar_vision.ocr.paddle import OCREngine
from sonar_vision.processing.pipeline import FrameProcessor
from sonar_vision.processing.roi import Frame


class RecordedDetector:
    def __init__(self, backend: Detector):
        self.backend = backend
        self.detections: list[Detection] = []
        self.wall_ms = 0.0

    @property
    def model_version(self) -> str:
        return self.backend.model_version

    def detect(self, frame: Frame) -> list[Detection]:
        start = time.perf_counter()
        self.detections = self.backend.detect(frame)
        self.wall_ms = (time.perf_counter() - start) * 1000
        return self.detections


class RecordedOCR:
    def __init__(self, backend: OCREngine):
        self.backend = backend
        self.calls: list[dict[str, Any]] = []

    def read(self, roi: Frame) -> list[OCRReading]:
        start = time.perf_counter()
        readings = self.backend.read(roi)
        self.calls.append(
            {
                "readings": [reading.model_dump() for reading in readings],
                "wall_ms": (time.perf_counter() - start) * 1000,
                "roi_shape": list(roi.shape),
            }
        )
        return readings


def code_visible_in_ocr(text: str, expected: str) -> bool:
    """Metric only: allow displayed slash/hyphen, never supply text to runtime OCR."""
    normalized = unicodedata.normalize("NFKC", text).upper()
    pattern = re.escape(expected.upper()).replace(r"\-", r"\s*[-/]\s*")
    return re.search(r"(?<![A-Z0-9])" + pattern + r"(?![A-Z0-9])", normalized) is not None


def observation_routes(processor: FrameProcessor) -> list[dict[str, Any]]:
    return [
        {
            "track_id": track,
            "normalization": asdict(history[-1].normalized),
            "ocr_confidence": history[-1].ocr_confidence,
            "detection_confidence": history[-1].detection_confidence,
        }
        for track, history in processor.temporal.history.items()
        if history
    ]
