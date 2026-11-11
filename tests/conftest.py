import numpy as np
import pytest

from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import Box, Detection, OCRReading


@pytest.fixture
def frame():
    return np.zeros((240, 400, 3), dtype=np.uint8)


@pytest.fixture
def sign():
    return Detection(box=Box(x1=20, y1=40, x2=220, y2=90), confidence=0.9, label="destination_sign")


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, model_path=tmp_path / "missing.pt")


class FakeDetector:
    model_version = "test-fixture"

    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


class FakeOCR:
    def __init__(self, text="971D-10", confidence=0.95):
        self.text, self.confidence = text, confidence

    def read(self, roi):
        return [OCRReading(text=self.text, confidence=self.confidence)] if self.text else []
