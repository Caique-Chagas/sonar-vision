"""Software regression fixtures, never real model performance evidence."""

import json

import numpy as np
import pytest

from sonar_vision.core.contracts import Box, Detection, OCRReading
from sonar_vision.desktop import v3
from sonar_vision.training.dataset import sha256


def test_two_class_bus_overlay_never_enters_ocr_pixels():
    frame = np.full((100, 160, 3), 75, np.uint8)

    class Detector:
        model_version = "software-fixture"

        def detect(self, image):
            assert image is frame
            return [
                Detection(label="bus", box=Box(x1=5, y1=5, x2=155, y2=95), confidence=0.9),
                Detection(
                    label="destination_sign", box=Box(x1=20, y1=10, x2=140, y2=35), confidence=0.8
                ),
            ]

    class OCR:
        def read(self, roi):
            assert np.all(roi == 75)
            return [OCRReading(text="FIXTURE RAW", confidence=0.8)]

    canvas, row = v3.VideoV3Analyzer(Detector(), OCR()).analyze(frame)
    assert np.all(frame == 75)
    assert canvas.shape[1] == 800
    assert len(row["buses"]) == 1 and len(row["findings"]) == 1
    assert row["findings"][0]["readings"][0]["text"] == "FIXTURE RAW"
    assert row["temporal_confirmation"] is False
    assert row["rights_verified"] is False
    assert row["inference_seconds"] > 0


def test_v3_rejects_single_class_weights_before_inference(tmp_path):
    class Model:
        names = {0: "destination_sign"}

    with pytest.raises(ValueError, match="DUAS"):
        v3.VideoV3Detector(tmp_path / "fixture.pt", model=Model())


def test_private_active_configuration_binds_weights_and_scope(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    monkeypatch.setattr(v3, "ACTIVE", active)
    assert v3.active_configuration() is None
    weight = tmp_path / "fixture.pt"
    weight.write_bytes(b"SOFTWARE FIXTURE")
    data = dict(
        status="VALIDATED_PRIVATE",
        rights_verified=False,
        weights=str(weight),
        weights_sha256=sha256(weight),
    )
    active.write_text(json.dumps(data))
    assert v3.active_configuration()["weights"] == str(weight)
    weight.write_bytes(b"ALTERED")
    with pytest.raises(ValueError, match="alterados"):
        v3.active_configuration()
