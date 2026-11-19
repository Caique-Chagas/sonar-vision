"""Desktop software tests; these fixture readings are not vision evidence."""

import numpy as np

from sonar_vision.core.contracts import OCRReading
from sonar_vision.desktop.engine import DesktopOCR, next_sample
from sonar_vision.desktop.profiles import profiles
from sonar_vision.desktop.render import render


def test_model_domain_and_unavailable_bangkok_are_explicit():
    p = profiles()
    assert p["curitiba"].imgsz == 1280
    assert p["sao-paulo"].weights == p["curitiba"].weights
    assert p["bangkok"].weights == "models/best.pt"
    assert p["bangkok"].video is None
    frame = render(p["bangkok"], None, "Sem mídia", False)
    assert frame.shape == (720, 1280, 3)


def test_first_frame_is_not_skipped_and_sampling_keeps_coordinates():
    class Capture:
        index = 0

        def read(self):
            if self.index >= 6:
                return False, None
            value = self.index
            self.index += 1
            return True, np.full((64, 96, 3), value, np.uint8)

    cap = Capture()
    results = []
    index = 0
    while (sample := next_sample(cap, index, 10)) is not None:
        results.append(sample[0])
        index = sample[0] + 1
        assert sample[2][0, 0, 0] == sample[0]
    assert results == [0, 2, 4]


def test_direct_fallback_runs_only_on_same_predicted_roi_when_e3_empty():
    roi = np.zeros((30, 120, 3), np.uint8)

    class Recognizer:
        def predict(self, input, batch_size):
            assert input[0] is roi and batch_size == 1
            return [{"rec_text": "RAW DIRECT 123", "rec_score": 0.23}]

    class E3:
        recognizer = Recognizer()

        def read(self, image):
            assert image is roi
            return []

    ocr = DesktopOCR(E3())
    result = ocr.read(roi)
    assert result[0].text == "RAW DIRECT 123" and result[0].confidence == 0.23
    assert "experimental" in ocr.calls[0]


def test_nonempty_e3_readings_are_never_replaced():
    class E3:
        def read(self, image):
            return [OCRReading(text="REAL RAW", confidence=0.8)]

    ocr = DesktopOCR(E3())
    assert ocr.read(np.zeros((10, 20, 3), np.uint8))[0].text == "REAL RAW"
    assert ocr.calls == ["E3"]


def test_missing_city_session_never_loads_other_model_or_reuses_readings(tmp_path):
    from sonar_vision.desktop.app import Session

    weights = tmp_path / "models/best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"software-fixture-only")
    session = Session(tmp_path, profiles()["bangkok"], tmp_path / "run")
    assert session.result is None and session.engine is None and session.cap is None
    report = session.close()
    assert report["model_loaded"] is False and report["frames_processed"] == 0
    assert not report["raw_texts"] and "Bangkok" in report["unavailable"]


def test_ocr_failure_still_records_attempt_for_pipeline_uncertainty():
    import pytest

    class Broken:
        def read(self, image):
            raise RuntimeError("software-fixture")

    reader = DesktopOCR(Broken())
    with pytest.raises(RuntimeError):
        reader.read(np.zeros((20, 50, 3), np.uint8))
    assert reader.calls == ["E3"]
