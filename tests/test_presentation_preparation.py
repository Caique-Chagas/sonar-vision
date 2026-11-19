"""Software media/ROI fixtures, not detector accuracy evidence."""

import json

import cv2
import numpy as np
import pytest

from sonar_vision.core.contracts import Box, Detection, OCRReading
from sonar_vision.desktop.presentation import ClipSpec, prepare_clip, verify_prepared
from sonar_vision.processing.local_video import DemoFrameAnalyzer


@pytest.fixture
def clip(tmp_path):
    source = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter.fourcc(*"mp4v"), 12, (160, 96))
    for i in range(8):
        frame = np.full((96, 160, 3), 40 + i * 8, np.uint8)
        cv2.rectangle(frame, (i * 3, 35), (i * 3 + 15, 50), (0, 100, 0), -1)
        writer.write(frame)
    writer.release()
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"SOFTWARE FIXTURE, NOT A MODEL")
    return ClipSpec("fixture", str(source), 2, 4, "FIXTURE"), dict(bus=model, sign=model)


def factory():
    class Detector:
        model_version = "software-fixture"
        expected = None

        def detect(self, frame):
            self.expected = frame[10:74, 20:100].copy()
            return [
                Detection(
                    box=Box(x1=20, y1=10, x2=100, y2=74), confidence=0.8, label="destination_sign"
                )
            ]

    detector = Detector()

    class OCR:
        def read(self, roi):
            assert np.array_equal(roi, detector.expected)
            return [OCRReading(text="FIXTURE RAW", confidence=0.9)]

    return DemoFrameAnalyzer(detector, OCR())


def test_fps_duration_timestamps_and_predicted_roi_are_preserved(tmp_path, clip):
    spec, models = clip
    out = tmp_path / "out"
    result = prepare_clip(spec, out, analyzer_factory=factory, model_files=models)
    assert result["annotated"]["fps"] == pytest.approx(12, abs=0.001)
    assert result["annotated"]["frames"] == 4
    assert result["duration_seconds"] == pytest.approx(4 / 12)
    rows = [json.loads(x) for x in (out / "predictions.jsonl").read_text().splitlines()]
    assert [r["source_index"] for r in rows] == [2, 3, 4, 5]
    assert [r["timestamp_seconds"] for r in rows] == pytest.approx([0, 1 / 12, 2 / 12, 3 / 12])
    assert [r["source_timestamp_seconds"] for r in rows] == pytest.approx(
        [2 / 12, 3 / 12, 4 / 12, 5 / 12]
    )
    assert all(r["row"]["findings"][0]["readings"][0]["text"] == "FIXTURE RAW" for r in rows)
    assert result["binding"]["evidence_kind"] == "software_fixture"
    assert verify_prepared(out)["status"] == "READY"

    def forbidden():
        raise AssertionError("Playback/cache must not load models again")

    assert (
        prepare_clip(spec, out, analyzer_factory=forbidden, model_files=models)["status"] == "READY"
    )
    (out / "predictions.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="alterado"):
        verify_prepared(out)


def test_invalid_video_never_loads_models(tmp_path):
    spec = ClipSpec("missing", str(tmp_path / "absent.mp4"), 0, 4, "FIXTURE")

    def forbidden():
        raise AssertionError("Should fail before model load")

    with pytest.raises(ValueError, match="ausente"):
        prepare_clip(spec, tmp_path / "out", analyzer_factory=forbidden)


def test_out_of_range_segment_rejected_before_models(tmp_path, clip):
    spec, models = clip
    with pytest.raises(ValueError, match="Intervalo"):
        prepare_clip(
            ClipSpec("bad", spec.source, 6, 10, "FIXTURE"),
            tmp_path / "out",
            analyzer_factory=factory,
            model_files=models,
        )


@pytest.mark.parametrize("fps", [30, 60000 / 1001])
def test_real_encoder_preserves_integer_and_fractional_source_fps(tmp_path, clip, fps):
    spec, models = clip
    source = tmp_path / "rate.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter.fourcc(*"mp4v"), fps, (160, 96))
    for i in range(8):
        writer.write(np.full((96, 160, 3), 50 + i, np.uint8))
    writer.release()
    result = prepare_clip(
        ClipSpec("rate", str(source), 2, 4, "FIXTURE"),
        tmp_path / "rate",
        analyzer_factory=factory,
        model_files=models,
    )
    assert result["annotated"]["fps"] == pytest.approx(result["source"]["fps"], abs=0.001)
    assert result["annotated"]["fps"] == pytest.approx(fps, abs=0.01)
    assert result["annotated"]["duration_seconds"] == pytest.approx(
        4 / result["source"]["fps"], abs=0.001
    )


def test_interrupted_encoding_reuses_all_predictions_without_loading_models(tmp_path, clip):
    spec, models = clip
    out = tmp_path / "out"
    prepare_clip(spec, out, analyzer_factory=factory, model_files=models)
    (out / "manifest.json").unlink()

    def forbidden():
        raise AssertionError("Cached predictions must be reused")

    result = prepare_clip(spec, out, analyzer_factory=forbidden, model_files=models)
    assert result["reused_frames"] == 4
    assert len(list(out.glob("annotated.mp4.unsealed-*"))) == 1


def test_misaligned_prediction_cache_is_rejected_before_rendering(tmp_path, clip):
    spec, models = clip
    out = tmp_path / "out"
    prepare_clip(spec, out, analyzer_factory=factory, model_files=models)
    (out / "manifest.json").unlink()
    path = out / "frames/000000.json"
    row = json.loads(path.read_text())
    row["input_sha256"] = "incorrect"
    path.write_text(json.dumps(row))
    before = (out / "annotated.mp4").read_bytes()
    with pytest.raises(ValueError, match="desalinhados"):
        prepare_clip(spec, out, analyzer_factory=factory, model_files=models)
    assert (out / "annotated.mp4").read_bytes() == before
