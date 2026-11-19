"""Video software fixtures below are synthetic and never model-accuracy evidence."""

import json

import cv2
import numpy as np
import pytest

from sonar_vision.core.contracts import Box, Detection, OCRReading
from sonar_vision.processing.local_video import (
    DemoFrameAnalyzer,
    LocalVideoError,
    analyze_video,
    demo_imgsz,
    inspect_video,
    validate_upload_name,
)


class Detector:
    model_version = "injected-software-fixture"

    def detect(self, frame):
        return [
            Detection(box=Box(x1=5, y1=5, x2=80, y2=30), confidence=0.91, label="destination_sign")
        ]


class OCR:
    def __init__(self, text="6030-10"):
        self.text = text

    def read(self, roi):
        return [OCRReading(text=self.text, confidence=0.93)]


def fixture_mp4(path, frames=6, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (96, 64))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((64, 96, 3), index * 15, dtype=np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize(
    "filename,mime,permission",
    [
        ("a.avi", "video/mp4", "confirmed"),
        ("../a.mp4", "video/mp4", "confirmed"),
        ("a.mp4", "text/html", "confirmed"),
        ("a.mp4", "video/mp4", ""),
    ],
)
def test_upload_name_and_permission_are_required(filename, mime, permission):
    with pytest.raises(LocalVideoError):
        validate_upload_name(filename, mime, permission)


def test_false_extension_and_corrupt_container_rejected(tmp_path):
    path = tmp_path / "file.mp4"
    path.write_bytes(b"<html>not a video</html>")
    with pytest.raises(LocalVideoError, match="Conteúdo"):
        inspect_video(path)
    path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"x" * 200)
    with pytest.raises(LocalVideoError):
        inspect_video(path)


def test_duration_limit_from_real_container_metadata(tmp_path):
    path = fixture_mp4(tmp_path / "long.mp4", frames=7, fps=0.1)
    with pytest.raises(LocalVideoError, match="Duração"):
        inspect_video(path)


def test_current_prediction_and_raw_OCR_no_confirmation_or_GT_override():
    analyzer = DemoFrameAnalyzer(Detector(), OCR("971D-10"))
    image = np.zeros((64, 96, 3), np.uint8)
    for _ in range(4):
        annotated, row = analyzer.analyze(image)
        assert row["findings"][0]["code"] == "971D-10"
        assert row["findings"][0]["readings"][0]["text"] == "971D-10"
        assert row["findings"][0]["detector_confidence"] == 0.91
        assert not row["temporal_confirmation"]
        assert row["state"] == "CODE_CANDIDATE"
        assert annotated.shape[0] > image.shape[0]
        assert annotated.shape[1] >= 800
        assert row["findings"][0]["box"] == {"x1": 5.0, "y1": 5.0, "x2": 80.0, "y2": 30.0}


def test_low_score_and_conflicting_readings_stay_uncertain():
    class ConflictOCR:
        def read(self, roi):
            return [
                OCRReading(text="6030-10", confidence=0.9),
                OCRReading(text="6030-11", confidence=0.91),
            ]

    _, row = DemoFrameAnalyzer(Detector(), ConflictOCR()).analyze(np.zeros((64, 96, 3), np.uint8))
    assert row["state"] == "UNCERTAIN" and row["findings"][0]["code"] is None


def test_OCR_failure_keeps_real_detection_and_error():
    class Broken:
        def read(self, roi):
            raise RuntimeError("fixture")

    _, row = DemoFrameAnalyzer(Detector(), Broken()).analyze(np.zeros((64, 96, 3), np.uint8))
    assert row["findings"][0]["ocr_error"] == "OCR_FAILURE"
    assert row["state"] == "UNCERTAIN"


def test_video_decoder_encoder_and_metrics_are_not_accuracy(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    source = fixture_mp4(tmp_path / "source.mp4")
    messages = []
    report = analyze_video(source, tmp_path, DemoFrameAnalyzer(Detector(), OCR()), messages.append)
    assert report["frames_processed"] == 3
    assert report["frames_with_detection"] == 3
    assert report["frames_with_code_candidate"] == 3
    assert report["accuracy"] is None and not report["ground_truth_used"]
    assert not report["temporal_confirmation"]
    assert (tmp_path / "annotated.mp4").is_file()
    assert (tmp_path / "first-detection.jpg").is_file()
    assert report["first_detection_timestamp_seconds"] == 0
    assert report["first_detection"]["findings"][0]["readings"][0]["text"] == "6030-10"
    assert len((tmp_path / "frames.jsonl").read_text().splitlines()) == 3
    assert (
        json.loads((tmp_path / "report.json").read_text())["mode"]
        == "recorded_video_offline_sampled"
    )
    assert inspect_video(tmp_path / "annotated.mp4")["frames"] == 3


def test_no_detection_does_not_call_OCR(tmp_path):
    class Empty:
        model_version = "fixture"

        def detect(self, frame):
            return []

    class Forbidden:
        def read(self, roi):
            raise AssertionError("must not call without ROI")

    _, row = DemoFrameAnalyzer(Empty(), Forbidden()).analyze(np.zeros((64, 96, 3), np.uint8))
    assert row["state"] == "NO_PANEL" and not row["findings"]


def test_static_evidence_is_labelled_and_short_regional_reading_remains_raw():
    _, row = DemoFrameAnalyzer(Detector(), OCR("333 TERMINAL"), static=True).analyze(
        np.zeros((64, 96, 3), np.uint8)
    )
    assert row["modality"] == "static_photograph"
    assert row["findings"][0]["readings"][0]["text"] == "333 TERMINAL"
    assert row["state"] == "UNCERTAIN" and not row["temporal_confirmation"]


def test_diagnostic_resolution_is_explicit_and_does_not_change_baseline():
    assert demo_imgsz("baseline") == 640
    assert demo_imgsz("diagnostic1280") == 1280
    with pytest.raises(LocalVideoError, match="Perfil"):
        demo_imgsz("custom-low-threshold")
