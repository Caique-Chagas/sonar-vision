"""Software fixtures only; native positive evidence is collected separately."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sonar_vision.api.app import create_app
from sonar_vision.core.contracts import Box, Detection, OCRReading
from sonar_vision.processing.local_video import DemoFrameAnalyzer, LocalVideoError
from sonar_vision.processing.photograph import (
    analyze_photograph,
    decode_photograph,
    validate_image_name,
)
from sonar_vision.processing.roi import extract_roi


@pytest.mark.parametrize(
    "filename,mime,consent",
    [
        ("../a.jpg", "image/jpeg", "confirmed"),
        ("a.svg", "image/svg+xml", "confirmed"),
        ("a.jpg", "image/png", "confirmed"),
        ("a.png", "image/png", ""),
    ],
)
def test_photo_names_formats_and_permission(filename, mime, consent):
    with pytest.raises(LocalVideoError):
        validate_image_name(filename, mime, consent)


def test_photo_decode_content_and_pixel_budget(tmp_path):
    path = tmp_path / "photo"
    path.write_bytes(b"<html>not an image</html>")
    with pytest.raises(LocalVideoError):
        decode_photograph(path)
    Image.new("RGB", (4097, 2)).save(path, format="PNG")
    with pytest.raises(LocalVideoError, match="4096"):
        decode_photograph(path)
    Image.new("RGB", (100, 60), (10, 20, 30)).save(path, format="PNG")
    frame = decode_photograph(path)
    assert frame.shape == (60, 100, 3)
    assert frame[0, 0].tolist() == [30, 20, 10]


def test_animated_photo_rejected(tmp_path):
    path = tmp_path / "animated.png"
    Image.new("RGB", (20, 20)).save(
        path, save_all=True, append_images=[Image.new("RGB", (20, 20), "red")]
    )
    with pytest.raises(LocalVideoError, match="animação"):
        decode_photograph(path)


def test_photo_uses_predicted_crop_preserves_text_and_uncertainty(tmp_path):
    box = Box(x1=4, y1=6, x2=65, y2=24)

    class Detector:
        model_version = "software-fixture-not-native"

        def detect(self, image):
            return [Detection(box=box, confidence=0.8, label="destination_sign")]

    class OCR:
        def read(self, roi):
            self.roi = roi.copy()
            return [OCRReading(text="333 TERMINAL", confidence=0.99)]

    image = np.arange(60 * 100 * 3, dtype=np.uint8).reshape(60, 100, 3)
    (tmp_path / "source-image").write_bytes(b"software-fixture")
    ocr = OCR()
    report = analyze_photograph(image, tmp_path, DemoFrameAnalyzer(Detector(), ocr, static=True))
    assert np.array_equal(cv2.imread(str(tmp_path / "roi-0.png")), ocr.roi)
    assert np.array_equal(ocr.roi, extract_roi(image, box))
    assert report["findings"][0]["readings"][0]["text"] == "333 TERMINAL"
    assert report["findings"][0]["code"] is None
    assert report["state"] == "UNCERTAIN"
    assert report["mode"] == "static_photograph"
    assert report["accuracy"] is None and not report["ground_truth_used"]
    assert not report["temporal_confirmation"]


def photo_fixture_worker(folder, weights, output):
    path = Path(folder)
    (path / "photo-report.json").write_text(json.dumps({"fixture": True}))
    (path / "photo-original.png").write_bytes(b"software-fixture")
    output.put({"kind": "done", "report": {"fixture": True}})


@pytest.fixture
def client_photo(tmp_path, settings):
    app = create_app(settings)
    manager = app.state.video_jobs
    manager.root = tmp_path / "jobs"
    manager.weights = tmp_path / "weights.pt"
    manager.weights.write_bytes(b"fixture")
    manager.worker = photo_fixture_worker
    with TestClient(app) as client:
        yield client, manager


def test_photo_upload_job_download_and_cleanup(client_photo):
    client, manager = client_photo
    headers = {"Content-Type": "image/jpeg", "X-Sonar-Media-Permission": "confirmed"}
    body = b"\xff\xd8\xff" + b"x" * 32
    response = client.post(
        "/api/video-demo/photo/jobs?filename=a.jpg", content=body, headers=headers
    )
    assert response.status_code == 202
    id = response.json()["job_id"]
    manager.jobs[id].thread.join(timeout=8)
    state = client.get("/api/video-demo/jobs/" + id).json()
    assert state["state"] == "COMPLETED" and state["mode"] == "static_photograph"
    assert client.get(f"/api/video-demo/photo/jobs/{id}/files/report").json() == {"fixture": True}
    assert client.get(f"/api/video-demo/photo/jobs/{id}/files/roi-999").status_code == 404
    assert not (manager.jobs[id].folder / "source-image").exists()


def test_photo_invalid_content_limits_and_shared_video_budget(client_photo, monkeypatch):
    client, manager = client_photo
    headers = {"Content-Type": "image/jpeg", "X-Sonar-Media-Permission": "confirmed"}
    assert (
        client.post(
            "/api/video-demo/photo/jobs?filename=a.jpg", content=b"x" * 20, headers=headers
        ).status_code
        == 422
    )
    assert not manager.jobs
    monkeypatch.setattr("sonar_vision.api.photo_routes.MAX_IMAGE", 24)
    assert (
        client.post(
            "/api/video-demo/photo/jobs?filename=a.jpg", content=b"x" * 30, headers=headers
        ).status_code
        == 413
    )
    assert not manager.jobs
    job = manager.reserve()
    assert (
        client.post(
            "/api/video-demo/photo/jobs?filename=a.jpg", content=b"x" * 20, headers=headers
        ).status_code
        == 429
    )
    manager.discard_upload(job)
    assert client.post("/api/sessions", content=b"x" * 20000).status_code == 413
