import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sonar_vision.api.app import create_app


def fixture_worker(folder, weights, output):
    """Only verifies transport/lifecycle; not real inference or a valid media result."""
    path = Path(folder)
    (path / "annotated.mp4").write_bytes(b"software-fixture-only")
    (path / "report.json").write_text(json.dumps({"fixture": True}))
    (path / "frames.jsonl").write_text("{}\n")
    output.put({"kind": "done", "report": {"fixture": True}})


def waiting_worker(folder, weights, output):
    """Lifecycle fixture: never a model or a vision result."""
    (Path(folder) / "partial.txt").write_text("fixture")
    output.put({"kind": "frame", "metrics": {"fixture": True}})
    time.sleep(20)


def failed_worker(folder, weights, output):
    (Path(folder) / "partial.txt").write_text("fixture")
    output.put({"kind": "error", "error": "Invalid video fixture"})


@pytest.fixture
def video_client(tmp_path, settings):
    app = create_app(settings)
    app.state.video_jobs.root = tmp_path / "jobs"
    app.state.video_jobs.weights = tmp_path / "weights.pt"
    app.state.video_jobs.weights.write_bytes(b"fixture-not-a-model")
    app.state.video_jobs.worker = fixture_worker
    with TestClient(app) as client:
        yield client, app.state.video_jobs


def headers():
    return {"Content-Type": "video/mp4", "X-Sonar-Media-Permission": "confirmed"}


def test_invalid_content_cleanup_and_no_worker(video_client):
    client, manager = video_client
    response = client.post(
        "/api/video-demo/jobs?filename=clip.mp4",
        content=b"<html>not an MP4</html>",
        headers=headers(),
    )
    assert response.status_code == 422
    assert not manager.jobs
    assert not list(manager.root.glob("run-*"))


def test_unknown_profile_rejected_before_reservation(video_client):
    client, manager = video_client
    response = client.post(
        "/api/video-demo/jobs?filename=clip.mp4&profile=arbitrary",
        content=b"\x00\x00\x00\x18ftypisom" + b"x" * 32,
        headers=headers(),
    )
    assert response.status_code == 422
    assert not manager.jobs


def test_permission_bad_names_and_legacy_limits(video_client):
    client, _ = video_client
    assert (
        client.post(
            "/api/video-demo/jobs?filename=clip.mp4",
            content=b"x" * 20,
            headers={"Content-Type": "video/mp4"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/video-demo/jobs?filename=../clip.mp4", content=b"x" * 20, headers=headers()
        ).status_code
        == 422
    )
    assert client.post("/api/sessions", content=b"x" * 20000).status_code == 413
    assert "media-src 'self' blob:" in client.get("/").headers["content-security-policy"]


def test_progress_download_ranges_and_original_input_cleanup(video_client):
    client, manager = video_client
    response = client.post(
        "/api/video-demo/jobs?filename=clip.mp4",
        content=b"\x00\x00\x00\x18ftypisom" + b"x" * 32,
        headers=headers(),
    )
    assert response.status_code == 202
    id = response.json()["job_id"]
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        state = client.get("/api/video-demo/jobs/" + id).json()
        if state["state"] in ["COMPLETED", "ERROR"]:
            break
        time.sleep(0.05)
    assert state["state"] == "COMPLETED"
    assert state["profile"] == "baseline"
    assert json.loads((manager.jobs[id].folder / "job-config.json").read_text()) == {
        "profile": "baseline"
    }
    assert not state["temporal_confirmation"]
    assert client.get(f"/api/video-demo/jobs/{id}/files/video").content == b"software-fixture-only"
    assert (
        client.get(
            f"/api/video-demo/jobs/{id}/files/video", headers={"Range": "bytes=0-3"}
        ).status_code
        == 206
    )
    assert client.get(f"/api/video-demo/jobs/{id}/files/report").json() == {"fixture": True}
    assert client.get(f"/api/video-demo/jobs/{id}/files/detection").status_code == 404
    manager.jobs[id].thread.join(timeout=3)
    assert not (manager.jobs[id].folder / "source.mp4").exists()
    assert client.get("/api/video-demo/jobs/" + ("a" * 32) + "/files/video").status_code == 404


def test_single_active_reservation_and_failed_upload_does_not_escape(video_client):
    client, manager = video_client
    job = manager.reserve()
    assert (
        client.post(
            "/api/video-demo/jobs?filename=clip.mp4",
            content=b"\x00\x00\x00\x18ftypisom" + b"x" * 32,
            headers=headers(),
        ).status_code
        == 429
    )
    manager.discard_upload(job)


def test_origin_guard_applies_to_video_upload(video_client):
    client, _ = video_client
    assert (
        client.post(
            "/api/video-demo/jobs?filename=clip.mp4",
            content=b"x" * 20,
            headers=headers() | {"Origin": "https://untrusted.example"},
        ).status_code
        == 403
    )


def test_stream_budget_enforced_even_if_header_lies(video_client, monkeypatch):
    client, manager = video_client
    monkeypatch.setattr("sonar_vision.api.video_routes.MAX_UPLOAD", 32)
    assert (
        client.post(
            "/api/video-demo/jobs?filename=clip.mp4",
            content=b"x" * 64,
            headers=headers() | {"Content-Length": "1"},
        ).status_code
        == 413
    )
    assert not manager.jobs


@pytest.mark.parametrize("cancel", [True, False])
def test_cancel_or_worker_failure_removes_uploaded_and_partial_files(video_client, cancel):
    client, manager = video_client
    manager.worker = waiting_worker if cancel else failed_worker
    response = client.post(
        "/api/video-demo/jobs?filename=clip.mp4",
        content=b"\x00\x00\x00\x18ftypisom" + b"x" * 32,
        headers=headers(),
    )
    id = response.json()["job_id"]
    if cancel:
        assert client.delete("/api/video-demo/jobs/" + id).status_code == 202
    manager.jobs[id].thread.join(timeout=8)
    assert manager.jobs[id].ended is not None
    assert client.get("/api/video-demo/jobs/" + id).json()["state"] == (
        "CANCELLED" if cancel else "ERROR"
    )
    assert not list(manager.jobs[id].folder.iterdir())
    assert client.get(f"/api/video-demo/jobs/{id}/files/video").status_code == 409
