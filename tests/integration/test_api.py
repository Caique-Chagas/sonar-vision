import socket
import time

import pytest
from fastapi.testclient import TestClient

from sonar_vision.api.app import create_app


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


def test_health_and_accessible_interface(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert 'lang="pt-BR"' in page.text and 'aria-live="polite"' in page.text
    assert '<label for="route">' in page.text and "Protótipo acadêmico" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"route_requested": "bad"},
        {"route_requested": "971D-10", "mode": "real"},
        {"route_requested": "971D-10", "mode": "simulation", "fps": 0},
        {"route_requested": "971D-10", "mode": "simulation", "max_frames": 50000},
    ],
)
def test_invalid_requests(client, payload):
    assert client.post("/api/sessions", json=payload).status_code == 422


def test_missing_model_503_and_ssrf_422(client, monkeypatch, settings):
    settings.video_hosts = "video.example"
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 443))]
    )
    response = client.post(
        "/api/sessions",
        json={"route_requested": "971D-10", "source_url": "https://video.example/bus.mp4"},
    )
    assert response.status_code == 503
    assert (
        client.post(
            "/api/sessions",
            json={"route_requested": "971D-10", "source_url": "http://127.0.0.1/test.mp4"},
        ).status_code
        == 422
    )


def test_simulation_session_frames_events_and_stop(client):
    response = client.post(
        "/api/sessions",
        json={
            "route_requested": "971D-10",
            "mode": "simulation",
            "fps": 10,
            "max_frames": 8,
            "max_seconds": 2,
        },
    )
    assert response.status_code == 202
    sid = response.json()["session_id"]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get(f"/api/sessions/{sid}").json()
        if state["state"] == "COMPLETED":
            break
        time.sleep(0.03)
    assert state["state"] == "COMPLETED"
    events = client.get(f"/api/sessions/{sid}/events").json()
    assert any(e["match_status"] == "MATCH" for e in events)
    assert all(e["simulated"] and "image" not in e for e in events)
    assert client.get(f"/api/sessions/{sid}/frame").headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/sessions/{sid}/legacy").json()[-1]["tempo_parado"] is None
    assert client.delete(f"/api/sessions/{sid}").status_code == 202
    assert client.get("/api/sessions/missing").status_code == 404


def test_session_limit_and_cancel(client, settings):
    settings.max_sessions = 1
    payload = {"route_requested": "971D-10", "mode": "simulation"}
    sid = client.post("/api/sessions", json=payload).json()["session_id"]
    assert client.post("/api/sessions", json=payload).status_code == 429
    assert client.delete(f"/api/sessions/{sid}").status_code == 202


def test_cross_origin_and_rebinding_rejected(client):
    assert (
        client.post(
            "/api/sessions", json={}, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert client.get("/health", headers={"Host": "evil.example"}).status_code == 400


def test_oversized_payload_and_local_api_docs(client):
    assert client.post("/api/sessions", content="x" * 17000).status_code == 413
    assert "openapi.json" in client.get("/docs").text
