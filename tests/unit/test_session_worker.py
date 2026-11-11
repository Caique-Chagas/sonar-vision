import queue
import time
from types import SimpleNamespace

import pytest

from sonar_vision.api.schemas import SessionRequest
from sonar_vision.api.sessions import Session, SessionManager


def fake_context(monkeypatch, messages):
    class Output:
        def get(self, timeout):
            if messages:
                return messages.pop(0)
            raise queue.Empty

        def cancel_join_thread(self):
            pass

        def close(self):
            pass

    class Process:
        alive = True
        terminated = False

        def start(self):
            pass

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout):
            pass

    process = Process()
    context = SimpleNamespace(Queue=lambda **k: Output(), Process=lambda **k: process)
    monkeypatch.setattr("sonar_vision.api.sessions.mp.get_context", lambda method: context)
    return process


def test_worker_reports_sanitized_external_error_and_closes(settings, monkeypatch):
    process = fake_context(monkeypatch, [{"kind": "error", "error": "SourceError"}])
    session = Session(
        "id",
        SessionRequest(
            route_requested="971D-10",
            mode="real",
            source_url="https://example.org/video.mp4?token=private",
        ),
    )
    SessionManager(settings)._real(session)
    assert session.state == "ERROR" and "SourceError" in session.error
    assert "private" not in session.error and process.terminated


def test_worker_deadline_terminates_process(settings, monkeypatch):
    process = fake_context(monkeypatch, [])
    session = Session(
        "id",
        SessionRequest(route_requested="971D-10", mode="simulation"),
        started=time.monotonic() - 500,
    )
    with pytest.raises(TimeoutError):
        SessionManager(settings)._real(session)
    assert process.terminated


def test_worker_cancellation_terminates_process(settings, monkeypatch):
    process = fake_context(monkeypatch, [])
    session = Session("id", SessionRequest(route_requested="971D-10", mode="simulation"))
    session.stop.set()
    SessionManager(settings)._real(session)
    assert process.terminated
