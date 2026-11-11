"""In-memory bounded session registry; real ML runs in a cancellable subprocess."""

import logging
import multiprocessing as mp
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import cv2

from sonar_vision.api.schemas import SessionRequest
from sonar_vision.core.config import Settings
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.integrations.sonar import FixtureRouteProvider
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.pipeline import Context, FrameProcessor
from sonar_vision.processing.simulation import (
    SimulationDetector,
    SimulationOCR,
    simulation_frame,
)
from sonar_vision.processing.sources import VideoSource
from sonar_vision.speech.announcements import announcement

logger = logging.getLogger(__name__)


def real_worker(request_data: dict[str, Any], settings_data: dict[str, Any], output: Any) -> None:
    """No URLs or low-level exceptions are emitted to clients or logs."""
    try:
        request, settings = SessionRequest(**request_data), Settings(**settings_data)
        processor = FrameProcessor(
            YoloSignDetector(settings.model_path, settings.min_detection),
            PaddleEngine(),
            Context(
                request.route_requested, request.stop_id, request.waiting_session_id, "video_stream"
            ),
            settings,
        )
        with VideoSource(
            request.source_url or "",
            settings.allowed_hosts,
            max_frames=request.max_frames,
            max_seconds=request.max_seconds,
            fps=request.fps,
        ) as source:
            for index, timestamp, frame in source:
                start = time.perf_counter()
                annotated, events = processor.process(frame, index, timestamp)
                elapsed = time.perf_counter() - start
                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                output.put(
                    {
                        "kind": "frame",
                        "index": index,
                        "jpeg": encoded.tobytes() if ok else None,
                        "latency": elapsed,
                        "detections_total": processor.detections_total,
                        "ocr_calls": processor.ocr_calls,
                        "events": [
                            e.model_dump(mode="json") | {"announcement": announcement(e)}
                            for e in events
                        ],
                    },
                    timeout=10,
                )
        output.put({"kind": "done"}, timeout=5)
    except Exception as exc:
        output.put({"kind": "error", "error": type(exc).__name__}, timeout=5)


@dataclass
class Session:
    id: str
    request: SessionRequest
    state: str = "STARTING"
    error: str | None = None
    frames: int = 0
    detections_total: int = 0
    ocr_calls: int = 0
    jpeg: bytes | None = None
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=1200))
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            elapsed = (self.ended or time.monotonic()) - self.started
            return {
                "session_id": self.id,
                "mode": self.request.mode,
                "state": self.state,
                "error": self.error,
                "frames_processed": self.frames,
                "detections_total": self.detections_total,
                "ocr_calls": self.ocr_calls,
                "events": list(self.events),
                "latency_ms_mean": (
                    sum(self.latencies) / len(self.latencies) * 1000 if self.latencies else None
                ),
                "observed_fps": self.frames / elapsed if elapsed > 0 else 0,
                "metrics_scope": "simulation" if self.request.mode == "simulation" else "session",
                "context_source": "fictional_fixture"
                if self.request.mode == "simulation"
                else "no_operational_context",
            }


class SessionManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()

    def create(self, request: SessionRequest) -> Session:
        with self.lock:
            active = [s for s in self.sessions.values() if s.ended is None]
            if len(active) >= self.settings.max_sessions:
                raise RuntimeError("Limite de sessões simultâneas atingido")
            if request.mode == "real" and any(s.request.mode == "real" for s in active):
                raise RuntimeError("Já existe uma sessão real em execução")
            while len(self.sessions) >= self.settings.session_retention:
                finished = next((k for k, s in self.sessions.items() if s.ended is not None), None)
                if finished is None:
                    raise RuntimeError("Histórico ocupado por sessões ativas")
                del self.sessions[finished]
            session = Session(uuid4().hex, request)
            self.sessions[session.id] = session
            session.thread = threading.Thread(target=self._run, args=(session,), daemon=True)
            session.thread.start()
            logger.info("session_started", extra={"session_id": session.id, "mode": request.mode})
            return session

    def get(self, session_id: str) -> Session:
        with self.lock:
            return self.sessions[session_id]

    def _run(self, session: Session) -> None:
        try:
            if session.request.mode == "simulation":
                self._simulate(session)
            else:
                self._real(session)
            with session.lock:
                if session.state != "ERROR":
                    session.state = "STOPPED" if session.stop.is_set() else "COMPLETED"
        except Exception:
            with session.lock:
                session.state, session.error = "ERROR", "Falha de processamento da sessão"
            logger.error("session_failed", extra={"session_id": session.id})
        finally:
            with session.lock:
                session.ended = time.monotonic()
            logger.info(
                "session_ended",
                extra={
                    "session_id": session.id,
                    "state": session.state,
                    "frames_processed": session.frames,
                },
            )

    def _simulate(self, session: Session) -> None:
        request = session.request
        ocr = SimulationOCR()
        routes = FixtureRouteProvider().allowed_routes(request.stop_id)
        processor = FrameProcessor(
            SimulationDetector(),
            ocr,
            Context(
                request.route_requested, request.stop_id, request.waiting_session_id, "simulation"
            ),
            self.settings,
            routes,
        )
        session.state = "RUNNING"
        start = time.monotonic()
        for index in range(request.max_frames):
            if session.stop.is_set() or time.monotonic() - start >= request.max_seconds:
                break
            # Three independent scenes, separated by tracker TTL to avoid carrying votes.
            phase = int((time.monotonic() - start) / 6)
            if phase and phase != getattr(ocr, "phase", 0):
                processor.tracker.tracks.clear()
            ocr.phase = phase
            ocr.text = ["971D-10", "875P-10", "971D"][phase % 3]
            annotated, events = processor.process(
                simulation_frame(ocr.text), index, time.monotonic() - start
            )
            ok, jpeg = cv2.imencode(".jpg", annotated)
            with session.lock:
                session.jpeg = jpeg.tobytes() if ok else None
                session.frames += 1
                session.events.extend(
                    e.model_dump(mode="json") | {"announcement": announcement(e)} for e in events
                )
            if session.stop.wait(1 / request.fps):
                break

    def _real(self, session: Session) -> None:
        context = mp.get_context("spawn")
        output = context.Queue(maxsize=3)
        process = context.Process(
            target=real_worker,
            args=(session.request.model_dump(), self.settings.model_dump(), output),
            daemon=True,
        )
        process.start()
        try:
            while not session.stop.is_set():
                if time.monotonic() - session.started > session.request.max_seconds + 90:
                    raise TimeoutError("Prazo máximo da sessão incluindo inicialização excedido")
                try:
                    message = output.get(timeout=0.25)
                except queue.Empty:
                    if not process.is_alive():
                        raise RuntimeError("Processo de visão encerrado sem resultado") from None
                    continue
                if message["kind"] == "done":
                    break
                if message["kind"] == "error":
                    with session.lock:
                        session.state = "ERROR"
                        session.error = (
                            "Não foi possível processar a fonte ou carregar os modelos. "
                            f"Categoria: {message['error']}. Consulte README e pesos customizados."
                        )
                    break
                with session.lock:
                    session.state, session.jpeg = "RUNNING", message["jpeg"]
                    session.frames = message["index"] + 1
                    session.detections_total = message["detections_total"]
                    session.ocr_calls = message["ocr_calls"]
                    session.events.extend(message["events"])
                    session.latencies.append(message["latency"])
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            output.cancel_join_thread()
            output.close()

    def close(self) -> None:
        for session in list(self.sessions.values()):
            session.stop.set()
        for session in list(self.sessions.values()):
            if session.thread:
                session.thread.join(timeout=5)
