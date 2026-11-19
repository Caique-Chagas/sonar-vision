"""One isolated video job, bounded retention and explicit cleanup."""

import multiprocessing as mp
import os
import queue
import re
import shutil
import signal
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from sonar_vision.processing.local_video import MAX_WALL_SECONDS, video_worker


@dataclass
class VideoJob:
    id: str
    folder: Path
    state: str = "UPLOADING"
    created: float = field(default_factory=time.monotonic)
    ended: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    jpeg: bytes | None = None
    error: str | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.id,
                "state": self.state,
                "error": self.error,
                "elapsed_seconds": (self.ended or time.monotonic()) - self.created,
                "mode": "recorded_video_offline_sampled",
                "temporal_confirmation": False,
                **self.data,
            }


class VideoJobManager:
    def __init__(self, root: Path, weights: Path, worker: Any = video_worker):
        self.root, self.weights, self.worker = root, weights, worker
        self.jobs: dict[str, VideoJob] = {}
        self.lock = threading.Lock()

    def reserve(self) -> VideoJob:
        with self.lock:
            if any(j.ended is None for j in self.jobs.values()):
                raise RuntimeError("Já existe um upload ou vídeo em processamento")
            # Never delete a previous server's folders: clean only owned jobs.
            while len(self.jobs) >= 2:
                key = next(iter(self.jobs))
                job = self.jobs.pop(key)
                shutil.rmtree(job.folder, ignore_errors=True)
            self.root.mkdir(parents=True, exist_ok=True)
            if (
                sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
                > 512 * 1024 * 1024
            ):
                raise RuntimeError("Armazenamento ocupado: salve e remova resultados antigos")
            folder = Path(tempfile.mkdtemp(prefix="run-", dir=self.root))
            job = VideoJob(uuid4().hex, folder)
            self.jobs[job.id] = job
            return job

    def get(self, id: str) -> VideoJob:
        if not re.fullmatch(r"[0-9a-f]{32}", id):
            raise KeyError(id)
        with self.lock:
            return self.jobs[id]

    def discard_upload(self, job: VideoJob) -> None:
        with self.lock:
            self.jobs.pop(job.id, None)
        shutil.rmtree(job.folder, ignore_errors=True)

    def start(self, job: VideoJob) -> None:
        job.state = "STARTING"
        job.thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        job.thread.start()

    def _run(self, job: VideoJob) -> None:
        context = mp.get_context("spawn")
        out = context.Queue(maxsize=2)
        process = context.Process(
            target=self.worker, args=(str(job.folder), str(self.weights), out), daemon=True
        )
        try:
            process.start()
            started = time.monotonic()
            while not job.stop.is_set():
                if time.monotonic() - started > MAX_WALL_SECONDS:
                    raise TimeoutError("Tempo máximo de300segundos excedido")
                try:
                    message = out.get(timeout=0.25)
                except queue.Empty:
                    if not process.is_alive():
                        raise RuntimeError("Processo de vídeo encerrou sem resultado") from None
                    continue
                kind = message["kind"]
                with job.lock:
                    if kind == "done":
                        job.state = "COMPLETED"
                        job.data.update(report=message["report"], progress=1.0)
                        break
                    if kind == "error":
                        job.state = "ERROR"
                        job.error = message["error"]
                        break
                    if kind == "encoding":
                        job.state = "ENCODING"
                        continue
                    job.state = "RUNNING"
                    job.jpeg = message.pop("jpeg", None)
                    job.data.update({k: v for k, v in message.items() if k != "kind"})
            if job.stop.is_set() and job.state != "COMPLETED":
                job.state = "CANCELLED"
        except Exception:
            with job.lock:
                job.state = "ERROR"
                job.error = "Falha ou limite de tempo do processamento local"
        finally:
            if process.pid is not None:
                process.join(timeout=1)
                if process.is_alive():
                    try:
                        if os.name == "posix" and os.getpgid(process.pid) == process.pid:
                            os.killpg(process.pid, signal.SIGTERM)
                        else:
                            process.terminate()
                    except ProcessLookupError:
                        pass
                    process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
            out.cancel_join_thread()
            out.close()
            (job.folder / "source.mp4").unlink(missing_ok=True)
            (job.folder / "source-image").unlink(missing_ok=True)
            (job.folder / "encoding.mp4").unlink(missing_ok=True)
            if job.state != "COMPLETED":
                for file in job.folder.iterdir():
                    if file.is_file():
                        file.unlink(missing_ok=True)
            with job.lock:
                job.ended = time.monotonic()

    def close(self) -> None:
        for job in list(self.jobs.values()):
            if job.state != "COMPLETED":
                job.stop.set()
        for job in list(self.jobs.values()):
            if job.thread:
                job.thread.join(timeout=5)
            elif job.state == "UPLOADING":
                self.discard_upload(job)
