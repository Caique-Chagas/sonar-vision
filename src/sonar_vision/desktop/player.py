"""Clock-paced desktop playback. No YOLO, OCR, model factory or inference imports."""

import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import cv2

from sonar_vision.desktop.playback_audio import PlaybackAudio
from sonar_vision.desktop.presentation import ROOT, SPECS, verify_prepared
from sonar_vision.processing.roi import Frame

TITLE = "SONAR VISION — APRESENTAÇÃO — INFERÊNCIA PRÉ-PROCESSADA"
ORDER = ["santos-a", "santos-b", "sao-paulo"]


def event_pump() -> tuple[Any, str]:
    source = Path("native/CocoaEventPump.swift")
    if (
        sys.platform == "darwin"
        and source.is_file()
        and os.environ.get("SONAR_EVENT_BACKEND") != "opencv"
    ):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        directory = ROOT / "native"
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / ("cocoa-events-" + digest + ".dylib")
        if not binary.is_file():
            result = subprocess.run(
                [
                    "xcrun",
                    "swiftc",
                    "-swift-version",
                    "5",
                    "-emit-library",
                    str(source),
                    "-o",
                    str(binary),
                    "-framework",
                    "AppKit",
                    "-framework",
                    "AVFoundation",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            (directory / "compile.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                return lambda: cv2.waitKey(1), "opencv-waitKey-fallback"
        library = ctypes.CDLL(str(binary.resolve()))
        poll = library.sonar_poll_key
        poll.restype = ctypes.c_int32
        cast(Any, poll).native_library = library
        # The ctypes function retains its library handle for the player lifetime.
        return poll, "cocoa-native-nonblocking"
    return lambda: cv2.waitKey(1), "opencv-waitKey"


class PlaybackClock:
    def __init__(self, fps: float, now: float):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS inválido")
        self.fps = fps
        self.origin = now
        self.paused_at: float | None = None

    def elapsed(self, now: float) -> float:
        return max(0.0, (self.paused_at if self.paused_at is not None else now) - self.origin)

    def due_frame(self, now: float) -> int:
        return int(self.elapsed(now) * self.fps)

    def pause(self, now: float) -> None:
        if self.paused_at is None:
            self.paused_at = now

    def resume(self, now: float) -> None:
        if self.paused_at is not None:
            self.origin += now - self.paused_at
            self.paused_at = None

    def restart(self, now: float) -> None:
        self.origin = now
        self.paused_at = None


class Playback:
    def __init__(self, directory: Path, now: float):
        self.directory = directory
        self.manifest = verify_prepared(directory)
        self.fps = float(self.manifest["annotated"]["fps"])
        self.count = int(self.manifest["annotated"]["frames"])
        self.cap = cv2.VideoCapture(str(directory / "annotated.mp4"), cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise ValueError("MP4 preparado indisponível")
        self.timeline = PlaybackClock(self.fps, now)
        self.index = -1
        self.frame: Frame | None = None
        self.eof = False
        self.decoded = 0
        self.displayed = 0
        self.skipped = 0
        self.first_display: float | None = None
        self.last_display: float | None = None
        self.audio = PlaybackAudio(directory, [], False)

    def configure_audio(self, enabled: bool, native_library: Any = None) -> None:
        name = self.manifest.get("audio_events_file")
        cues = json.loads((self.directory / name).read_text()) if name else []
        for cue in cues:
            path = self.directory / cue["audio_file"]
            timestamp = cue["timestamp_seconds"]
            if (
                not path.resolve().is_relative_to(self.directory.resolve())
                or path.suffix != ".wav"
                or not math.isfinite(timestamp)
                or not 0 <= timestamp < self.count / self.fps
            ):
                raise ValueError("Evento de áudio inválido")
        self.audio = PlaybackAudio(self.directory, cues, enabled)
        if native_library is not None and cues:
            from sonar_vision.desktop.native_audio import NativeAudio

            try:
                native = NativeAudio(native_library)
                native.prepare(self.directory / cues[0]["audio_file"])
                self.audio.native = native
            except (OSError, AttributeError):
                self.audio.log.append(dict(result="native_prepare_failed_using_afplay"))

    def advance(self, now: float) -> bool:
        self.audio.tick(self.timeline.elapsed(now), self.timeline.paused_at is not None)
        if self.eof or self.timeline.paused_at is not None:
            return False
        due = min(self.count - 1, self.timeline.due_frame(now))
        old = self.index
        while self.index < due:
            ok, frame = self.cap.read()
            if not ok:
                raise ValueError(f"Falha de decodificação no frame{self.index + 1}")
            self.index += 1
            self.decoded += 1
            self.frame = cast(Frame, frame)
        if self.index != old:
            self.skipped += max(0, self.index - old - 1)
            return True
        if self.timeline.elapsed(now) >= self.count / self.fps:
            self.eof = True
            self.timeline.pause(now)
        return False

    def displayed_at(self, now: float) -> None:
        elapsed = self.timeline.elapsed(now)
        if self.first_display is None:
            self.first_display = elapsed
        self.last_display = elapsed
        self.displayed += 1

    def report(self, now: float) -> dict[str, Any]:
        interval = (self.last_display or 0) - (self.first_display or 0)
        return dict(
            clip=self.manifest["spec"]["id"],
            source_fps=self.fps,
            duration_seconds=self.count / self.fps,
            decoded_frames=self.decoded,
            displayed_frames=self.displayed,
            skipped_display_frames=self.skipped,
            measured_display_fps=(self.displayed - 1) / interval if interval > 0 else None,
            active_playback_wall_seconds=self.timeline.elapsed(now),
            EOF=self.eof,
            models_loaded_in_player=[
                n for n in ["torch", "ultralytics", "paddleocr", "paddle"] if n in sys.modules
            ],
            models_executed_during_playback=False,
            audio_enabled=self.audio.enabled,
            accessibility_audio_events=list(self.audio.log),
        )

    def restart(self, now: float) -> None:
        self.audio.restart()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.timeline.restart(now)
        self.index = -1
        self.frame = None
        self.eof = False
        self.decoded = 0
        self.displayed = 0
        self.skipped = 0
        self.first_display = None
        self.last_display = None

    def close(self) -> None:
        self.audio.stop()
        self.cap.release()


def ensure_prepared(keys: list[str]) -> None:
    missing = [key for key in keys if not (ROOT / key / "manifest.json").exists()]
    for key in missing:
        print("APRESENTAÇÃO: preparando inferência real antes da reprodução:", key, flush=True)
        subprocess.run(
            [sys.executable, "scripts/prepare_final_presentation.py", "--example", key], check=True
        )


def run_presentation(
    city: str,
    *,
    smoke: bool = False,
    test_controls: bool = False,
    prepared_directories: dict[str, Path] | None = None,
    output_root: Path | None = None,
    audio: bool = False,
) -> None:
    requested = "santos-a" if city in ["best", "santos-yolo"] else city
    keys = ORDER[:] if city == "all" else [requested]
    if any(k not in SPECS for k in keys):
        raise ValueError("Exemplo de apresentação desconhecido")
    overrides = prepared_directories or {}
    if any(key not in SPECS for key in overrides):
        raise ValueError("Exemplo de apresentação desconhecido")
    directories = {key: overrides.get(key, ROOT / key) for key in ORDER}
    ensure_prepared([key for key in keys if key not in overrides])
    run = (output_root or ROOT) / ("playback-" + uuid4().hex[:10])
    run.mkdir(parents=True)
    events = (run / "events.jsonl").open("w")
    reports = []
    playback = None
    position = 0
    started = time.perf_counter()
    schedule = [
        (0.6, 32),
        (1.0, 32),
        (2.0, ord("m")),
        (7.5, ord("m")),
        (8.0, ord("r")),
        (15.2, 32),
        (16.0, 32),
        (17.0, ord("1")),
        (17.7, ord("2")),
        (18.4, ord("3")),
        (19.1, ord("n")),
        (19.8, ord("s")),
        (20.6, ord("q")),
    ]

    def event(action: str, **fields: Any) -> None:
        events.write(
            json.dumps(dict(action=action, wall_seconds=time.perf_counter() - started, **fields))
            + "\n"
        )
        events.flush()

    pump, backend = event_pump()
    pump_seconds = 0.0
    decode_seconds = 0.0
    display_seconds = 0.0
    last_pump = 0.0
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, 1280, 720)
    try:
        while True:
            if playback is None:
                clip_key = keys[position]
                playback = Playback(directories[clip_key], time.perf_counter())
                playback.configure_audio(audio, getattr(pump, "native_library", None))
                cv2.setWindowTitle(TITLE, TITLE + (" | ÁUDIO ATIVO" if audio else " | MUDO"))
                playback.timeline.restart(time.perf_counter())
                event(
                    "opened",
                    clip=clip_key,
                    window_visible=cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE),
                )
            now = time.perf_counter()
            before = time.perf_counter()
            changed = playback.advance(now)
            decode_seconds += time.perf_counter() - before
            if changed and playback.frame is not None:
                before = time.perf_counter()
                cv2.imshow(TITLE, playback.frame)
                display_seconds += time.perf_counter() - before
                playback.displayed_at(time.perf_counter())
            before = time.perf_counter()
            pumped = changed or before - last_pump >= 1 / 60
            key = int(pump()) & 255 if pumped else 255
            if pumped:
                last_pump = before
            pump_seconds += time.perf_counter() - before
            if backend.startswith("cocoa-native"):
                time.sleep(0.001)
            if 65 <= key <= 90:
                key += 32
            if test_controls and schedule and time.perf_counter() - started >= schedule[0][0]:
                _, key = schedule.pop(0)
                event("scripted_control", key=key)
            if key in [27, ord("q")] or (
                pumped and cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1
            ):
                event("quit")
                break
            switch = None
            if key == 32:
                if playback.timeline.paused_at is None:
                    playback.timeline.pause(time.perf_counter())
                    event("pause", frame=playback.index)
                elif not playback.eof:
                    playback.timeline.resume(time.perf_counter())
                    event("resume", frame=playback.index)
            elif key == ord("r"):
                reports.append(playback.report(time.perf_counter()))
                playback.restart(time.perf_counter())
                event("restart")
            elif key == ord("m"):
                audio = not playback.audio.enabled
                playback.audio.mute(audio)
                cv2.setWindowTitle(TITLE, TITLE + (" | ÁUDIO ATIVO" if audio else " | MUDO"))
                event("audio_toggled", enabled=audio)
            elif key == ord("s") and playback.frame is not None:
                cv2.imwrite(
                    str(run / f"frame-{len(list(run.glob('frame-*.png'))):03d}.png"), playback.frame
                )
                event("saved", frame=playback.index)
            elif key == ord("n"):
                switch = ORDER[(ORDER.index(keys[position]) + 1) % len(ORDER)]
            elif key in [ord("1"), ord("2"), ord("3")]:
                switch = {ord("1"): "sao-paulo", ord("2"): "santos-a", ord("3"): "santos-b"}[key]
            if playback.eof and smoke and not test_controls:
                if position == len(keys) - 1:
                    break
                switch = keys[position + 1]
            elif playback.eof and city == "all" and not test_controls:
                if position < len(keys) - 1:
                    switch = keys[position + 1]
            if switch and not (directories[switch] / "manifest.json").is_file():
                event("example_not_prepared", clip=switch)
                print("Exemplo ainda não preparado:", switch, flush=True)
                switch = None
            if switch:
                reports.append(playback.report(time.perf_counter()))
                playback.close()
                playback = None
                if switch not in keys:
                    keys.append(switch)
                position = keys.index(switch)
                event("switched", clip=switch)
    finally:
        if playback is not None:
            reports.append(playback.report(time.perf_counter()))
            playback.close()
        cv2.destroyWindow(TITLE)
        cv2.waitKey(1)
        events.close()
        (run / "report.json").write_text(
            json.dumps(
                dict(
                    mode="presentation",
                    reports=reports,
                    tests_scripted=test_controls,
                    event_backend=backend,
                    event_pump_seconds=pump_seconds,
                    decode_seconds=decode_seconds,
                    display_seconds=display_seconds,
                ),
                indent=2,
            )
            + "\n"
        )
        print("Presentation evidence:", run, flush=True)
