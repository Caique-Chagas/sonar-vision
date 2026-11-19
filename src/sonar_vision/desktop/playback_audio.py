"""Nonblocking playback of prepared accessibility cues using the video clock."""

import signal
import subprocess
from pathlib import Path
from typing import Any


class PlaybackAudio:
    def __init__(self, directory: Path, cues: list[dict[str, Any]], enabled: bool):
        self.directory, self.cues, self.enabled = directory, cues, enabled
        self.played: set[int] = set()
        self.process: Any = None
        self.native: Any = None
        self.paused = False
        self.active_event: dict[str, Any] | None = None
        self.log: list[dict[str, Any]] = []

    def tick(self, elapsed: float, paused: bool) -> None:
        if self.process and self.process.poll() is not None:
            if self.active_event is not None:
                self.active_event["process_returncode"] = self.process.returncode
                self.active_event["completed_video_seconds"] = elapsed
            self.process = None
            self.active_event = None
        if self.process and paused != self.paused:
            self.process.send_signal(signal.SIGSTOP if paused else signal.SIGCONT)
        self.paused = paused
        if paused:
            return
        for index, cue in enumerate(self.cues):
            timestamp = cue["timestamp_seconds"]
            if index in self.played or elapsed < timestamp:
                continue
            self.played.add(index)
            reason = None
            if not self.enabled:
                reason = "muted"
            elif elapsed - timestamp > 0.25:
                reason = "late_event_skipped"
            elif self.process:
                reason = "previous_cue_busy"
            path = self.directory / cue["audio_file"]
            if not reason and (
                (not self.native and not Path("/usr/bin/afplay").is_file()) or not path.is_file()
            ):
                reason = "audio_unavailable"
            event = dict(
                timestamp_seconds=timestamp,
                actual_video_seconds=elapsed,
                delta_seconds=elapsed - timestamp,
                code=cue["code"],
                track_id=cue["track_id"],
                result=reason or "started",
                backend="AVFoundation_preloaded" if self.native else "afplay",
            )
            if not reason:
                try:
                    self.process = (
                        self.native.start(path)
                        if self.native
                        else subprocess.Popen(
                            ["/usr/bin/afplay", str(path)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    )
                except OSError:
                    event["result"] = "audio_start_failed"
            self.log.append(event)
            if self.process and not reason:
                self.active_event = event

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            if self.paused:
                self.process.send_signal(signal.SIGCONT)
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.active_event is not None:
            self.active_event["stopped_by_control"] = True
        self.active_event = None
        self.process = None
        self.paused = False

    def mute(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.stop()

    def restart(self) -> None:
        self.stop()
        self.played.clear()
