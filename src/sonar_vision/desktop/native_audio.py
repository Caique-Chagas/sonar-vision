"""Preloaded AVFoundation cue using the existing Cocoa helper, on the main thread."""

import ctypes
import signal
from pathlib import Path
from typing import Any


class NativeAudio:
    def __init__(self, library: Any):
        self.library = library
        library.sonar_audio_prepare.argtypes = [ctypes.c_char_p]
        library.sonar_audio_prepare.restype = ctypes.c_int32
        for name in ["sonar_audio_play", "sonar_audio_playing"]:
            getattr(library, name).restype = ctypes.c_int32
        self.prepared: Path | None = None

    def prepare(self, path: Path) -> None:
        if self.library.sonar_audio_prepare(str(path.resolve()).encode()) != 1:
            raise OSError("AVFoundation could not prepare the accessibility cue")
        self.prepared = path

    def start(self, path: Path) -> Any:
        if self.prepared != path:
            self.prepare(path)
        if self.library.sonar_audio_play() != 1:
            raise OSError("AVFoundation playback failed")
        return NativePlayback(self.library)


class NativePlayback:
    def __init__(self, library: Any):
        self.library, self.paused = library, False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is None and not self.paused and not self.library.sonar_audio_playing():
            self.returncode = 0
        return self.returncode

    def send_signal(self, value: int) -> None:
        if value == signal.SIGSTOP:
            self.library.sonar_audio_pause()
            self.paused = True
        elif value == signal.SIGCONT:
            self.library.sonar_audio_play()
            self.paused = False

    def terminate(self) -> None:
        self.library.sonar_audio_stop()
        self.returncode = 0

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0
