"""Offline macOS accessibility cues; not academic narration or line verification."""

import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from sonar_vision.desktop.evidence import COMPLETE_CODE
from sonar_vision.training.dataset import sha256


def voice_available() -> bool:
    if not Path("/usr/bin/say").is_file():
        return False
    result = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True, timeout=10)
    return result.returncode == 0 and any(
        line.startswith("Luciana ") and "pt_BR" in line for line in result.stdout.splitlines()
    )


def inspect_wave(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as stream:
        if stream.getsampwidth() != 2 or stream.getnchannels() != 1:
            raise ValueError("Expected PCM16 mono accessibility cue")
        count, rate = stream.getnframes(), stream.getframerate()
        samples = np.frombuffer(stream.readframes(count), dtype="<i2").astype(float) / 32768
    duration = count / rate
    rms = float(np.sqrt(np.mean(samples**2))) if count else 0.0
    peak = float(np.max(np.abs(samples))) if count else 0.0
    if not 0.2 <= duration <= 15 or rms < 0.001 or peak >= 0.9999:
        raise ValueError("Silent, clipped or invalid accessibility cue")
    return dict(
        duration_seconds=duration,
        sample_rate=rate,
        frames=count,
        rms=rms,
        peak=peak,
        sample_format="PCM16mono",
    )


def synthesize_code(code: str, directory: Path) -> dict[str, Any]:
    if not COMPLETE_CODE.fullmatch(code):
        raise ValueError("Only complete, stable code may be synthesized")
    binding: dict[str, Any] = dict(
        code=code,
        voice="Luciana",
        locale="pt_BR",
        rate=175,
        text=f"Código de linha {code} identificado.",
        purpose="accessibility_not_narration",
    )
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / (code + ".json")
    if record.exists():
        prior: dict[str, Any] = json.loads(record.read_text())
        if prior["binding"] != binding or sha256(directory / prior["file"]) != prior["sha256"]:
            raise ValueError("Existing voice cue changed; preserve and use another directory")
        return prior
    if not voice_available():
        raise RuntimeError("Portuguese offline voice Luciana unavailable")
    import imageio_ffmpeg

    work = directory / (code + ".working")
    work.mkdir(exist_ok=False)
    try:
        aiff, wav = work / "voice.aiff", work / "voice.wav"
        subprocess.run(
            ["/usr/bin/say", "-v", "Luciana", "-r", "175", "-o", str(aiff), binding["text"]],
            check=True,
            timeout=30,
            capture_output=True,
        )
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(aiff),
                "-ar",
                "22050",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(wav),
            ],
            check=True,
            timeout=30,
            capture_output=True,
        )
        metadata = inspect_wave(wav)
        destination = directory / (code + ".wav")
        if destination.exists():
            raise ValueError("Unsealed cue exists; refusing overwrite")
        shutil.copy2(wav, destination)
        shutil.copy2(aiff, directory / (code + ".aiff"))
        result = dict(
            binding=binding, file=destination.name, sha256=sha256(destination), **metadata
        )
        record.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return result
    finally:
        shutil.rmtree(work)
