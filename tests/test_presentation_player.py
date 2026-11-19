"""Clock, decode and controls tests; fixture clips are not ML evidence."""

import builtins

import cv2
import numpy as np
import pytest

from sonar_vision.desktop.player import Playback, PlaybackClock
from sonar_vision.desktop.presentation import ClipSpec, prepare_clip


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter.fourcc(*"mp4v"), 10, (160, 96))
    for i in range(16):
        writer.write(np.full((96, 160, 3), 40 + i * 5, np.uint8))
    writer.release()
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"SOFTWARE FIXTURE ONLY")

    class NoOCR:
        def read(self, roi):
            raise AssertionError("No ROI in this fixture")

    class Analyzer:
        ocr = NoOCR()

        def analyze(self, frame):
            return frame, dict(findings=[], state="NO_PANEL")

    out = tmp_path / "prepared"
    prepare_clip(
        ClipSpec("fixture", str(source), 0, 16, "FIXTURE"),
        out,
        analyzer_factory=Analyzer,
        model_files=dict(bus=model, sign=model),
    )
    return out


def test_pause_resume_and_restart_preserve_video_clock():
    clock = PlaybackClock(30, 10)
    assert clock.due_frame(11) == 30
    clock.pause(11)
    assert clock.due_frame(20) == 30
    clock.resume(20)
    assert clock.due_frame(21) == 60
    clock.restart(25)
    assert clock.due_frame(25) == 0
    for fps in [0, float("nan"), -1]:
        with pytest.raises(ValueError):
            PlaybackClock(fps, 0)


def test_decode_is_clock_paced_and_never_reexecutes_models(prepared, monkeypatch):
    original = builtins.__import__

    def forbid(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ultralytics", "paddleocr", "paddle"}:
            raise AssertionError("Model import attempted during prepared playback")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid)
    player = Playback(prepared, 0)
    try:
        assert player.advance(0) and player.index == 0
        assert not player.advance(0.05)
        assert player.advance(0.25) and player.index == 2
        assert player.skipped == 1
        player.timeline.pause(0.25)
        assert not player.advance(10) and player.index == 2
        player.timeline.resume(10)
        assert player.advance(10.1) and player.index == 3
        player.restart(20)
        assert player.frame is None and player.index == -1
        assert player.advance(20) and player.index == 0
        assert player.advance(22) and player.index == 15
        player.advance(22.1)
        assert player.eof
        assert player.report(22.1)["models_executed_during_playback"] is False
    finally:
        player.close()


def test_new_video_starts_without_old_frame_state(prepared):
    first = Playback(prepared, 0)
    first.advance(0.8)
    assert first.index == 8
    first.close()
    second = Playback(prepared, 10)
    try:
        assert second.frame is None and second.index == -1
        assert second.advance(10) and second.index == 0
    finally:
        second.close()


def test_missing_video_or_manifest_fails_explicitly(tmp_path):
    with pytest.raises((OSError, ValueError)):
        Playback(tmp_path / "missing", 0)


def test_versioned_directory_reuses_player_without_preparation(tmp_path, monkeypatch):
    from sonar_vision.desktop import player

    selected = tmp_path / "selected"
    calls = []
    monkeypatch.setattr(player, "ensure_prepared", lambda keys: calls.append(keys))
    monkeypatch.setattr(player, "event_pump", lambda: (lambda: -1, "fixture"))
    for name in ["namedWindow", "resizeWindow", "destroyWindow", "waitKey"]:
        monkeypatch.setattr(player.cv2, name, lambda *args: None)

    def opened(directory, now):
        assert directory == selected
        raise ValueError("fixture stops before GUI decoding")

    monkeypatch.setattr(player, "Playback", opened)
    with pytest.raises(ValueError, match="fixture stops"):
        player.run_presentation(
            "sao-paulo", prepared_directories={"sao-paulo": selected}, output_root=tmp_path
        )
    assert calls == [[]]


def test_unknown_versioned_example_rejected_before_gui(tmp_path):
    from sonar_vision.desktop.player import run_presentation

    with pytest.raises(ValueError, match="desconhecido"):
        run_presentation("sao-paulo", prepared_directories={"unknown": tmp_path})


def test_native_audio_failure_preserves_video_and_uses_fallback(prepared, monkeypatch):
    import json
    from unittest.mock import Mock

    playback = Playback(prepared, 0)
    try:
        (prepared / "cue.wav").write_bytes(b"fixture only")
        (prepared / "audio-events.json").write_text(
            json.dumps(
                [dict(timestamp_seconds=0.5, audio_file="cue.wav", code="637P-10", track_id=1)]
            )
        )
        playback.manifest["audio_events_file"] = "audio-events.json"
        native = Mock()
        native.prepare.side_effect = OSError("fixture audio unavailable")
        monkeypatch.setattr("sonar_vision.desktop.native_audio.NativeAudio", lambda lib: native)
        playback.configure_audio(True, object())
        assert playback.audio.native is None
        assert playback.audio.log[0]["result"] == "native_prepare_failed_using_afplay"
        assert playback.advance(0)
    finally:
        playback.close()
