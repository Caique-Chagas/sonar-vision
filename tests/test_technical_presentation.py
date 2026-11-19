"""Software regression fixtures; these are not real-model accuracy measurements."""

import copy
import signal
import wave

import numpy as np
import pytest

from sonar_vision.desktop.evidence import PresentationEvidence
from sonar_vision.desktop.label_layout import LabelLayout, intersects
from sonar_vision.desktop.playback_audio import PlaybackAudio
from sonar_vision.speech.offline import inspect_wave, synthesize_code


def record(index, code="637P-10", digest=None):
    return dict(
        source_index=index,
        timestamp_seconds=index * 0.2,
        input_sha256=digest or f"{index + 1:064x}",
        row=dict(
            buses=[dict(box=dict(x1=20, y1=20, x2=120, y2=180), confidence=0.9, label="bus")],
            findings=[
                dict(
                    box=dict(x1=30, y1=30, x2=100, y2=50),
                    detector_confidence=0.8,
                    readings=[dict(text=code or "TERMINAL", confidence=0.9)],
                    code=code,
                    ocr_error=None,
                )
            ],
        ),
    )


def stable():
    evidence = PresentationEvidence()
    results = [evidence.observe(record(i)) for i in range(5)]
    return evidence, results


def test_real_distinct_evidence_reaches_code_stable_once_without_line_verification():
    _, results = stable()
    assert results[0]["state"] == "OCR_CANDIDATE"
    assert results[1]["state"] == "OCR_CANDIDATE"
    assert results[2]["state"] == "CODE_STABLE"
    assert results[2]["audio_event"]["timestamp_seconds"] == 0.4
    assert sum(r["audio_event"] is not None for r in results) == 1
    assert all(r["line_verified"] is False for r in results)
    assert results[2]["audio_event"]["text"] == "Código de linha 637P-10 identificado."


def test_duplicate_pixels_never_form_consensus():
    evidence = PresentationEvidence()
    results = [evidence.observe(record(i, digest="a" * 64)) for i in range(5)]
    assert all(r["state"] != "CODE_STABLE" and not r["audio_event"] for r in results)


def test_raw_ocr_and_predictions_unchanged():
    original = record(0)
    before = copy.deepcopy(original)
    PresentationEvidence().observe(original)
    assert original == before


@pytest.mark.parametrize("origin", ["tracking", "interpolated", "ground_truth"])
def test_non_inference_boxes_never_create_audio(origin):
    evidence = PresentationEvidence()
    for i in range(5):
        source = record(i)
        source["row"]["findings"][0]["detection_origin"] = origin
        assert not evidence.observe(source)["audio_event"]


def test_contradiction_clears_stable_state_before_next_sampling_instant():
    evidence = PresentationEvidence()
    for i in range(3):
        evidence.observe(record(i))
    source = record(3, "477A-10")
    source["timestamp_seconds"] = 0.41
    result = evidence.observe(source)
    assert result["state"] == "OCR_CANDIDATE" and not result["stable_codes"]


def test_ambiguous_vehicle_association_never_votes():
    evidence = PresentationEvidence()
    for i in range(4):
        source = record(i)
        source["row"]["buses"] *= 2
        assert not evidence.observe(source)["audio_event"]


def test_missing_panel_and_low_confidence_never_announce():
    evidence = PresentationEvidence()
    for i in range(4):
        source = record(i)
        source["row"]["findings"][0]["readings"][0]["confidence"] = 0.1
        assert not evidence.observe(source)["audio_event"]
    source = record(4)
    source["row"]["findings"] = []
    assert evidence.observe(source)["state"] == "NO_DETECTION"


def test_abbreviated_code_has_no_announcement():
    evidence = PresentationEvidence()
    for i in range(4):
        source = record(i, None)
        source["row"]["findings"][0]["readings"][0]["text"] = "155"
        assert evidence.observe(source)["state"] == "PANEL_DETECTED"


@pytest.mark.parametrize(
    "change",
    [{"source_index": -1}, {"timestamp_seconds": float("nan")}, {"input_sha256": "invalid"}],
)
def test_invalid_identity_is_rejected(change):
    source = record(0)
    source.update(change)
    with pytest.raises(ValueError):
        PresentationEvidence().observe(source)


def test_label_placement_preserves_panel_and_other_labels():
    panel = (10.0, 40.0, 210.0, 80.0)
    layout = LabelLayout([panel])
    first = layout.allocate(panel, 250, 60)
    second = layout.allocate(panel, 250, 60)
    assert first and second
    assert not intersects(first, panel) and not intersects(second, panel)
    assert not intersects(first, second)
    assert layout.allocate(panel, 2000, 60) is None


class Process:
    def __init__(self):
        self.signals = []
        self.done = False

    def poll(self):
        return 0 if self.done else None

    def send_signal(self, signal):
        self.signals.append(signal)

    def terminate(self):
        self.done = True

    def wait(self, timeout):
        return 0


def audio(tmp_path, monkeypatch):
    (tmp_path / "cue.wav").write_bytes(b"fixture")
    process = Process()
    monkeypatch.setattr(
        "sonar_vision.desktop.playback_audio.subprocess.Popen", lambda *a, **kw: process
    )
    monkeypatch.setattr("sonar_vision.desktop.playback_audio.Path.is_file", lambda p: True)
    result = PlaybackAudio(
        tmp_path,
        [dict(timestamp_seconds=1.0, audio_file="cue.wav", code="637P-10", track_id=1)],
        True,
    )
    return result, process


def test_audio_clock_dedup_pause_restart_and_stop(tmp_path, monkeypatch):
    player, process = audio(tmp_path, monkeypatch)
    player.tick(0.9, False)
    assert not player.log
    player.tick(1.01, False)
    assert player.log[0]["result"] == "started"
    player.tick(1.1, True)
    assert process.signals[-1] == signal.SIGSTOP
    player.tick(1.1, False)
    assert process.signals[-1] == signal.SIGCONT
    player.tick(1.2, False)
    assert len(player.log) == 1
    player.restart()
    assert process.done and not player.played


def test_muted_and_late_events_are_not_replayed(tmp_path, monkeypatch):
    player, _ = audio(tmp_path, monkeypatch)
    player.mute(False)
    player.tick(1.0, False)
    assert player.log[0]["result"] == "muted"
    player.mute(True)
    player.tick(1.1, False)
    assert len(player.log) == 1
    player.restart()
    player.tick(2.0, False)
    assert player.log[-1]["result"] == "late_event_skipped"


def test_no_sound_from_empty_cues(tmp_path):
    player = PlaybackAudio(tmp_path, [], True)
    player.tick(10.0, False)
    assert not player.log and player.process is None


@pytest.mark.parametrize("code", ["155", "637P-1O", "--help", "637P-10 confirmado"])
def test_synthesis_rejects_uncertain_or_unbounded_text(code, tmp_path):
    with pytest.raises(ValueError):
        synthesize_code(code, tmp_path)


def test_wave_audit_rejects_silence_and_accepts_bounded_signal(tmp_path):
    path = tmp_path / "cue.wav"

    def write(samples):
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(22050)
            stream.writeframes(samples.astype("<i2").tobytes())

    write(np.zeros(22050))
    with pytest.raises(ValueError):
        inspect_wave(path)
    write(np.sin(np.arange(22050) * 0.1) * 1000)
    assert inspect_wave(path)["duration_seconds"] == 1


def test_native_preloaded_audio_controls_without_subprocess():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sonar_vision.desktop.native_audio import NativeAudio

    library = SimpleNamespace(
        **{
            name: Mock(return_value=1)
            for name in [
                "sonar_audio_prepare",
                "sonar_audio_play",
                "sonar_audio_playing",
                "sonar_audio_pause",
                "sonar_audio_stop",
            ]
        }
    )
    native = NativeAudio(library)
    path = __import__("pathlib").Path("fixture.wav")
    native.prepare(path)
    player = native.start(path)
    library.sonar_audio_prepare.assert_called_once()
    assert player.poll() is None
    player.send_signal(signal.SIGSTOP)
    assert player.poll() is None
    library.sonar_audio_pause.assert_called_once()
    player.send_signal(signal.SIGCONT)
    library.sonar_audio_playing.return_value = 0
    assert player.poll() == 0
    player.terminate()
    library.sonar_audio_stop.assert_called_once()


def test_final_catalog_requires_all_versions_and_matching_hashes(tmp_path, monkeypatch):
    import json

    from sonar_vision.desktop import catalog
    from sonar_vision.training.dataset import sha256

    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    assert catalog.final_directories() is None
    manifests = {}
    for key in ["sao-paulo", "santos-a", "santos-b"]:
        path = tmp_path / "presentation" / key / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        manifests[key] = sha256(path)
    active = tmp_path / "active.json"
    active.write_text(
        json.dumps(
            dict(status="TECHNICALLY_VALIDATED_PRIVATE", line_verified=False, manifests=manifests)
        )
    )
    assert set(catalog.final_directories()) == set(manifests)
    (tmp_path / "presentation/sao-paulo/manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        catalog.final_directories()


def test_completed_audio_records_backend_return_code(tmp_path, monkeypatch):
    player, process = audio(tmp_path, monkeypatch)
    player.tick(1.0, False)
    process.done = True
    process.returncode = 0
    player.tick(1.2, False)
    assert player.log[0]["process_returncode"] == 0


@pytest.mark.parametrize(
    "arguments,expected,audio_enabled",
    [
        ([], "sao-paulo", True),
        (["sao-paulo"], "sao-paulo", True),
        (["santos-a", "--mute"], "santos-a", False),
        (["santos-b"], "santos-b", True),
        (["all"], "all", True),
    ],
)
def test_cli_selects_validated_catalog_and_keeps_all_commands(
    arguments, expected, audio_enabled, tmp_path, monkeypatch
):
    import sys

    from sonar_vision.desktop import app, catalog, player

    directories = {k: tmp_path / k for k in ["sao-paulo", "santos-a", "santos-b"]}
    monkeypatch.setattr(catalog, "final_directories", lambda: directories)
    calls = []
    monkeypatch.setattr(
        player, "run_presentation", lambda city, **kwargs: calls.append((city, kwargs))
    )
    monkeypatch.setattr(sys, "argv", ["sonar", *arguments])
    app.main()
    assert calls[0][0] == expected
    assert calls[0][1]["prepared_directories"] == directories
    assert calls[0][1]["audio"] is audio_enabled
