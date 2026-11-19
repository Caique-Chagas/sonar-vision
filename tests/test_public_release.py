import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "runtime_download", Path("scripts/download_runtime_models.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
URL = "https://github.com/Caique-Chagas/sonar-vision/releases/download/v0.2.0-fiap/fixture.pt"


def test_verified_download_and_existing_file_preservation(tmp_path, monkeypatch):
    payload = b"SOFTWARE FIXTURE,NOT WEIGHTS"
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    path = tmp_path / "model.pt"
    digest = hashlib.sha256(payload).hexdigest()
    module.download_model(URL, path, digest, len(payload))
    assert path.read_bytes() == payload
    module.download_model(URL, path, digest, len(payload))
    with pytest.raises(ValueError, match="Existing"):
        module.download_model(URL, path, "0" * 64, len(payload))
    assert path.read_bytes() == payload


def test_wrong_checksum_leaves_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"fake"))
    path = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="checksum"):
        module.download_model(URL, path, "0" * 64, 4)
    assert not path.exists() and not list(tmp_path.iterdir())


def test_oversized_transfer_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"toolong"))
    with pytest.raises(ValueError, match="size"):
        module.download_model(URL, tmp_path / "model.pt", "0" * 64, 2)
    assert not list(tmp_path.iterdir())


def test_untrusted_model_location_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="official"):
        module.download_model("https://example.org/model.pt", tmp_path / "model.pt", "0" * 64, 4)


def test_published_counts_are_recomputed_from_actual_evidence():
    import json

    expected = {
        "sao-paulo": (360, 271, 55, 21, 1),
        "santos-a": (600, 418, 0, 0, 0),
        "santos-b": (600, 452, 0, 0, 0),
    }
    for key, wanted in expected.items():
        rows = [
            json.loads(s)
            for s in (Path("reports/video-evidence") / (key + ".jsonl")).read_text().splitlines()
        ]
        assert (
            len(rows),
            sum(bool(r["findings"]) for r in rows),
            sum(any(f["code"] for f in r["findings"]) for r in rows),
            sum(r["presentation_state"] == "CODE_STABLE" for r in rows),
            sum(r["audio_event"] is not None for r in rows),
        ) == wanted
        assert all(not r["line_verified"] for r in rows)


def test_desktop_launcher_is_part_of_the_delivery():
    launcher = Path("scripts/demo_final.sh")
    assert launcher.is_file() and launcher.stat().st_mode & 0o100


def test_photo_command_preserves_source_and_produces_a_report(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    import cv2
    import numpy as np

    spec = importlib.util.spec_from_file_location("photo_command", Path("scripts/analyze_photo.py"))
    command = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(command)
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), np.zeros((64, 96, 3), dtype=np.uint8))
    weights = tmp_path / "fixture.pt"
    weights.write_bytes(b"SOFTWARE FIXTURE ONLY")
    target = tmp_path / "output"
    original = source.read_bytes()

    class Analyzer:
        static = True
        detector = SimpleNamespace(model_version="software-fixture")

        def analyze(self, image):
            return image.copy(), dict(findings=[], state="UNCERTAIN")

    monkeypatch.setattr(command, "create_demo_analyzer", lambda *a, **kw: Analyzer())
    monkeypatch.setattr(
        sys,
        "argv",
        ["photo", "--image", str(source), "--weights", str(weights), "--output", str(target)],
    )
    command.main()
    report = json.loads((target / "report.json").read_text())
    assert report["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert (target / "source-image").read_bytes() == original == source.read_bytes()
    with pytest.raises(SystemExit):
        command.main()
