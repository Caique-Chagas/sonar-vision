import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/download_model.py"
spec = importlib.util.spec_from_file_location("public_model_download", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_verified_download(tmp_path, monkeypatch):
    content = b"controlled-test-payload"
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content))
    destination = tmp_path / "weights.pt"
    module.download(destination, expected=hashlib.sha256(content).hexdigest())
    assert destination.read_bytes() == content


def test_bad_hash_never_installs(tmp_path, monkeypatch):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"changed"))
    with pytest.raises(ValueError, match="SHA-256"):
        module.download(tmp_path / "weights.pt", expected="0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_size_limit_cleans_partial_download(tmp_path, monkeypatch):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x" * 20))
    with pytest.raises(ValueError, match="limite"):
        module.download(tmp_path / "weights.pt", limit=10)
    assert list(tmp_path.iterdir()) == []


def test_existing_different_weights_preserved(tmp_path):
    destination = tmp_path / "weights.pt"
    destination.write_bytes(b"existing-user-weights")
    with pytest.raises(ValueError, match="Destino"):
        module.download(destination)
    assert destination.read_bytes() == b"existing-user-weights"
