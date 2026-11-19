"""Select only sealed, technically validated private presentation artifacts."""

import json
from pathlib import Path

from sonar_vision.training.dataset import sha256

ROOT = Path("artifacts/demo/sonar-final-technical")


def final_directories() -> dict[str, Path] | None:
    active = ROOT / "active.json"
    if not active.exists():
        return None
    data = json.loads(active.read_text())
    if (
        data.get("status") != "TECHNICALLY_VALIDATED_PRIVATE"
        or data.get("line_verified") is not False
    ):
        raise ValueError("Final presentation not technically validated")
    if set(data["manifests"]) != {"sao-paulo", "santos-a", "santos-b"}:
        raise ValueError("Final catalog must preserve all three examples")
    directories = {}
    for key, digest in data["manifests"].items():
        directory = ROOT / "presentation" / key
        if sha256(directory / "manifest.json") != digest:
            raise ValueError("Final catalog manifest changed")
        directories[key] = directory
    return directories
