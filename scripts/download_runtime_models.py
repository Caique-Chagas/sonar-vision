"""Install the two essential release models; reject altered/existing files."""

import argparse
import hashlib
import json
import ssl
import tempfile
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]


def download_model(url: str, destination: Path, expected: str, size: int) -> None:
    if not url.startswith("https://github.com/Caique-Chagas/sonar-vision/releases/download/"):
        raise ValueError("Only the official versioned release is accepted")
    if not 0 < size <= 16 * 1024 * 1024:
        raise ValueError("Invalid model size")
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
            return
        raise ValueError("Existing model differs; preserve it and select another destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        count = 0
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(
                url, timeout=30, context=ssl.create_default_context(cafile=certifi.where())
            ) as response:
                while block := response.read(64 * 1024):
                    count += len(block)
                    if count > size:
                        raise ValueError("Unexpected model size")
                    digest.update(block)
                    stream.write(block)
        if count != size or digest.hexdigest() != expected:
            raise ValueError("Model size/checksum mismatch")
        if destination.exists():
            raise ValueError("Destination appeared during download; refusing overwrite")
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Preserve documented relative paths"
    )
    args = parser.parse_args()
    records = json.loads((ROOT / "reports/runtime-models.json").read_text())
    for record in records:
        destination = args.root / record["destination"]
        download_model(record["url"], destination, record["sha256"], record["size"])
        print(record["id"], "SHA256 verified", destination)


if __name__ == "__main__":
    main()
