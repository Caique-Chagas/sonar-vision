"""Download the public pilot weights with size and SHA-256 verification."""

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = (
    "https://github.com/Caique-Chagas/sonar-vision/releases/download/v0.1.0-academic/best.pt"
)
MODEL_SHA256 = "9de9b7abdf4e543fe6976d0461110d02656973d78edeb7684882d3b904aaccc9"
MAX_BYTES = 16 * 1024 * 1024


def download(destination: Path, *, expected: str = MODEL_SHA256, limit: int = MAX_BYTES) -> None:
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
            return
        raise ValueError("Destino já existe com outro conteúdo; preserve-o e use outro --output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        digest = hashlib.sha256()
        transferred = 0
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
            temporary = Path(file.name)
            with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
                while chunk := response.read(64 * 1024):
                    transferred += len(chunk)
                    if transferred > limit:
                        raise ValueError("Modelo excede o limite de transferência")
                    digest.update(chunk)
                    file.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("SHA-256 inválido; pesos não foram instalados")
        if destination.exists():
            raise ValueError("Destino criado durante o download; nenhum arquivo foi sobrescrito")
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "models/best.pt")
    args = parser.parse_args()
    download(args.output)
    print(f"Modelo verificado: {args.output}")
    print(f"SHA-256: {MODEL_SHA256}")


if __name__ == "__main__":
    main()
