"""Verify published file hashes; excludes runtime data and downloaded weights."""

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports/SHA256SUMS"
FOLDERS = {
    "src",
    "scripts",
    "tests",
    "configs",
    "notebooks",
    "docs",
    "reports",
    "artifacts",
    ".github",
}
TOP = {"README.md", "LICENSE", "pyproject.toml", ".env.example", ".gitignore", ".gitattributes"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        candidates = [ROOT / name for name in TOP]
        for folder in sorted(FOLDERS):
            candidates.extend((ROOT / folder).rglob("*"))
        files = [
            p
            for p in sorted(candidates)
            if p.is_file()
            and not p.is_symlink()
            and "__pycache__" not in p.parts
            and not any(part.endswith(".egg-info") for part in p.parts)
            and p.suffix != ".pyc"
            and p != MANIFEST
            and p.name != ".DS_Store"
            and not p.name.startswith("._")
            and not p.relative_to(ROOT)
            .as_posix()
            .startswith(("artifacts/source/", "artifacts/demo/build/"))
        ]
        MANIFEST.write_text(
            "".join(f"{digest(file)}  {file.relative_to(ROOT).as_posix()}\n" for file in files)
        )
        print(f"Manifesto: {len(files)} arquivos públicos")
    else:
        count = 0
        for line in MANIFEST.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            file = (ROOT / relative).resolve(strict=True)
            if not file.is_relative_to(ROOT) or digest(file) != expected:
                raise SystemExit(f"Integridade divergente: {relative}")
            count += 1
        print(f"Integridade verificada: {count} arquivos públicos")


if __name__ == "__main__":
    main()
