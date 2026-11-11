"""Create a reviewable source archive for Drive/Colab, without secrets, weights or data."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"src", "scripts", "tests", "configs", "notebooks", "docs", ".github"}
TOP = {
    "README.md",
    "LICENSE",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    ".gitattributes",
}


def main() -> None:
    destination = ROOT / "artifacts/source/sonar-vision-source.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        candidates = [ROOT / name for name in TOP]
        for folder in sorted(ALLOWED):
            candidates.extend((ROOT / folder).rglob("*"))
        for file in sorted(candidates):
            relative = file.relative_to(ROOT)
            if not file.is_file() or file.is_symlink():
                continue
            if relative.parts[0] not in ALLOWED and str(relative) not in TOP:
                continue
            if (
                "__pycache__" in relative.parts
                or any(part.endswith(".egg-info") for part in relative.parts)
                or file.suffix == ".pyc"
                or file.name in {".env", ".DS_Store"}
                or file.name.startswith("._")
            ):
                continue
            archive.write(file, str(relative))
    print(destination)


if __name__ == "__main__":
    main()
