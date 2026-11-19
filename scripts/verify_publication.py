"""Audit the selected public Git index, relative documentation links and file limits."""

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = {
    "AGENTS.md",
    "STATE.md",
    "PLAN.md",
    "PROGRESS.md",
    "HUMAN_ACTIONS.md",
    "DECISIONS.md",
    "CODEX_ACTIVITY_LOG.md",
}
MEDIA = {".mp4", ".webm", ".wav", ".aiff", ".pt", ".onnx", ".zip", ".dylib"}


def main():
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached"], cwd=ROOT, text=True
    ).splitlines()
    errors = []
    required = {
        "scripts/demo_final.sh",
        "reports/SONAR_VISION_FIAP.pdf",
        "reports/runtime-models.json",
        "README.md",
    }
    errors.extend("Required file not tracked: " + p for p in sorted(required - set(paths)))
    total = 0
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            errors.append("Missing selected file: " + name)
            continue
        total += path.stat().st_size
        if path.name in INTERNAL or path.name == ".env" or path.suffix.lower() in MEDIA:
            errors.append("Private/unexpected file: " + name)
        if any(part in {".venv", "__pycache__", "snapshots"} for part in path.parts):
            errors.append("Generated/private directory: " + name)
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append("Unexpected file >10MiB: " + name)
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}:
            continue
        try:
            text = path.read_text()
        except UnicodeError:
            errors.append("Unexpected binary: " + name)
            continue
        if (chr(47) + "Users" + chr(47)) in text:
            errors.append("Absolute Mac path: " + name)
        if re.search(
            r"ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            text,
        ):
            errors.append("Credential pattern: " + name)
        if path.suffix == ".md":
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                if target.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                target = unquote(target.split("#")[0].strip("<>"))
                resolved = (path.parent / target).resolve()
                if not resolved.exists() or not resolved.is_relative_to(ROOT):
                    errors.append("Broken relative link: " + name + " -> " + target)
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        json.dumps(
            dict(
                status="PASS",
                selected_files=len(paths),
                tree_bytes=total,
                video_files=0,
                private_agent_files=0,
                relative_links="PASS",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
