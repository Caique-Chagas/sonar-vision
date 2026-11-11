"""Persist real QA outputs and exit nonzero on any failing required check."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "reports/qa"
    output.mkdir(parents=True, exist_ok=True)
    commands = {
        "lint": [sys.executable, "-m", "ruff", "check", "."],
        "format": [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "src",
            "scripts",
            "tests",
            "notebooks",
        ],
        "types": [sys.executable, "-m", "mypy", "src"],
        "dependencies": [sys.executable, "-m", "pip", "check"],
        "tests": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--cov=sonar_vision",
            "--cov-report=json:reports/coverage.json",
            "--junitxml=reports/pytest.xml",
        ],
    }
    results = []
    for name, command in commands.items():
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        log = result.stdout + result.stderr
        (output / f"{name}.log").write_text(log.replace(str(ROOT), "<PROJECT>"))
        results.append(
            {"check": name, "exit_code": result.returncode, "log": f"reports/qa/{name}.log"}
        )
        print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}")
    junit = ROOT / "reports/pytest.xml"
    if junit.exists():
        tree = ET.parse(junit)
        for element in tree.iter():
            element.attrib.pop("hostname", None)
        tree.write(junit, encoding="utf-8", xml_declaration=True)
    passed = all(result["exit_code"] == 0 for result in results)
    report = {
        "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
        "status": "PASSED" if passed else "FAILED",
        "checks": results,
    }
    (ROOT / "reports/qa-results.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
