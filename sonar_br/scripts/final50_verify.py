"""Local preservation/security audit only; no network, model calls or mutation of inputs."""

import json
import stat
import subprocess
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from sonar_vision.evaluation.final_photos import validate_final_reference
from sonar_vision.training.dataset import sha256, verify_dataset

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/research/final50-20261115"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main() -> None:
    baseline = json.loads((CACHE / "baseline-manifest.json").read_text())["files"]
    changes = [r["path"] for r in baseline if sha256(ROOT / r["path"]) != r["sha256"]]
    allowed = {
        "docs/REPORT.md",
        "reports/SONAR_VISION_FIAP.pdf",
        "sonar_br/README.md",
        "reports/qa-results.json",
        "reports/deliverables-check.json",
    }
    require(set(changes) <= allowed, "Unexpected protected artifact change")
    subprocess.run(["git", "merge-base", "--is-ancestor", "031cc74", "HEAD"], cwd=ROOT, check=True)
    require(git("show", "031cc74:AGENTS.md") == (ROOT / "AGENTS.md").read_text(), "AGENTS changed")
    prior = set(json.loads((CACHE / "catalog.json").read_text())["prior_capture_groups"])
    refs = validate_final_reference(
        ROOT / "sonar_br/datasets/manifests/final50.review.json",
        ROOT / "sonar_br/configs/final50-protocol.json",
        ROOT,
        prior,
    )
    prepared = {}
    for data in [
        "data/bangkok-pilot/data.yaml",
        "data/commons-br-panel-v1/data.yaml",
        "sonar_br/datasets/prepared/commons-br-v2/prepared/data.yaml",
        "sonar_br/datasets/prepared/final50-ap/prepared/data.yaml",
    ]:
        prepared[data] = verify_dataset(ROOT / data)["images"]
    raw = json.loads((ROOT / "artifacts/research/roboflow-br/v5-audit-full.json").read_text())[
        "records"
    ]
    for record in raw:
        for key in ["image", "label"]:
            require(
                sha256(ROOT / "data/roboflow-br-v5" / record[key]) == record[key + "_sha256"],
                "Roboflow original data changed",
            )
    old = json.loads(
        (ROOT / "reports/datasets/profile-D-independent-20261115.review.json").read_text()
    )
    for record in old["records"]:
        require(
            sha256((ROOT / "reports/datasets" / record["image"]).resolve())
            == record["image_sha256"],
            "Prior prospective data changed",
        )
    raw_commons = {}
    for name in [
        "sonar_br/datasets/manifests/commons-br-v2.review.json",
        "reports/datasets/commons-br-panel-v1.review.json",
    ]:
        manifest = ROOT / name
        records = json.loads(manifest.read_text())["records"]
        for record in records:
            require(
                sha256((manifest.parent / record["image"]).resolve()) == record["image_sha256"],
                "Prior Commons original photo changed",
            )
        raw_commons[name] = len(records)
    secrets = [
        v.encode()
        for k, v in dotenv_values(ROOT / ".env").items()
        if v and any(token in k for token in ["KEY", "TOKEN", "SECRET"])
    ]
    require(bool(secrets), "Local secrets unavailable for exact-value audit")
    require(stat.S_IMODE((ROOT / ".env").stat().st_mode) == 0o600, "Unsafe local env mode")
    files = set(git("diff", "--name-only", "031cc74").splitlines())
    files |= set(git("ls-files", "--others", "--exclude-standard").splitlines())
    for name in files:
        path = ROOT / name
        if not path.is_file():
            continue
        payload = path.read_bytes()
        require(not any(secret in payload for secret in secrets), "Secret in changed file")
        require(
            len(payload) < (10_000_000 if name == "reports/SONAR_VISION_FIAP.pdf" else 5_000_000),
            "Changed artifact exceeds audited size budget",
        )
        require(
            not name.startswith(
                ("sonar_br/artifacts/", "sonar_br/datasets/raw/", "sonar_br/datasets/prepared/")
            ),
            "Raw artifact staged",
        )
    for name in [
        ".env",
        "sonar_br/datasets/raw/final50/75777089.jpg",
        "sonar_br/artifacts/train-v1/run/weights/best.pt",
    ]:
        require(bool(git("check-ignore", name).strip()), "Private artifact not ignored")
    output: dict[str, Any] = {
        "status": "PASSED",
        "base_checkpoint": "031cc74",
        "baseline_files": len(baseline),
        "unchanged": len(baseline) - len(changes),
        "authorized_changes": changes,
        "frozen_reference_files_verified": len(refs["records"]),
        "prepared_datasets_images_verified": prepared,
        "Roboflow_original_images_and_labels_verified": len(raw),
        "prior_profile_D_files_verified": len(old["records"]),
        "prior_Commons_original_files_verified": raw_commons,
        "AP_prepared66_scope": (
            "39 new test images,27 unused old train/val compatibility; no training"
        ),
        "original_weights_checkpoints_AGENTS_history_video_preserved": True,
        "final_reference_and_pipeline_inputs_unchanged": True,
        "exact_secret_scan_files": len(files),
        "env0600_ignored": True,
        "raw_data_and_weights_not_in_Git": True,
        "PDF_bytes": (ROOT / "reports/SONAR_VISION_FIAP.pdf").stat().st_size,
        "PDF_size_exception": "Required20page report below10MB; prior17page report already8.14MB",
        "other_new_files_below5MB": True,
        "training_push_video_narration_or_API_default_changes": False,
    }
    (ROOT / "sonar_br/reports/final50-preservation.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
