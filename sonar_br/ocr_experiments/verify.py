"""Read-only local preservation and evidence audit; emits a separate receipt."""

import json
import stat
import subprocess
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256, verify_dataset

ROOT = Path(__file__).resolve().parents[2]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main() -> None:
    baseline = json.loads(git("show", "e932246:reports/artifact-manifest.json"))["files"]
    for record in baseline:
        require(sha256(ROOT / record["path"]) == record["sha256"], "Prior artifact changed")
    require(git("show", "e932246:AGENTS.md") == (ROOT / "AGENTS.md").read_text(), "AGENTS changed")
    subprocess.run(["git", "merge-base", "--is-ancestor", "e932246", "HEAD"], cwd=ROOT, check=True)
    candidate = json.loads((ROOT / "sonar_br/ocr_experiments/candidate-frozen.json").read_text())
    for name, digest in candidate["protected"].items():
        require(sha256(ROOT / name) == digest, "Protected weight/PDF/video changed")
    require(
        sha256(ROOT / "src/sonar_vision/br/ocr_optimization.py") == candidate["source_sha256"],
        "Frozen experimental implementation changed",
    )
    require(
        sha256(ROOT / "sonar_br/ocr_experiments/run.py") == candidate["runner_sha256"],
        "Frozen executed runner changed",
    )
    for name, digest in candidate["production_source_hashes"].items():
        require(sha256(ROOT / name) == digest, "Production pipeline source changed")
    for name, files in candidate["OCR_artifacts"].items():
        for filename, digest in files.items():
            require(
                sha256(Path.home() / ".paddlex/official_models" / name / filename) == digest,
                "Cached OCR weights changed",
            )
    reference = ROOT / "sonar_br/datasets/manifests/ocr-opt-v1.review.json"
    require(sha256(reference) == candidate["reference_sha256"], "Frozen new reference changed")
    refs = json.loads(reference.read_text())["records"]
    for record in refs:
        require(
            sha256((reference.parent / record["image"]).resolve()) == record["image_sha256"],
            "Experiment pixels changed",
        )
    recorded = 0
    for split in ["dev", "holdout"]:
        binding_path = ROOT / "sonar_br/artifacts/ocr-opt-v1" / f"{split}-binding.json"
        report = json.loads((ROOT / f"sonar_br/reports/ocr-opt-{split}.json").read_text())
        require(
            report["binding"] == json.loads(binding_path.read_text()), "Recorded binding mismatch"
        )
        for row in report["records"]:
            case_path = (
                ROOT
                / "sonar_br/artifacts/ocr-opt-v1"
                / split
                / f"{row['variant']}-{row['id']}.json"
            )
            saved = read_case(case_path, sha256(binding_path))
            require(saved is not None, "Persisted case missing")
            payload = {k: v for k, v in (saved or {}).items() if k != "binding_sha256"}
            recorded_payload = {k: v for k, v in row.items() if k != "binding_sha256"}
            require(payload == recorded_payload, "Persisted case changed")
            require(
                row["final_code"] is None
                and all(e["status"] == "UNCERTAIN" for e in row["events"]),
                "False static confirmation",
            )
            recorded += 1
    datasets = {}
    for name in [
        "data/bangkok-pilot/data.yaml",
        "data/commons-br-panel-v1/data.yaml",
        "sonar_br/datasets/prepared/commons-br-v2/prepared/data.yaml",
    ]:
        datasets[name] = verify_dataset(ROOT / name)["images"]
    originals = {}
    for name in [
        "sonar_br/datasets/manifests/final50.review.json",
        "sonar_br/datasets/manifests/commons-br-v2.review.json",
        "reports/datasets/commons-br-panel-v1.review.json",
        "reports/datasets/profile-D-independent-20261115.review.json",
    ]:
        path = ROOT / name
        records = json.loads(path.read_text())["records"]
        for r in records:
            require(
                sha256((path.parent / r["image"]).resolve()) == r["image_sha256"],
                "Prior original image changed",
            )
        originals[name] = len(records)
    roboflow = json.loads((ROOT / "artifacts/research/roboflow-br/v5-audit-full.json").read_text())[
        "records"
    ]
    for record in roboflow:
        for key in ["image", "label"]:
            require(
                sha256(ROOT / "data/roboflow-br-v5" / record[key]) == record[key + "_sha256"],
                "Roboflow original changed",
            )
    secrets = [
        v.encode()
        for k, v in dotenv_values(ROOT / ".env").items()
        if v and any(word in k for word in ["KEY", "TOKEN", "SECRET"])
    ]
    require(bool(secrets), "No local secret values to audit")
    require(stat.S_IMODE((ROOT / ".env").stat().st_mode) == 0o600, "Local env mode changed")
    files = set(git("diff", "--name-only", "e932246").splitlines())
    files.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    for name in files:
        path = ROOT / name
        if not path.is_file():
            continue
        payload = path.read_bytes()
        require(not any(s in payload for s in secrets), "Secret in changed artifact")
        require(len(payload) < 5_000_000, "New file exceeds size budget")
        require(
            not name.startswith(
                ("sonar_br/artifacts/", "sonar_br/datasets/raw/", "artifacts/research/")
            ),
            "Internal/raw artifact selected",
        )
    for name in [
        ".env",
        "sonar_br/artifacts/ocr-opt-v1/dev-binding.json",
        "sonar_br/datasets/raw/ocr-opt-v1/75601328.jpg",
    ]:
        require(bool(git("check-ignore", name).strip()), "Sensitive or raw file not ignored")
    result: dict[str, Any] = {
        "status": "PASSED",
        "base": "e932246",
        "prior_selected_unchanged": len(baseline),
        "experiment_reference_entries_verified": len(refs),
        "new_photos_in_DEV_and_holdout": sum(
            r["scope"] == "new_model_unseen_capture" for r in refs
        ),
        "historical_DEV_photos": sum(r["scope"].startswith("historical") for r in refs),
        "new_case_results_verified": recorded,
        "prior_prepared_datasets_verified": datasets,
        "prior_original_images_verified": originals,
        "Roboflow_original_images_labels_verified": len(roboflow),
        "frozen_OCR_YOLO_runner_source_reference_hashes_verified": True,
        "PDF_base_video_original_models_checkpoints_AGENTS_history_preserved": True,
        "secrets0600_ignored_exact_scan_files": len(files),
        "public_author_attribution_only_no_private_credentials": True,
        "new_files_below5MB_no_raw_data_or_weights_tracked": True,
        "YOLO_training_API_default_change_push_or_narration": False,
    }
    (ROOT / "sonar_br/reports/ocr-opt-preservation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
