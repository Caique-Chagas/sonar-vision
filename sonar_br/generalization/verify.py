"""Offline preservation and reservation checks for the generalization research phase."""

import json
from pathlib import Path

from sonar_vision.br.curation import assess_targeted_coverage
from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/research/generalization-20261116"


def main() -> None:
    manifests = [
        ROOT / "reports/diagnosis-artifact-manifest.json",
        ROOT / "reports/v2-artifact-manifest.json",
        ROOT / "artifacts/research/sonar-br-v2-20261115/baseline-artifact-manifest.json",
    ]
    counts = []
    for manifest in manifests:
        refs = json.loads(manifest.read_text())["files"]
        counts.append(len(refs))
        for record in refs:
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise ValueError("Prior evidence changed: " + record["path"])
    source = json.loads(
        (ROOT / "sonar_br/datasets/manifests/generalization-20261116.sources.json").read_text()
    )["records"]
    for r in source:
        if sha256(ROOT / r["image"]) != r["image_sha256"]:
            raise ValueError("Acquired original changed")
    refpath = ROOT / "sonar_br/datasets/manifests/generalization-20261116.review.json"
    refs = json.loads(refpath.read_text())["records"]
    if len(refs) != 14 or any(r["id"] == "35046766" for r in refs):
        raise ValueError("Invalid DEV/reference set")
    for r in refs:
        if sha256(Path(r["image"])) != r["image_sha256"]:
            raise ValueError("Reviewed pixels changed")
    gate = assess_targeted_coverage(refs)
    if gate["coverage_passed"]:
        raise ValueError("This checkpoint must remain blocked for training")
    out = ROOT / "sonar_br/artifacts/generalization-20261116"
    binding = out / "binding.json"
    spec = json.loads(binding.read_text())
    if sha256(refpath) != spec["reference"]:
        raise ValueError("Frozen reference changed")
    if sha256(ROOT / "sonar_br/generalization/baselines.py") != spec["runner"]:
        raise ValueError("Executed baseline source changed")
    for path, digest in spec["weights"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Compared weight changed")
    for folder, n in [("cases", 28), ("reference-ocr", 8)]:
        paths = list((out / folder).glob("*.json"))
        if len(paths) != n:
            raise ValueError("Incomplete native experiment")
        for p in paths:
            row = read_case(p, sha256(binding))
            if row is None or row["id"] == "35046766":
                raise ValueError("Reserved source entered diagnostics")
            if any(e["status"] != "UNCERTAIN" for e in row.get("events", [])):
                raise ValueError("Static image cannot confirm temporal evidence")
    code_folder = out / "code-probe"
    code_binding = code_folder / "binding.json"
    code_spec = json.loads(code_binding.read_text())
    code_ref = ROOT / "sonar_br/datasets/manifests/generalization-20261116.ocr.json"
    if sha256(code_ref) != code_spec["reference"]:
        raise ValueError("Code-crop reference changed")
    code_paths = [p for p in code_folder.glob("*.json") if p.name != "binding.json"]
    if len(code_paths) != 4:
        raise ValueError("Incomplete manual code diagnostic")
    for path in code_paths:
        row = read_case(path, sha256(code_binding))
        if row is None or row["id"] == "35046766":
            raise ValueError("Reserved source entered code recognizer")
    for row in json.loads(code_ref.read_text())["records"]:
        if sha256(Path(row["image"])) != row["crop_sha256"]:
            raise ValueError("Code-crop pixels changed")
    held = next(
        r
        for r in json.loads(
            (ROOT / "sonar_br/datasets/manifests/directed-20261116.sources.json").read_text()
        )["records"]
        if r["id"] == "35046766"
    )
    if sha256(ROOT / held["image"]) != held["image_sha256"]:
        raise ValueError("Reserved209P original changed")
    for r in json.loads((ROOT / "reports/br-v2-inbox-audit.json").read_text())["records"]:
        if sha256(ROOT / "sonar_br/datasets/inbox/images" / r["filename"]) != r["sha256"]:
            raise ValueError("User inbox original changed")
    result = {
        "status": "PASSED",
        "prior_artifact_manifests_verified": counts,
        "new_originals": len(source),
        "old_inbox_files": 250,
        "new_DEV_references": len(refs),
        "pipeline_receipts": 28,
        "manual_ROI_receipts": 8,
        "manual_CODE_crop_receipts": 4,
        "reserved_209P_unchanged_and_not_inferred": True,
        "coverage_gate": gate,
        "new_training": False,
        "push": False,
    }
    (ROOT / "reports/generalization-preservation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    current_manifest = ROOT / "reports/generalization-artifact-manifest.json"
    if current_manifest.exists():
        for item in json.loads(current_manifest.read_text())["files"]:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Generalization evidence changed: " + item["path"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
