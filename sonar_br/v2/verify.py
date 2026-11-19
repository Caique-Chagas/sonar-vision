"""Offline final V2 evidence audit, without inference, training or remote calls."""

import json
from pathlib import Path

from sonar_vision.br.dataset import audit_training_readiness
from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/research/sonar-br-v2-20261115"


def main() -> None:
    baseline = json.loads((CACHE / "baseline-artifact-manifest.json").read_text())
    for item in baseline["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Prior artifact changed: " + item["path"])
    audit = json.loads((ROOT / "reports/br-v2-inbox-audit.json").read_text())
    if len(audit["records"]) != 250:
        raise ValueError("Incomplete inbox audit")
    for ref in audit["records"]:
        file = ROOT / "sonar_br/datasets/inbox/images" / ref["filename"]
        if sha256(file) != ref["sha256"]:
            raise ValueError("Original inbox bytes changed")
    frozen_path = ROOT / "sonar_br/configs/v2-comparison-frozen.json"
    spec = json.loads(frozen_path.read_text())
    for path, digest in spec["weights_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Compared weights changed")
    for path, digest in spec["source_hashes"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Executed comparison source changed")
    for model, files in spec["OCR_artifacts"].items():
        for path, digest in files.items():
            if sha256(Path.home() / ".paddlex/official_models" / model / path) != digest:
                raise ValueError("OCR artifact changed")
    for name, expected in [("new_test", 44), ("known_benchmark", 56)]:
        paths = list((ROOT / "sonar_br/artifacts/v2-evaluation" / name).glob("*.json"))
        if len(paths) != expected:
            raise ValueError("Incomplete paired cases")
        for path in paths:
            row = read_case(path, sha256(frozen_path))
            if row is None or row["final_code"] is not None or row["final_status"] != "UNCERTAIN":
                raise ValueError("A photograph cannot confirm a line")
    ready = json.loads((ROOT / "reports/br-v2-readiness.json").read_text())
    result = audit_training_readiness(
        ROOT / "sonar_br/datasets/prepared/br-v2/prepared/data.yaml",
        set(ready["reserved_hashes"]),
    )
    if not result["training_eligible"]:
        raise ValueError("Dataset review no longer passes")
    ocr_refs = json.loads((ROOT / "sonar_br/datasets/manifests/br-v2-ocr.review.json").read_text())
    for ref in ocr_refs["records"]:
        if sha256(Path(ref["crop"])) != ref["crop_sha256"]:
            raise ValueError("Reviewed OCR crop changed")
    report = {
        "status": "PASSED",
        "prior_artifacts_unchanged": len(baseline["files"]),
        "original_inbox_files_unchanged": 250,
        "paired_final_receipts": 100,
        "reviewed_OCR_crops": len(ocr_refs["records"]),
        "frozen_comparison_code_and_models": "identical",
        "OCR_cache_models": "identical",
        "dataset": result,
        "known_benchmark": "historical, not new independent test",
        "public_push": False,
        "temporal_validation": False,
    }
    (ROOT / "reports/v2-preservation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
