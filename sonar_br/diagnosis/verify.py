"""Offline checkpoint audit: preserve prior artifacts and new source reservations."""

import json
from pathlib import Path

from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/research/v2-diagnosis-20261116"


def main() -> None:
    baseline = json.loads((CACHE / "baseline-v2.json").read_text())
    for item in baseline["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("V2 artifact changed: " + item["path"])
    older = json.loads(
        (
            ROOT / "artifacts/research/sonar-br-v2-20261115/baseline-artifact-manifest.json"
        ).read_text()
    )
    for item in older["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Historical artifact changed: " + item["path"])
    inbox = json.loads((ROOT / "reports/br-v2-inbox-audit.json").read_text())["records"]
    for r in inbox:
        if sha256(ROOT / "sonar_br/datasets/inbox/images" / r["filename"]) != r["sha256"]:
            raise ValueError("Original inbox changed")
    for name in ["", "source3200/"]:
        folder = ROOT / "sonar_br/artifacts/v2-diagnosis" / name
        binding = sha256(folder / "binding.json")
        paths = list((folder / "cases").glob("*.json"))
        if len(paths) != 22:
            raise ValueError("Incomplete paired experiment")
        for path in paths:
            r = read_case(path, binding)
            if r is None or any(e["status"] != "UNCERTAIN" for e in r["events"]):
                raise ValueError("Unexpected photo confirmation")
    sources = json.loads(
        (ROOT / "sonar_br/datasets/manifests/directed-20261116.sources.json").read_text()
    )["records"]
    for r in sources:
        if sha256(ROOT / r["image"]) != r["image_sha256"]:
            raise ValueError("New original changed")
        if not r["rights_verified"] or r["training_approved"]:
            raise ValueError("Source must remain reviewed but unapproved for training")
    # No candidate-source image may enter this phase's known-photo inference.
    inferred = {
        read_case(p, sha256(p.parent.parent / "binding.json"))["id"]
        for p in (ROOT / "sonar_br/artifacts/v2-diagnosis/cases").glob("*.json")
    }
    if inferred & {r["id"] for r in sources}:
        raise ValueError("Reserved new source inferred")
    evidence = {
        "status": "PASSED",
        "prior_V2_artifacts_identical": len(baseline["files"]),
        "earlier_artifacts_identical": len(older["files"]),
        "original_inbox_files": len(inbox),
        "new_originals_verified": len(sources),
        "new_test_candidate_not_inferred": True,
        "paired_diagnostic_receipts": 44,
        "no_training_or_promotion": True,
    }
    (ROOT / "reports/diagnosis-preservation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
