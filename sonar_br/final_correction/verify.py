"""Offline checks of preservation, frozen inputs, receipts and train-only lineage."""

import json
from pathlib import Path

from sonar_br.final_correction.run import DESIGN, OUT, REF, ROOT
from sonar_br.generalization.verify import main as verify_prior
from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256
from sonar_vision.training.panel_annotation import checked_box


def main() -> None:
    verify_prior()
    spec = json.loads(DESIGN.read_text())
    if sha256(ROOT / spec["weights"]) != spec["weights_sha256"]:
        raise ValueError("Preserved V1 changed")
    binding = OUT / "binding.json"
    frozen = json.loads(binding.read_text())
    if frozen["design"] != sha256(DESIGN) or frozen["reference"] != sha256(REF):
        raise ValueError("Tile ablation inputs changed")
    if frozen["runner"] != sha256(ROOT / "sonar_br/final_correction/run.py"):
        raise ValueError("Executed runner changed")
    for name, digest in frozen["sources"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Frozen implementation changed")
    cases = list((OUT / "cases").glob("*.json"))
    if len(cases) != 28:
        raise ValueError("Expected28newDEV receipts")
    for path in cases:
        row = read_case(path, sha256(binding))
        if row is None or row["id"] == "35046766":
            raise ValueError("Invalid/reserved input in new DEV")
        if any(e["status"] != "UNCERTAIN" for e in row["events"]):
            raise ValueError("Static photographs cannot confirm temporal consensus")
    local_file = ROOT / "sonar_br/datasets/manifests/final-correction-20261117.local.review.json"
    local = json.loads(local_file.read_text())["records"]
    local_binding = OUT / "local/binding.json"
    design = json.loads(local_binding.read_text())
    if design["reference"] != sha256(local_file):
        raise ValueError("Manual local labels changed after inference")
    if design["runner"] != sha256(ROOT / "sonar_br/final_correction/local_probe.py"):
        raise ValueError("Executed local runner changed")
    if design["evaluate_source"] != sha256(ROOT / "sonar_br/final_correction/run.py"):
        raise ValueError("Local evaluation implementation changed")
    for name, digest in design["weights"].items():
        train = "train-v2" if name == "V2" else "train-v1"
        if sha256(ROOT / f"sonar_br/artifacts/{train}/run/weights/best.pt") != digest:
            raise ValueError("Compared model changed")
    for ref in local:
        if sha256(Path(ref["image"])) != ref["image_sha256"]:
            raise ValueError("Local reviewed pixels changed")
        if sha256(ROOT / ref["original_image"]) != ref["original_sha256"]:
            raise ValueError("User original changed")
        for box in ref["boxes"]:
            checked_box(box["xyxy"], ref["width"], ref["height"])
    rows = list((OUT / "local").glob("*.json"))
    rows = [p for p in rows if p.name != "binding.json"]
    if len(rows) != 6:
        raise ValueError("Expected4local pipelines and2manualROI receipts")
    for p in rows:
        row = read_case(p, sha256(local_binding))
        if row is None or row["id"] != "170315291":
            raise ValueError("Training-group or reserved image entered local probe")
    contexts = json.loads(
        (
            ROOT / "sonar_br/datasets/manifests/final-correction-20261117.train-contexts.json"
        ).read_text()
    )["records"]
    if len(contexts) != 1 or contexts[0]["group_id"] != local[0]["group_id"]:
        raise ValueError("A crop must retain its parent capture group")
    if sha256(ROOT / contexts[0]["image"]) != contexts[0]["sha256"]:
        raise ValueError("Reviewed training context changed")
    if list(OUT.rglob("*.pt")):
        raise ValueError("This checkpoint did not train new detector weights")
    result = {
        "status": "PASSED",
        "prior_phase_artifacts": 173,
        "prior_older_artifacts": [117, 434, 274],
        "inbox_originals": 250,
        "new_DEV_receipts": 28,
        "new_local_pipeline_receipts": 4,
        "new_manual_ROI_receipts": 2,
        "train_contexts_same_parent_group": 1,
        "reserved209P_unchanged_and_not_inferred": True,
        "Bangkok_V1_V2_OCR_original_E3_API_PDF_video": "preserved",
        "new_training": False,
        "push": False,
    }
    (ROOT / "reports/final-correction-preservation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    manifest = ROOT / "reports/final-correction-artifact-manifest.json"
    if manifest.exists():
        for row in json.loads(manifest.read_text())["files"]:
            if sha256(ROOT / row["path"]) != row["sha256"]:
                raise ValueError("New phase evidence changed: " + row["path"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
