"""Offline preservation and provenance checks for the source-import phase."""

import ast
import json
from pathlib import Path

from sonar_br.final_correction.verify import main as verify_previous
from sonar_br.source_import.build import OUTPUT, ROOT
from sonar_br.source_import.report import main as verify_cells
from sonar_vision.training.dataset import sha256
from sonar_vision.training.recognition_dataset import verify_recognition


def main() -> None:
    verify_previous()
    verify_cells()
    cache = ROOT / "artifacts/research/source-import-20261117"
    audit = verify_recognition(OUTPUT / "ocr")
    if audit["counts"] != {"train": 4, "val": 1, "test": 0}:
        raise ValueError("Corpus must retain empty independent test")
    if any(not r["source"].startswith("https://commons.wikimedia.org/") for r in audit["records"]):
        raise ValueError("Quarantined source entered OCR corpus")
    executed = cache / "compare-executed.py"
    formatted = ROOT / "sonar_br/source_import/compare.py"
    binding = ROOT / "sonar_br/artifacts/source-import-20261117/binding.json"
    bound = json.loads(binding.read_text())
    protocol = ROOT / "sonar_br/configs/source-import-ocr-protocol.json"
    if bound["protocol"] != sha256(protocol) or bound["corpus"] != sha256(
        OUTPUT / "ocr/audit.json"
    ):
        raise ValueError("Frozen protocol/corpus changed")
    if sha256(executed) != bound["runner"]:
        raise ValueError("Exact executed source changed")
    if ast.dump(ast.parse(executed.read_text()), include_attributes=False) != ast.dump(
        ast.parse(formatted.read_text()), include_attributes=False
    ):
        raise ValueError("Formatting changed experiment logic")
    spec = json.loads((ROOT / "sonar_br/configs/source-import-ocr-protocol.json").read_text())
    for path, digest in spec["files"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Frozen model/source/receipt changed")
    for name in ["en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_det"]:
        for row in json.loads((cache / (name + "-preservation.json")).read_text()):
            if sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("Preserved OCR weight changed")
    # Recheck sampled originals and labels against the pre-existing complete audit.
    samples = json.loads((cache / "br-sample100.json").read_text())["records"]
    for row in samples:
        for key, digest in [("image", row["image_sha256"]), ("label", row["label_sha256"])]:
            if sha256(ROOT / "data/roboflow-br-v5" / row[key]) != digest:
                raise ValueError("Original Roboflow export changed")
    yolo_review = ROOT / "sonar_br/datasets/manifests/import-20261117.yolo.json"
    yolo = json.loads(yolo_review.read_text())
    receipt = json.loads((OUTPUT / "yolo-annotations/review-receipt.json").read_text())
    if sha256(yolo_review) != receipt["review_sha256"] or len(yolo["records"]) != 5:
        raise ValueError("YOLO intermediate review changed")
    for row in yolo["records"]:
        if sha256(Path(row["image"])) != row["image_sha256"]:
            raise ValueError("YOLO reviewed pixels changed")
    result = {
        "status": "PASSED",
        "old_manifests_verified": [75, 173, 117, 434, 274],
        "new_OCR_cells": 30,
        "OCR_splits": audit["counts"],
        "newly_listed_sources_ML_images": 0,
        "reserved209P_unchanged_and_not_inferred": True,
        "old_OCR_weight_files_unchanged": True,
        "API_PDF_video_original_weights": "preserved",
        "training": False,
        "push": False,
    }
    (ROOT / "reports/source-import-preservation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    manifest = ROOT / "reports/source-import-artifact-manifest.json"
    if manifest.exists():
        for row in json.loads(manifest.read_text())["files"]:
            if sha256(ROOT / row["path"]) != row["sha256"]:
                raise ValueError("Phase evidence changed:" + row["path"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
