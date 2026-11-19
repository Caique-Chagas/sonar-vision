"""Prepare small licensed DEV corpora; blocked source photographs never enter them."""

import json
from dataclasses import asdict
from pathlib import Path

from sonar_vision.training.dataset import sha256
from sonar_vision.training.panel_annotation import export_reviewed
from sonar_vision.training.recognition_dataset import (
    RecognitionSample,
    export_recognition,
    verify_recognition,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "sonar_br/datasets/manifests"
OUTPUT = ROOT / "sonar_br/datasets/prepared/source-import-20261117"


def write_once(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("Preserve frozen corpus manifest: " + str(path))
    if not path.exists():
        path.write_text(text)


def main() -> None:
    old = json.loads((MANIFESTS / "generalization-20261116.ocr.json").read_text())["records"]
    previous = json.loads((MANIFESTS / "generalization-20261116.review.json").read_text())[
        "records"
    ]
    for r in old:
        parent = next(p for p in previous if p["id"] == r["id"])
        if parent["rights_verified"] is not True or parent["reviewed"] is not True:
            raise ValueError("Prior crop source lacks explicit rights/pixel review")
    samples = [
        RecognitionSample(
            image=Path(r["image"]),
            transcript=r["expected_code"],
            split="val" if r["id"] == "189553067" else "train",
            parent_id=r["id"],
            group=r["group_id"],
            author=r["author"],
            source=r["source"],
            license=r["license"],
            image_sha256=r["crop_sha256"],
            rights_verified=True,
            manual_verified=True,
            scope="known_development",
        )
        for r in old
    ]
    local = json.loads((MANIFESTS / "final-correction-20261117.local.review.json").read_text())[
        "records"
    ][0]
    code = ROOT / "sonar_br/datasets/raw/source-import-20261117/5701-code.png"
    samples.append(
        RecognitionSample(
            image=code,
            transcript="5701-10",
            split="train",
            parent_id=local["id"],
            group=local["group_id"],
            author=local["author"],
            source=local["source"],
            license=local["license"],
            image_sha256=sha256(code),
            rights_verified=True,
            manual_verified=True,
            scope="known_development",
            vehicle="SPTrans6-1479",
        )
    )
    ocr_manifest = MANIFESTS / "import-20261117.ocr.json"
    write_once(
        ocr_manifest,
        {
            "status": "PREPARATION_DEV_ONLY_TRAIN4_VAL1_TEST0_NOT_A_BENCHMARK",
            "no_newly_audited_source_photo_approved": True,
            "crop5701": {
                "parent": local["original_image"],
                "native_xyxy": [1321, 1660, 1535, 1720],
                "verified_from_pixels_before_any_new_OCR": True,
            },
            "older_crop_provenance": "generalization-20261116.ocr.json",
            "records": [asdict(s) for s in samples],
        },
    )
    if (OUTPUT / "ocr").exists():
        ocr_receipt = verify_recognition(OUTPUT / "ocr")
    else:
        ocr_receipt = export_recognition(samples, OUTPUT / "ocr")
    refs = json.loads((MANIFESTS / "generalization-20261116.review.json").read_text())["records"]
    selected = [
        dict(r) for r in refs if r["id"] in {"69139074", "64802621", "10190263", "22865593"}
    ]
    context = json.loads((MANIFESTS / "final-correction-20261117.train-contexts.json").read_text())[
        "records"
    ][0]
    selected.append(
        dict(
            context,
            image=str(ROOT / context["image"]),
            image_sha256=context["sha256"],
            group_id=context["group_id"],
            boxes=local["boxes"],
        )
    )
    for r in selected:
        if r.get("exhaustive_annotation") is not True:
            raise ValueError("Incomplete frame cannot enter YOLO annotation corpus")
        r["split"] = "val" if r["id"] == "22865593" else "train"
        r["group_evidence"] = (
            "Documented photographer/date and physical source; no crop counted as new group"
        )
        r["visual_evidence"] = (
            "Previously reviewed complete physical main panel, same geometry, no pseudo-label"
        )
        r["rights_evidence"] = (
            "Original Commons photographer/own work or reviewed Flickr chain, "
            "retained source and license"
        )
    yolo_manifest = MANIFESTS / "import-20261117.yolo.json"
    write_once(
        yolo_manifest,
        {
            "geometry_policy": "whole_main_front_destination_display",
            "reviewer": "engineering agent; complete-frame reference from preserved reviews",
            "independent_human_review": False,
            "status": "ANNOTATIONS_ONLY_TEST0_TRAINING_BLOCKED",
            "records": selected,
        },
    )
    annotations = OUTPUT / "yolo-annotations"
    if not annotations.exists():
        export_reviewed(yolo_manifest, annotations)
    elif json.loads((annotations / "review-receipt.json").read_text())["review_sha256"] != sha256(
        yolo_manifest
    ):
        raise ValueError("YOLO annotations belong to a different review")
    result = {
        "OCR": ocr_receipt["counts"],
        "YOLO": {"train": 4, "val": 1, "test": 0},
        "training_allowed": False,
        "reason": ("Insufficient data/no independent test; strict YOLO preparer not bypassed"),
        "new_source_images_for_ML": 0,
    }
    (ROOT / "sonar_br/reports/source-import-corpora.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
