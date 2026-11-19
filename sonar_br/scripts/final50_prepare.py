"""Prepare an AP-only test subset; previous train/val paths are compatibility inputs only."""

import csv
import json
from pathlib import Path
from typing import Any

from sonar_vision.training.dataset import prepare_dataset
from sonar_vision.training.panel_annotation import export_reviewed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sonar_br/datasets/prepared/final50-ap"


def main() -> None:
    if OUT.exists():
        raise ValueError("AP dataset version exists; preserve it")
    reference = ROOT / "sonar_br/datasets/manifests/final50.review.json"
    refs = json.loads(reference.read_text())
    eligible = []
    for row in refs["records"]:
        if not row["ap_eligible"]:
            continue
        eligible.append(
            row
            | {
                "image": str((reference.parent / row["image"]).resolve()),
                "split": "test",
            }
        )
    review = {
        "geometry_policy": "whole_main_front_destination_display",
        "records": eligible,
        "reviewer": "same frozen pre-inference reference; AP-complete subset only",
        "independent_human_review": False,
    }
    OUT.mkdir(parents=True)
    review_path = OUT / "ap-review.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n")
    annotations = OUT / "annotations"
    export_reviewed(review_path, annotations)
    with (annotations / "manifest.csv").open(newline="") as stream:
        rows: list[dict[str, Any]] = list(csv.DictReader(stream))
    for row in rows:
        row["label"] = str((annotations / row["label"]).resolve())
    old_data = ROOT / "sonar_br/datasets/prepared/commons-br-v2/prepared"
    old = json.loads((old_data / "audit.json").read_text())
    for record in old["records"]:
        if record["split"] not in {"train", "val"}:
            continue
        image = old_data / record["prepared_image"]
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        rows.append(
            {
                "image": str(image),
                "label": str(label),
                "split": record["split"],
                "group_id": record["group_id"],
                "source": record["source"],
                "license": record["license"],
            }
        )
    combined = OUT / "manifest.csv"
    with combined.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = prepare_dataset(
        combined,
        OUT / "prepared",
        split_assurance="new author-day test groups; old train/val unused by test-only evaluation",
    )
    receipt = {
        "scope": "Ultralytics split=test only, NO training performed",
        "test_images": report["counts"]["test"],
        "test_panels": sum(r["boxes"] for r in report["records"] if r["split"] == "test"),
        "unused_compatibility_train_val_images": 27,
        "no_new_test_image_in_train_or_val": True,
        "excluded_from_AP": [r["id"] for r in refs["records"] if not r["ap_eligible"]],
        "all50_still_receive_operating_pipeline": True,
    }
    (ROOT / "sonar_br/reports/final50-ap-preparation.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
