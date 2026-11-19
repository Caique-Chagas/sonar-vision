"""Eligibility gates; geometry/export/hash handling stays in shared dataset modules."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sonar_vision.training.dataset import prepare_dataset, sha256, verify_dataset
from sonar_vision.training.panel_annotation import export_reviewed

MINIMUM_IMAGES = {"train": 20, "val": 5, "test": 5}
MINIMUM_POSITIVE_GROUPS = {"train": 15, "val": 4, "test": 4}


def prepare_reviewed_dataset(review: Path, output: Path, reserved_hashes: set[str]) -> None:
    if output.exists():
        raise ValueError("Preserve the existing dataset version")
    records = json.loads(review.read_text())["records"]
    if any(record["image_sha256"] in reserved_hashes for record in records):
        raise ValueError("Previously reserved evaluation image in new dataset")
    annotations = output / "annotations"
    export_reviewed(review, annotations)
    prepared = output / "prepared"
    prepare_dataset(
        annotations / "manifest.csv",
        prepared,
        split_assurance="reviewed_capture_vehicle_groups_not_fleet_representative",
    )
    receipt = {
        "geometry_policy": "whole_main_front_destination_display",
        "review_path": str(review.resolve()),
        "review_sha256": sha256(review),
        "manifest_sha256": sha256(annotations / "manifest.csv"),
    }
    (prepared / "br-review.json").write_text(json.dumps(receipt, indent=2) + "\n")


def audit_training_readiness(
    data: Path, reserved_hashes: set[str], *, allow_known_test: bool = False
) -> dict[str, Any]:
    """A small-pilot gate, not a statistical claim of dataset sufficiency.

    The existing preparer must have produced the audited dataset from reviewed
    full-panel annotations. Neither an empty negative nor a repeated vehicle can
    satisfy the positive-group requirement. Test hashes cannot enter train/val.
    """
    audit = verify_dataset(data)
    receipt = json.loads((data.parent / "br-review.json").read_text())
    if (
        receipt["geometry_policy"] != "whole_main_front_destination_display"
        or receipt["manifest_sha256"] != audit["manifest_sha256"]
        or sha256(Path(receipt["review_path"])) != receipt["review_sha256"]
    ):
        raise ValueError("Full-panel review is missing or changed")
    if audit.get("split_strategy") != "explicit_reviewed":
        raise ValueError("Explicit visually reviewed group splits are required")
    group_splits: dict[str, str] = {}
    positives: dict[str, set[str]] = {split: set() for split in MINIMUM_IMAGES}
    counts: Counter[str] = Counter()
    for record in audit["records"]:
        split, group = record["split"], record["group_id"]
        if split not in positives or not group:
            raise ValueError("Invalid split or capture group")
        if group in group_splits and group_splits[group] != split:
            raise ValueError("Capture group crosses splits")
        group_splits[group] = split
        if record["sha256"] in reserved_hashes and (split != "test" or not allow_known_test):
            raise ValueError("Previously reserved evaluation image cannot enter this experiment")
        counts[split] += 1
        if record["boxes"]:
            positives[split].add(group)
    reasons = []
    for split, minimum in MINIMUM_IMAGES.items():
        if counts[split] < minimum:
            reasons.append(f"{split}: {counts[split]} images, minimum pilot gate {minimum}")
        if len(positives[split]) < MINIMUM_POSITIVE_GROUPS[split]:
            reasons.append(f"{split}: insufficient distinct positive capture groups")
    return {
        "training_eligible": not reasons,
        "reasons": reasons,
        "images": dict(counts),
        "positive_groups": {split: len(groups) for split, groups in positives.items()},
        "scope": "engineering gate for a bounded pilot, not representativeness",
    }


def reserved_test_hashes(manifests: list[Path]) -> set[str]:
    """Read prior frozen test manifests; no image decoding or OCR is needed."""
    hashes: set[str] = set()
    for manifest in manifests:
        records = json.loads(manifest.read_text())["records"]
        for record in records:
            if record.get("split", "").startswith("prospective_test") or record.get("split") in {
                "val",
                "test",
            }:
                hashes.add(record.get("image_sha256") or record["sha256"])
    return hashes
