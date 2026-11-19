"""Frozen-reference validation and case persistence for paired photo-only evaluation."""

import json
from pathlib import Path
from typing import Any

from sonar_vision.training.dataset import sha256
from sonar_vision.training.panel_annotation import checked_box

DIFFICULTIES = {"easy", "medium", "hard", "negative"}


def validate_final_reference(
    reference: Path, protocol: Path, root: Path, prior_groups: set[str]
) -> dict[str, Any]:
    spec = json.loads(protocol.read_text())
    result: dict[str, Any] = json.loads(reference.read_text())
    if result.get("protocol_sha256") != sha256(protocol):
        raise ValueError("Reference does not match frozen protocol")
    if not result.get("reviewed_before_model_outputs"):
        raise ValueError("Independent truth must precede model outputs")
    for record in spec["weights"].values():
        if sha256(root / record["path"]) != record["sha256"]:
            raise ValueError("Frozen model changed")
    for name, digest in spec["source_hashes"].items():
        if sha256(root / name) != digest:
            raise ValueError("Frozen implementation changed")
    records = result["records"]
    if not 1 <= len(records) <= 50:
        raise ValueError("Reference requires 1..50 real independent photographs")
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    seen_hashes: set[str] = set()
    for record in records:
        if record.get("rights_verified") is not True or not record.get("rights_evidence"):
            raise ValueError("Rights not verified")
        if not all(record.get(key) for key in ("author", "license", "source", "group_id")):
            raise ValueError("Authorship, source, license and capture group required")
        if record.get("difficulty") not in DIFFICULTIES:
            raise ValueError("Pre-inference difficulty required")
        if record.get("group_id") in seen_groups | prior_groups:
            raise ValueError("Capture group reused")
        if record["id"] in seen_ids or record["image_sha256"] in seen_hashes:
            raise ValueError("Duplicate photograph")
        seen_ids.add(record["id"])
        seen_groups.add(record["group_id"])
        seen_hashes.add(record["image_sha256"])
        path = (reference.parent / record["image"]).resolve(strict=True)
        if sha256(path) != record["image_sha256"]:
            raise ValueError("Image changed after annotation")
        if record.get("expected_code") and not record.get("ground_truth_method"):
            raise ValueError("Code without independent truth")
        if record.get("expected_code") and record.get("code_location") not in {
            "main_display",
            "auxiliary_front_placard",
        }:
            raise ValueError("Body or fleet markings are not display-code ground truth")
        boxes = record.get("boxes", [])
        if (record.get("difficulty") == "negative") != (not boxes):
            raise ValueError("Negative difficulty must agree with geometry")
        if record.get("expected_code") and not boxes:
            raise ValueError("Code without a visible panel")
        if not boxes and record.get("confirmed_negative") is not True:
            raise ValueError("Empty annotation is not a verified negative")
        for box in boxes:
            if not box.get("approved") or not box.get("boundary_evidence"):
                raise ValueError("Whole-panel boundary not reviewed")
            checked_box(box["xyxy"], record["width"], record["height"])
    return result


def read_case(path: Path, binding: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result: dict[str, Any] = json.loads(path.read_text())
    if result.get("binding_sha256") != binding:
        raise ValueError("Cached case belongs to different frozen inputs")
    return result


def write_case(path: Path, value: dict[str, Any], binding: str) -> None:
    if path.exists():
        raise ValueError("Completed case must not be overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value | {"binding_sha256": binding}, output, ensure_ascii=False, indent=2)
        output.write("\n")
