"""Synthetic software data only; never real training/rights evidence."""

import copy
import json

import cv2
import numpy as np
import pytest

from sonar_vision.training.dataset import sha256
from sonar_vision.training.video_dataset import (
    export_video_dataset,
    review_records,
    verify_video_export,
)


@pytest.fixture
def reviewed(tmp_path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("SOFTWARE FIXTURE ONLY, not a real rights grant")
    rows, permits = [], {}
    for i, split in enumerate(["train", "val"]):
        video = tmp_path / f"video-{i}.mp4"
        video.write_bytes(f"software-fixture-{i}".encode())
        digest = sha256(video)
        image = tmp_path / f"frame-{i}.jpg"
        cv2.imwrite(
            str(image), np.random.default_rng(i).integers(0, 255, (100, 150, 3), dtype=np.uint8)
        )
        rows.append(
            {
                "image": str(image),
                "image_sha256": sha256(image),
                "video_sha256": digest,
                "split": split,
                "capture_group": f"capture{i}",
                "vehicle_group": f"vehicle{i}",
                "reviewed": True,
                "all_objects_reviewed": True,
                "boxes": [
                    {"class_id": 0, "xyxy": [10, 10, 140, 90]},
                    {"class_id": 1, "xyxy": [25, 12, 100, 30]},
                ],
            }
        )
        permits[digest] = {
            "train_ML": True,
            "evaluate_ML": True,
            "rights_reviewed": True,
            "author": f"fixture-author-{i}",
            "source_url": "https://example.org/fixture",
            "evidence_file": str(evidence),
            "evidence_sha256": sha256(evidence),
            "video_file": str(video),
        }
    return {"images": rows}, permits


def test_two_class_export_and_hash_verification(tmp_path, reviewed):
    review, permits = reviewed
    rp, pp = tmp_path / "review.json", tmp_path / "permissions.json"
    rp.write_text(json.dumps(review))
    pp.write_text(json.dumps(permits))
    output = tmp_path / "export"
    audit = export_video_dataset(rp, pp, output)
    assert audit["counts"] == {"train": 1, "val": 1}
    assert verify_video_export(output / "data.yaml")["status"] == "REVIEWED_EXPORT"
    assert (output / "labels/train/00000.txt").read_text().startswith("0 ")
    with pytest.raises(ValueError, match="existente"):
        export_video_dataset(rp, pp, output)
    (output / "labels/train/00000.txt").write_text("0 0.5 0.5 1 1\n")
    with pytest.raises(ValueError, match="alterados"):
        verify_video_export(output / "data.yaml")


def test_no_rights_never_opens_unlicensed_frame(reviewed):
    review, _ = reviewed
    review["images"][0]["image"] = "/does/not/exist"
    with pytest.raises(PermissionError, match="autorizado"):
        review_records(review, {})


def test_pexels_download_permission_is_not_an_ML_grant(reviewed):
    review, permits = reviewed
    permits[review["images"][0]["video_sha256"]]["provider"] = "pexels"
    with pytest.raises(PermissionError, match="Pexels"):
        review_records(review, permits)


@pytest.mark.parametrize("field", ["capture_group", "vehicle_group"])
def test_group_leakage_rejected(reviewed, field):
    review, permits = reviewed
    review["images"][1][field] = review["images"][0][field]
    with pytest.raises(ValueError, match="cruza"):
        review_records(review, permits)


@pytest.mark.parametrize("kind", ["geometry", "class", "review", "coverage", "ignore"])
def test_unreviewed_or_invalid_labels_cannot_train(reviewed, kind):
    review, permits = copy.deepcopy(reviewed)
    row = review["images"][0]
    if kind == "geometry":
        row["boxes"][0]["xyxy"][0] = -1
    if kind == "class":
        row["boxes"][0]["class_id"] = 2
    if kind == "review":
        row["reviewed"] = False
    if kind == "coverage":
        row["all_objects_reviewed"] = False
    if kind == "ignore":
        row["ignore_regions"] = [[1, 1, 10, 10]]
    with pytest.raises(ValueError):
        review_records(review, permits)


def test_private_instruction_does_not_assert_rights_or_allow_redistribution(tmp_path, reviewed):
    review, permits = reviewed
    record = tmp_path / "private.txt"
    record.write_text("SOFTWARE FIXTURE user instruction; not a third-party grant")
    for permit in permits.values():
        permit.update(
            train_ML=False,
            evaluate_ML=False,
            rights_reviewed=False,
            usage_basis="user_directed_private_experiment",
            private_experiment_recorded=True,
            public_demo=False,
            redistribution=False,
            user_instruction_record=str(record),
            user_instruction_sha256=sha256(record),
        )
    assert len(review_records(review, permits)) == 2
    assert all(p["rights_reviewed"] is False for p in permits.values())
    first = next(iter(permits.values()))
    first["redistribution"] = True
    with pytest.raises(PermissionError):
        review_records(review, permits)
    first["redistribution"] = False
    record.write_text("ALTERED")
    with pytest.raises(PermissionError, match="registro"):
        review_records(review, permits)
