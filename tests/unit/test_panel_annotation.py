import csv
import json

import cv2
import numpy as np
import pytest

from sonar_vision.training.dataset import prepare_dataset, sha256
from sonar_vision.training.panel_annotation import checked_box, export_reviewed, propose_panels


def review_fixture(tmp_path):
    image = tmp_path / "image.png"
    cv2.imwrite(str(image), np.full((80, 160, 3), 220, np.uint8))
    review = {
        "geometry_policy": "whole_main_front_destination_display",
        "reviewer": "test_fixture_only",
        "records": [
            {
                "image": image.name,
                "image_sha256": sha256(image),
                "reviewed": True,
                "visual_evidence": "fixture",
                "rights_verified": True,
                "rights_evidence": "fixture",
                "author": "fixture",
                "license": "CC0",
                "source": "https://example.org/photo",
                "group_id": "vehicle-a",
                "group_evidence": "fixture",
                "split": "train",
                "boxes": [
                    {"xyxy": [10, 10, 110, 30], "approved": True, "boundary_evidence": "fixture"}
                ],
            }
        ],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    return path, review


def test_grounded_component_is_only_unapproved_proposal():
    image = np.full((100, 250, 3), 220, np.uint8)
    image[30:60, 30:210] = 10
    image[40:50, 85:155] = 230
    report = propose_panels(image, [[80, 35, 160, 55]])
    candidates = report["proposals"][0]["candidates"]
    assert candidates and candidates[0]["xyxy"] == [30, 30, 210, 60]
    assert report["training_eligible"] is False
    assert all(c["approved"] is False and c["semantic_confidence"] is None for c in candidates)


def test_no_visible_boundary_never_fabricates_padded_panel():
    image = np.full((80, 160, 3), 200, np.uint8)
    assert propose_panels(image, [[20, 20, 80, 40]])["proposals"][0]["candidates"] == []


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, 200, 50], [0, 0, 40, float("nan")]])
def test_invalid_geometry_rejected(box):
    with pytest.raises(ValueError):
        checked_box(box, 160, 80)


def test_corrupt_frame_and_resource_budget():
    with pytest.raises(ValueError):
        propose_panels(None, [])
    with pytest.raises(ValueError, match="Orçamento"):
        propose_panels(np.zeros((10, 10, 3), np.uint8), [[1, 1, 3, 3]] * 65)


@pytest.mark.parametrize("change", ["rights", "review", "proposal", "hash", "negative", "geometry"])
def test_export_refuses_unverified_data(tmp_path, change):
    path, review = review_fixture(tmp_path)
    r = review["records"][0]
    if change == "rights":
        r["rights_verified"] = False
    elif change == "review":
        r["reviewed"] = False
    elif change == "proposal":
        r["boxes"][0]["approved"] = False
    elif change == "hash":
        r["image_sha256"] = "wrong"
    elif change == "negative":
        r["boxes"] = []
    else:
        review["geometry_policy"] = "text_fragment"
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        export_reviewed(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_export_preserves_inputs_and_refuses_overwrite(tmp_path):
    path, _ = review_fixture(tmp_path)
    original = path.read_bytes()
    result = export_reviewed(path, tmp_path / "out")
    assert result["boxes"] == 1 and result["independent_human_review"] is False
    assert path.read_bytes() == original
    assert (tmp_path / "out/00000.txt").read_text().startswith("0 ")
    with pytest.raises(ValueError, match="existe"):
        export_reviewed(path, tmp_path / "out")


def test_explicit_splits_preserve_reserved_holdout(tmp_path):
    from test_dataset_metrics import dataset_manifest

    manifest = dataset_manifest(tmp_path)
    rows = list(csv.DictReader(manifest.open()))
    for i, row in enumerate(rows):
        row["split"] = "test" if i == 0 else "val" if i == 1 else "train"
    with manifest.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = prepare_dataset(manifest, tmp_path / "prepared", seed=777)
    assert report["records"][0]["split"] == "test"
    assert report["counts"] == {"test": 1, "val": 1, "train": 8}
    rows[2]["group_id"] = rows[0]["group_id"]
    with manifest.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Grupo cruza"):
        prepare_dataset(manifest, tmp_path / "bad")
