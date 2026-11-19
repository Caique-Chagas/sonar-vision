"""Synthetic export fixtures verify safeguards, not dataset suitability or model accuracy."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from sonar_vision.training.dataset import prepare_dataset
from sonar_vision.training.roboflow_audit import audit_export, convert_reviewed

URL = "https://universe.roboflow.com/cakoworspace/deteccao-letreiro-onibus-gqjt2"


def export_fixture(tmp_path, duplicate=False):
    root = tmp_path / "export"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["DESTINO", "OUTRO"], "nc": 2}))
    rng = np.random.default_rng(40)
    shared = rng.integers(0, 256, (80, 100, 3), np.uint8)
    for i, split in enumerate(["train", "valid", "test"]):
        for folder in ["images", "labels"]:
            (root / split / folder).mkdir(parents=True)
        image = shared if duplicate else rng.integers(0, 256, (80, 100, 3), np.uint8)
        cv2.imwrite(str(root / split / "images" / f"bus{i}.png"), image)
        (root / split / "labels" / f"bus{i}.txt").write_text("1 0.5 0.4 0.6 0.1\n")
    return root


def review_fixture(root, path):
    records = audit_export(root)["records"]
    review = {
        "source_url": URL,
        "license": "CC BY 4.0",
        "rights_reviewed": True,
        "rights_evidence": "synthetic fixture owns generated pixels",
        "images": [],
    }
    for i, r in enumerate(records):
        review["images"].append(
            {
                **{k: r[k] for k in ["image", "image_sha256", "label_sha256"]},
                "reviewed": True,
                "group_id": f"sequence-{i}",
                "group_evidence": "fixture source ID",
                "boxes": [{"index": 0, "geometry": "whole_destination_sign"}],
            }
        )
    path.write_text(json.dumps(review))
    return review


def test_audit_and_explicit_conversion_preserve_provenance(tmp_path):
    root = export_fixture(tmp_path)
    review = tmp_path / "review.json"
    review_fixture(root, review)
    audit = audit_export(root)
    assert audit["images"] == 3 and audit["class_box_counts"] == {"OUTRO": 3}
    assert not audit["geometry_reviewed"] and not audit["training_eligible"]
    manifest = convert_reviewed(root, review, tmp_path / "converted")
    receipt = json.loads((manifest.parent / "conversion-audit.json").read_text())
    assert receipt["records"][0]["selected_original_boxes"][0]["class_id"] == 1
    assert (manifest.parent / "00000.txt").read_text().startswith("0 ")
    prepared = prepare_dataset(manifest, tmp_path / "prepared")
    assert prepared["images"] == 3
    with pytest.raises(ValueError, match="existe"):
        convert_reviewed(root, review, manifest.parent)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewed", False),
        ("reviewed", "false"),
        ("group_id", ""),
        ("group_id", "   "),
        ("group_evidence", ""),
        ("image_sha256", "wrong"),
    ],
)
def test_unreviewed_or_mutated_data_cannot_convert(tmp_path, field, value):
    root = export_fixture(tmp_path)
    p = tmp_path / "review.json"
    r = review_fixture(root, p)
    r["images"][0][field] = value
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        convert_reviewed(root, p, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("geometry", ["word", "fragment", "unknown"])
def test_semantic_destinations_do_not_prove_whole_sign_boxes(tmp_path, geometry):
    root = export_fixture(tmp_path)
    p = tmp_path / "review.json"
    r = review_fixture(root, p)
    r["images"][0]["boxes"][0]["geometry"] = geometry
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="fragmento"):
        convert_reviewed(root, p, tmp_path / "out")


def test_duplicate_leakage_reported_and_conversion_groups_enforced(tmp_path):
    root = export_fixture(tmp_path, duplicate=True)
    a = audit_export(root)
    assert a["cross_split_pairs"] == 3
    p = tmp_path / "review.json"
    review_fixture(root, p)
    with pytest.raises(ValueError, match="compartilhar grupo"):
        convert_reviewed(root, p, tmp_path / "out")


@pytest.mark.parametrize(
    "label", ["3 0.5 0.5 0.2 0.1", "0 0.9 0.5 0.8 0.1", "0 nan 0.5 0.2 0.1", "0 0.5"]
)
def test_unknown_class_or_invalid_box(tmp_path, label):
    root = export_fixture(tmp_path)
    (root / "train/labels/bus0.txt").write_text(label)
    with pytest.raises(ValueError):
        audit_export(root)


def test_missing_label_is_not_a_negative(tmp_path):
    root = export_fixture(tmp_path)
    (root / "train/labels/bus0.txt").unlink()
    with pytest.raises(ValueError, match="Label ausente"):
        audit_export(root)


def test_missing_api_key_is_sanitized(tmp_path, monkeypatch, capsys):
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts/download_roboflow.py"
    spec = importlib.util.spec_from_file_location("rf_export_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_dotenv", lambda: False)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "Configure ROBOFLOW_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "script,receipt",
    [
        ("diagnose_brazilian_model.py", "E1-20261114.json"),
        ("diagnose_brazilian_roi.py", "E1-automatic-roi-20261114.json"),
    ],
)
def test_completed_experiment_cannot_be_overwritten(tmp_path, monkeypatch, script, receipt):
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / script
    spec = importlib.util.spec_from_file_location("diagnostic_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    p = tmp_path / "reports/experiments" / receipt
    p.parent.mkdir(parents=True)
    p.write_text("immutable checkpoint")
    with pytest.raises(SystemExit):
        module.main()
    assert p.read_text() == "immutable checkpoint"


def test_empty_label_requires_explicit_negative_review(tmp_path):
    root = export_fixture(tmp_path)
    (root / "train/labels/bus0.txt").write_text("")
    assert audit_export(root)["negative_images"] == 1
    p = tmp_path / "review.json"
    review = review_fixture(root, p)
    item = next(r for r in review["images"] if r["image"].startswith("train/"))
    item["boxes"] = []
    item["confirmed_negative"] = True
    p.write_text(json.dumps(review))
    manifest = convert_reviewed(root, p, tmp_path / "out")
    assert any(not x.read_text() for x in manifest.parent.glob("*.txt"))


@pytest.mark.parametrize("index", [-1, 9])
def test_invalid_review_box_indices_rejected(tmp_path, index):
    root = export_fixture(tmp_path)
    p = tmp_path / "review.json"
    review = review_fixture(root, p)
    review["images"][0]["boxes"][0]["index"] = index
    p.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="Índice"):
        convert_reviewed(root, p, tmp_path / "out")


def test_polygon_vertices_preserved_and_only_reviewed_envelope_converted(tmp_path):
    root = export_fixture(tmp_path)
    label = root / "train/labels/bus0.txt"
    label.write_text("1 0.1 0.2 0.8 0.21 0.79 0.4 0.12 0.39\n")
    a = audit_export(root)
    assert a["annotation_geometry_counts"] == {"polygon": 1, "box": 2}
    polygon = next(r for r in a["records"] if r["split"] == "train")["boxes"][0]
    assert len(polygon["source_polygon"]) == 4 and polygon["source_geometry"] == "polygon"
    p = tmp_path / "review.json"
    review_fixture(root, p)
    manifest = convert_reviewed(root, p, tmp_path / "converted")
    receipt = json.loads((manifest.parent / "conversion-audit.json").read_text())
    assert any(
        b["source_polygon"] for r in receipt["records"] for b in r["selected_original_boxes"]
    )


@pytest.mark.parametrize(
    "annotation",
    [
        "1 0.1 0.2 0.3 0.4 0.5 0.6 0.7",
        "1 0.1 0.2 1.4 0.2 0.8 0.4 0.1 0.4",
        "1 0.1 0.2 0.2 0.2 0.3 0.2",
        "1 nan 0.2 0.8 0.2 0.8 0.4 0.1 0.4",
    ],
)
def test_malformed_polygon_rejected(tmp_path, annotation):
    root = export_fixture(tmp_path)
    (root / "train/labels/bus0.txt").write_text(annotation)
    with pytest.raises(ValueError):
        audit_export(root)


def test_declared_license_does_not_replace_source_rights_review(tmp_path):
    root = export_fixture(tmp_path)
    p = tmp_path / "review.json"
    r = review_fixture(root, p)
    r["rights_reviewed"] = False
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="direitos"):
        convert_reviewed(root, p, tmp_path / "out")
