import csv

import cv2
import numpy as np
import pytest

from sonar_vision.evaluation.metrics import route_metrics
from sonar_vision.training.dataset import prepare_dataset, validate_labels, verify_dataset


def dataset_manifest(root, identical=False):
    rng = np.random.default_rng(42)
    rows = []
    shared = rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)
    for i in range(10):
        image = shared if identical else rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(root / f"{i}.png"), image)
        (root / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        rows.append(
            dict(
                image=f"{i}.png",
                label=f"{i}.txt",
                group_id=f"sequence-{i}",
                source="https://example.org/licensed-data",
                license="CC0",
            )
        )
    path = root / "manifest.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return path


def test_grouped_split_reproducible_audited_and_tampering_detected(tmp_path):
    manifest = dataset_manifest(tmp_path)
    output = tmp_path / "prepared"
    a = prepare_dataset(manifest, output)
    b = prepare_dataset(manifest, tmp_path / "second")
    assert a["counts"] == b["counts"] == {"train": 6, "val": 2, "test": 2}
    assert [r["split"] for r in a["records"]] == [r["split"] for r in b["records"]]
    assert verify_dataset(output / "data.yaml")["images"] == 10
    first = a["records"][0]["prepared_image"]
    (output / first).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="alterado"):
        verify_dataset(output / "data.yaml")


def test_leakage_rejected(tmp_path):
    with pytest.raises(ValueError, match="Vazamento"):
        prepare_dataset(dataset_manifest(tmp_path, identical=True), tmp_path / "prepared")


@pytest.mark.parametrize(
    "label",
    ["0 0.1 0.1 0.9 0.9", "2 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1", "0 0.5 0.5 -0.1 0.1", "0 0.5"],
)
def test_bad_labels_rejected(tmp_path, label):
    path = tmp_path / "label.txt"
    path.write_text(label)
    with pytest.raises(ValueError):
        validate_labels(path, 1)


def test_metrics_denominators_and_missing_negatives():
    result = route_metrics(
        [
            {"encounter_id": "a", "expected": "971D-10", "predicted": "971D-10"},
            {"encounter_id": "b", "expected": "971D-10", "predicted": "875P-10"},
            {"encounter_id": "c", "expected": None, "predicted": None},
        ]
    )
    assert result["route_accuracy"] == 0.5
    assert result["wrong_confirmation_rate_all_encounters"] == pytest.approx(1 / 3)
    assert result["correct_rejection_rate_negative_encounters"] == 1
    assert (
        route_metrics([{"encounter_id": "a", "expected": "a", "predicted": None}])[
            "false_positive_rate_negative_encounters"
        ]
        is None
    )
