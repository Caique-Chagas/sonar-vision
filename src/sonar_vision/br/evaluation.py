"""One detector protocol shared across both domains and both trained models."""

import json
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from sonar_vision.training.dataset import sha256, verify_dataset
from sonar_vision.training.runner import hardware


class EvaluationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    imgsz: int = Field(default=640, ge=320, le=1920)
    device: str = "cpu"
    metrics_conf: float = Field(default=0.001, gt=0, lt=1)
    operating_conf: float = Field(default=0.35, gt=0, lt=1)
    nms_iou: float = Field(default=0.5, gt=0, lt=1)
    max_det: int = Field(default=16, ge=1, le=64)


def evaluate_model(
    weights: Path, data: Path, output: Path, protocol: EvaluationProtocol, *, split: str = "test"
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Evaluation output already exists; preserve completed evidence")
    if split not in {"val", "test"}:
        raise ValueError("No training images in generalization evaluation")
    audit = verify_dataset(data)
    import ultralytics

    output.mkdir(parents=True)
    model = cast(Any, ultralytics).YOLO(str(weights.resolve(strict=True)))
    parameters = {
        "device": protocol.device,
        "imgsz": protocol.imgsz,
        "iou": protocol.nms_iou,
        "max_det": protocol.max_det,
        "batch": 1,
        "workers": 0,
        "rect": True,
        "augment": False,
        "plots": False,
        "verbose": False,
    }
    metrics = model.val(
        data=str(data.resolve()),
        split=split,
        conf=protocol.metrics_conf,
        project=str(output.resolve()),
        name="metrics",
        **parameters,
    )
    result = {
        "model_sha256": sha256(weights),
        "dataset_audit_sha256": sha256(data.parent / "audit.json"),
        "protocol": protocol.model_dump(),
        "split": split,
        "images": audit["counts"][split],
        "panels": sum(r["boxes"] for r in audit["records"] if r["split"] == split),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "speed_ms": metrics.speed,
        "hardware": hardware(),
        "temporal_validation": False,
        "metric_scope": "detector AP sweep, not fixed-threshold OCR or encounter success",
    }
    (output / "detector.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_same_dataset(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    for key in ["dataset_audit_sha256", "protocol", "split", "images", "panels"]:
        if left[key] != right[key]:
            raise ValueError(f"Incompatible comparison: {key}")
    return {
        "left_model_sha256": left["model_sha256"],
        "right_model_sha256": right["model_sha256"],
        "dataset_audit_sha256": left["dataset_audit_sha256"],
        "split": left["split"],
        "images": left["images"],
        "panels": left["panels"],
        "right_minus_left": {
            metric: right[metric] - left[metric]
            for metric in ["precision", "recall", "map50", "map50_95"]
        },
        "promotion": "Not automatic: also require OCR, false positives and temporal evidence",
    }
