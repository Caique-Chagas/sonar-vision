"""Paired static pipeline and reference-ROI OCR; resumable without repeated cases."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2

from sonar_vision.br.observations import (
    RecordedDetector,
    RecordedOCR,
    code_visible_in_ocr,
    observation_routes,
)
from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import Box
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.evaluation.prospective import match_panels
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.pipeline import Context, FrameProcessor
from sonar_vision.processing.roi import extract_roi
from sonar_vision.tracking.iou import iou
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "sonar_br/datasets/manifests/commons-br-v2.review.json"
LOCAL = ROOT / "sonar_br/artifacts/pipeline-cases"


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checked_image(reference: dict[str, Any]) -> Any:
    path = (REFERENCE.parent / reference["image"]).resolve()
    if sha256(path) != reference["image_sha256"]:
        raise ValueError("Reference image changed")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Reference image is corrupt")
    return image


def reference_boxes(reference: dict[str, Any]) -> list[Box]:
    return [
        Box(**dict(zip(["x1", "y1", "x2", "y2"], b["xyxy"], strict=True)))
        for b in reference["boxes"]
    ]


def cached_case(path: Path, freeze_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    saved: dict[str, Any] = json.loads(path.read_text())
    if saved["freeze_sha256"] != freeze_hash:
        raise ValueError("Cached case differs from frozen inputs")
    return saved


def predictions_with_iou(
    detector: RecordedDetector,
    ocr: RecordedOCR,
    boxes: list[Box],
    expected: str | None,
) -> list[dict[str, Any]]:
    predictions = []
    for detection, call in zip(detector.detections, ocr.calls, strict=True):
        text = " | ".join(r["text"] for r in call["readings"])
        predictions.append(
            {
                "box": detection.box.model_dump(),
                "confidence": detection.confidence,
                "reference_ious": [iou(detection.box, box) for box in boxes],
                **call,
                "visible_true_code_in_OCR": code_visible_in_ocr(text, expected)
                if expected
                else None,
            }
        )
    return predictions


def evaluate_case(
    ref: dict[str, Any],
    model_name: str,
    backend: YoloSignDetector,
    engine: PaddleEngine,
    freeze_hash: str,
) -> dict[str, Any]:
    detector, ocr = RecordedDetector(backend), RecordedOCR(engine)
    processor = FrameProcessor(
        detector,
        ocr,
        Context(ref.get("expected_code") or "6030-10", "static-test", ref["id"], "local_video"),
        Settings(_env_file=None, roi_padding=0),
    )
    image = checked_image(ref)
    start = time.perf_counter()
    _, events = processor.process(image, 0, 0.0)
    wall_ms = (time.perf_counter() - start) * 1000
    assert not any(event.status == "CONFIRMED" for event in events)
    boxes = reference_boxes(ref)
    predictions = predictions_with_iou(detector, ocr, boxes, ref.get("expected_code"))
    return {
        "id": ref["id"],
        "model": model_name,
        "expected_code": ref.get("expected_code"),
        "reference_boxes": [box.model_dump() for box in boxes],
        "predictions": predictions,
        "detection": match_panels(predictions, boxes, []),
        "runtime_normalizations": observation_routes(processor),
        "events": [event.model_dump(mode="json") for event in events],
        "final_status": "UNCERTAIN",
        "actual_source": "real photograph, not video",
        "pipeline_wall_ms": wall_ms,
        "detector_wall_ms": detector.wall_ms,
        "freeze_sha256": freeze_hash,
    }


def reference_ocr(ref: dict[str, Any], engine: PaddleEngine, freeze_hash: str) -> dict[str, Any]:
    image, readings = checked_image(ref), []
    for box in reference_boxes(ref):
        readings.extend(r.model_dump() for r in engine.read(extract_roi(image, box)))
    return {
        "id": ref["id"],
        "expected_code": ref.get("expected_code"),
        "readings": readings,
        "manual_reference_ROI_not_detector_success": True,
        "freeze_sha256": freeze_hash,
        "exact_code": code_visible_in_ocr(
            " | ".join(r["text"] for r in readings), ref["expected_code"]
        )
        if ref.get("expected_code")
        else None,
    }


def freeze_inputs(models: dict[str, Path]) -> tuple[dict[str, Any], str]:
    protocol = json.loads((ROOT / "sonar_br/configs/experiment-v1.json").read_text())
    if sha256(REFERENCE) != protocol["BR_review_sha256"]:
        raise ValueError("Frozen reference changed")
    freeze = {
        "reference_sha256": sha256(REFERENCE),
        "model_hashes": {key: sha256(path) for key, path in models.items()},
        "script_sha256": sha256(Path(__file__)),
        "observer_sha256": sha256(ROOT / "src/sonar_vision/br/observations.py"),
        "ROI": "same existing default resize, padding0, no tiles, CPU640/conf.35/NMS.5",
        "OCR": "existing Paddle OCR frozen in experiment-v1.json",
        "single_frame_only": True,
        "GT_not_used_to_select_predictions_or_crops": True,
        "requested_context": "known full code as user request; no other GT catalog",
        "pipeline_time_scope": (
            "excludes hash check/decode and model initialization; detector time separate"
        ),
    }
    path = ROOT / "sonar_br/configs/pipeline-test-frozen.json"
    if path.exists() and json.loads(path.read_text()) != freeze:
        raise ValueError("Previous attempt differs from frozen inputs; preserve it")
    if not path.exists():
        save_json(path, freeze)
    return freeze, sha256(path)


def run_model_cases(
    name: str,
    path: Path,
    refs: list[dict[str, Any]],
    engine: PaddleEngine,
    freeze_hash: str,
) -> list[dict[str, Any]]:
    backend = YoloSignDetector(path)
    backend.model.overrides.update(device="cpu", iou=0.5, rect=True, quantize=None, batch=1)
    rows, first_actual_call = [], True
    for ref in refs:
        case_path = LOCAL / f"{name}-{ref['id']}.json"
        row = cached_case(case_path, freeze_hash)
        if row is None:
            row = evaluate_case(ref, name, backend, engine, freeze_hash)
            row["cold_first_detector_call"] = first_actual_call
            first_actual_call = False
            save_json(case_path, row)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--br-weights", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / "sonar_br/reports/pipeline-test.json"
    if output.exists():
        parser.error("Completed comparison exists; do not rerun or overwrite")
    refs = [r for r in json.loads(REFERENCE.read_text())["records"] if r["split"] == "test"]
    models = {"Bangkok": ROOT / "models/best.pt", "BR": args.br_weights}
    freeze, freeze_hash = freeze_inputs(models)
    engine, records, manual = PaddleEngine(), [], []
    for name, path in models.items():
        records.extend(run_model_cases(name, path, refs, engine, freeze_hash))
    for ref in refs:
        path = LOCAL / f"manual-{ref['id']}.json"
        row = cached_case(path, freeze_hash)
        if row is None:
            row = reference_ocr(ref, engine, freeze_hash)
            save_json(path, row)
        manual.append(row)
    save_json(
        output,
        {
            "frozen": freeze,
            "records": records,
            "reference_ROI_OCR": manual,
            "temporal_validation": False,
            "single_frame_confirmations": 0,
        },
    )


if __name__ == "__main__":
    main()
