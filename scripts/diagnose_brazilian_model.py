"""E1 controlled diagnostics on already-inspected development photos, not a test benchmark."""

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from sonar_vision.core.contracts import Box
from sonar_vision.evaluation.screening import normalize_frame
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.roi import extract_roi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research/e1-br"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    destination = ROOT / "reports/experiments/E1-20261114.json"
    if destination.exists():
        raise SystemExit("Experimento já persistido; não repetir nem sobrescrever")
    from ultralytics import YOLO

    weights = ROOT / "models/best.pt"
    model = YOLO(str(weights))
    ocr = PaddleEngine()
    results = []
    for source in ["D", "F"]:
        frame = normalize_frame(cv2.imread(str(ROOT / f"data/brazil-validation/{source}.jpg")))
        # Ground-truth crops inspected independently before OCR; diagnostics only.
        box = [1035, 477, 1625, 589] if source == "D" else [930, 100, 1695, 240]
        reference = extract_roi(frame, Box(**dict(zip(["x1", "y1", "x2", "y2"], box, strict=True))))
        cv2.imwrite(str(OUT / f"{source}-reference.png"), reference)
        # Cached E0 and D1280 are retained in previous reports. Only new comparisons run here.
        for confidence in [0.10, 0.05]:
            start = time.perf_counter()
            prediction = model.predict(
                frame, conf=confidence, imgsz=640, classes=[0], max_det=16, verbose=False
            )[0]
            results.append(
                {
                    "source": source,
                    "kind": "detection",
                    "variable": "conf",
                    "imgsz": 640,
                    "confidence": confidence,
                    "boxes": prediction.boxes.xyxy.cpu().tolist(),
                    "scores": prediction.boxes.conf.cpu().tolist(),
                    "ms": (time.perf_counter() - start) * 1000,
                }
            )
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants = {
            "gray": gray,
            "otsu": binary,
            "otsu_close2": cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)),
        }
        for name, image in variants.items():
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            start = time.perf_counter()
            readings = ocr.read(image)
            cv2.imwrite(str(OUT / f"{source}-{name}.png"), image)
            results.append(
                {
                    "source": source,
                    "kind": "manual_reference_ocr_not_e2e",
                    "preprocessing": name,
                    "crop_size": list(image.shape[:2]),
                    "readings": [r.model_dump() for r in readings],
                    "ms": (time.perf_counter() - start) * 1000,
                }
            )
    result = {
        "experiment": "E1-20261114",
        "split": "already-inspected development diagnostics",
        "hypotheses": [
            "confidence threshold suppresses candidates",
            "LED dot pattern/banding affects OCR",
        ],
        "original_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "ultralytics": importlib.metadata.version("ultralytics"),
        "python": platform.python_version(),
        "model_classes": model.names,
        "parameter_device": str(next(model.model.parameters()).device),
        "effective_nms_iou": model.predictor.args.iou,
        "effective_imgsz": model.predictor.args.imgsz,
        "records": results,
        "independent_test": False,
        "training_performed": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
