"""E1b: Otsu and padding on archived model-predicted boxes, never manual boxes."""

import json
import time
from pathlib import Path

import cv2

from sonar_vision.core.contracts import Box
from sonar_vision.evaluation.screening import normalize_frame
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.roi import extract_roi

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    root = ROOT
    p = root / "reports/experiments/E1-automatic-roi-20261114.json"
    if p.exists():
        raise SystemExit("Experimento já persistido; não sobrescrever")
    s = json.loads((root / "reports/experiments/E1-20261114.json").read_text())
    engine = PaddleEngine()
    out = []
    for sid in ["D", "F"]:
        f = normalize_frame(cv2.imread(str(root / f"data/brazil-validation/{sid}.jpg")))
        rec = next(
            r
            for r in s["records"]
            if r["source"] == sid and r["kind"] == "detection" and r["confidence"] == 0.1
        )
        b = rec["boxes"][0]
        score = rec["scores"][0]
        for padding in [0.0, 0.5]:
            x1, y1, x2, y2 = b
            w = x2 - x1
            roi = extract_roi(
                f,
                Box(
                    x1=max(0, x1 - padding * w), y1=y1, x2=min(f.shape[1], x2 + padding * w), y2=y2
                ),
            )
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            start = time.perf_counter()
            readings = engine.read(image)
            cv2.imwrite(str(root / f"artifacts/research/e1-br/{sid}-auto-pad{padding}.png"), image)
            out.append(
                {
                    "source": sid,
                    "selection": (
                        "top scored predicted box at conf=.10; no ground-truth coordinates used"
                    ),
                    "padding_fraction_each_side": padding,
                    "detector_confidence": score,
                    "ocr": [r.model_dump() for r in readings],
                    "ocr_ms": (time.perf_counter() - start) * 1000,
                    "confirmation_eligible_at_original_min_detection_035": score >= 0.35,
                }
            )
    p.write_text(
        json.dumps(
            {
                "scope": (
                    "automatic ROI diagnostics on inspected development samples; "
                    "Otsu fixed from E1; only padding varied; no temporal confirmation"
                ),
                "records": out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(out)


if __name__ == "__main__":
    main()
