import json
import time
from pathlib import Path

import cv2

from sonar_vision.core.contracts import Box
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.processing.roi import extract_roi
from sonar_vision.validation.normalizer import clean_code

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    annotations = json.loads((ROOT / "reports/ocr-groundtruth.json").read_text())
    crops = {
        r["id"]: r for r in json.loads((ROOT / "artifacts/research/ocr-crop-map.json").read_text())
    }
    engine = PaddleEngine()
    records = []
    for crop_id, expected in annotations["included"].items():
        crop = crops[crop_id]
        frame = cv2.imread(str(ROOT / "data/bangkok-pilot" / crop["image"]))
        roi = extract_roi(frame, Box(**crop["box"]))
        start = time.perf_counter()
        readings = engine.read(roi)
        elapsed = time.perf_counter() - start
        # Full display comparison; no best-of-many candidate selection against ground truth.
        actual = "".join(reading.text for reading in readings)
        records.append(
            {
                "crop_id": crop_id,
                "expected": expected,
                "actual": actual,
                "readings": [r.model_dump() for r in readings],
                "latency_ms": elapsed * 1000,
                "exact_match": clean_code(actual) == clean_code(expected),
            }
        )
    result = {
        "scope": annotations["scope"],
        "annotation_method": annotations["annotation_method"],
        "preprocessing": "original crop, resize height up to 64px, no CLAHE",
        "normalization": "Unicode NFKC, uppercase, strip whitespace, normalize dash",
        "total_crops": annotations["total_crops"],
        "excluded_illegible": len(annotations["excluded"]),
        "evaluated": len(records),
        "exact_matches": sum(r["exact_match"] for r in records),
        "exact_accuracy_readable_subset": sum(r["exact_match"] for r in records) / len(records),
        "records": records,
    }
    output = ROOT / "reports/evaluation/ocr_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
