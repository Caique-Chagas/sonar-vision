"""Select on DEV only; compare one planned combination using recorded native OCR."""

import json
from statistics import median
from typing import Any

import cv2

from sonar_br.ocr_experiments.run import REF, ROOT, SavedDetector, SavedOCR, case
from sonar_vision.core.contracts import Box, Detection
from sonar_vision.training.dataset import sha256


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coded = [r for r in rows if r["expected_code"]]
    return {
        "codes": len(coded),
        "raw_exact": sum(bool(r["raw_exact"]) for r in coded),
        "eligible_exact": sum(bool(r["eligible_exact"]) for r in coded),
        "wrong_eligible": sum(len(r["wrong_eligible_codes"]) for r in rows),
        "negative_eligible": sum(len(r["negative_eligible_codes"]) for r in rows),
        "negative_FP_boxes": sum(r["detection"]["fp"] for r in rows if r["negative"]),
        "OCR_median_ms": median(r["OCR_ms"] for r in rows if not r["first_photo"]),
        "cached_OCR": all(r["cached_OCR"] for r in rows),
        "pipeline_median_ms": median(r["pipeline_ms"] for r in rows if not r["first_photo"]),
    }


def main() -> None:
    folder = ROOT / "sonar_br/ocr_experiments"
    output = folder / "candidate-frozen.json"
    if output.exists():
        raise ValueError("Candidate already frozen; do not select again")
    dev = json.loads((ROOT / "sonar_br/reports/ocr-opt-dev.json").read_text())
    reference = {r["id"]: r for r in json.loads(REF.read_text())["records"] if r["split"] == "dev"}
    design = json.loads((folder / "design.json").read_text())
    rows = dev["records"]
    comparison = {v: aggregate([r for r in rows if r["variant"] == v]) for v in design["variants"]}
    safe = [
        v
        for v, m in comparison.items()
        if m["wrong_eligible"] == 0
        and m["negative_eligible"] == 0
        and v != "E4"
        and m["OCR_median_ms"] <= 3 * comparison["E0"]["OCR_median_ms"]
    ]
    # Ordered design gives the simpler baseline priority for a tie.
    winner = max(safe, key=lambda v: comparison[v]["eligible_exact"])
    combo = []
    for row in rows:
        if row["variant"] != winner:
            continue
        ref = reference[row["id"]]
        image = cv2.imread(str((REF.parent / ref["image"]).resolve()))
        backend = SavedDetector(
            [
                Detection(box=Box(**p["box"]), confidence=p["confidence"], label="destination_sign")
                for p in row["predictions"]
            ],
            "frozen_DEV_detections",
        )
        value = case(image, ref, backend, SavedOCR(row["predictions"]), True)
        value.update(variant="winner+E4", first_photo=row["first_photo"], cached_OCR=True)
        combo.append(value)
    comparison["winner+E4"] = aggregate(combo)
    # Latency of combination is not inferred from cache-only timing.
    accept_combo = (
        comparison["winner+E4"]["wrong_eligible"] == 0
        and comparison["winner+E4"]["negative_eligible"] == 0
        and comparison["winner+E4"]["eligible_exact"] > comparison[winner]["eligible_exact"]
    )
    configuration = dict(design["variants"][winner])
    if accept_combo:
        configuration["separators"] = True
    proof = {
        "comparison": comparison,
        "winner_OCR": winner,
        "combo_selected": accept_combo,
        "combo_records": combo,
        "selection_scope": "DEV only; no holdout models run",
        "estimated_latency_note": (
            "DEV variants share cached YOLO; E4/combo reuse OCR. "
            "Their raw wall time cannot represent end-to-end latency."
        ),
    }
    (ROOT / "sonar_br/reports/ocr-opt-selection.json").write_text(
        json.dumps(proof, indent=2, ensure_ascii=False) + "\n"
    )
    protocol = json.loads((ROOT / "sonar_br/configs/final50-protocol.json").read_text())
    for path, digest in protocol["source_hashes"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Original production source changed")
    candidate = {
        "status": "frozen_before_holdout_model_calls",
        "configuration": configuration,
        "source_sha256": sha256(ROOT / "src/sonar_vision/br/ocr_optimization.py"),
        "runner_sha256": sha256(folder / "run.py"),
        "reference_sha256": sha256(REF),
        "DEV_selection_sha256": sha256(ROOT / "sonar_br/reports/ocr-opt-selection.json"),
        "protected": design["protected"],
        "production_source_hashes": protocol["source_hashes"],
        "OCR_artifacts": protocol["ocr_artifacts"],
        "hardware": protocol["hardware"],
        "dependencies": json.loads(
            (ROOT / "sonar_br/reports/final50-environment.json").read_text()
        ),
        "no_YOLO_training_no_validator_changes": True,
        "experimental_no_default_promotion": True,
    }
    output.write_text(json.dumps(candidate, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    print("Frozen", winner, "E4", accept_combo)


if __name__ == "__main__":
    main()
