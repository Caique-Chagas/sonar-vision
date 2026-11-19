"""Verify and export all thirty recorded cells without initializing ML models."""

import csv
import json

from sonar_br.source_import.build import OUTPUT, ROOT
from sonar_br.source_import.compare import OUT, summarize
from sonar_vision.evaluation.final_photos import read_case
from sonar_vision.training.dataset import sha256
from sonar_vision.training.recognition_dataset import verify_recognition


def main() -> None:
    refs = verify_recognition(OUTPUT / "ocr")["records"]
    ids = {r["parent_id"] for r in refs}
    binding = sha256(OUT / "binding.json")
    saved = json.loads((ROOT / "sonar_br/reports/SOURCE_IMPORT_OCR.json").read_text())
    lines = []
    for folder, key, names in [
        (
            "code-crops",
            "code_crop_DEV",
            ["original", "E3", "en_PP-OCRv5_mobile_rec", "en_PP-OCRv4_mobile_rec"],
        ),
        ("predicted-panels", "V1_predicted_panel_DEV", ["original", "E3"]),
    ]:
        for name in names:
            rows = []
            for id in sorted(ids):
                row = read_case(OUT / folder / f"{name}-{id}.json", binding)
                if row is None:
                    raise ValueError("Missing example; never shrink the denominator")
                rows.append(row)
                lines.append(
                    {
                        "stage": folder,
                        "variant": name,
                        "id": id,
                        "truth": row["truth"],
                        "prediction": row["prediction"],
                        "exact": row["exact"],
                        "OCR_ms": row["ms"],
                        "scope": row["scope"],
                        "readings_or_ROIs": json.dumps(
                            row.get("readings", row.get("predicted_ROIs")), ensure_ascii=False
                        ),
                    }
                )
            if summarize(rows, ids) != saved[key][name]:
                raise ValueError("Summary does not match all recorded cases")
    if len(lines) != 30 or saved["independent_test_n"] != 0:
        raise ValueError("Invalid closed experiment")
    dest = ROOT / "sonar_br/reports/source-import-OCR-cases.csv"
    with dest.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(lines[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)
    print("30cells verified; independent test remains0; no new inference")


if __name__ == "__main__":
    main()
