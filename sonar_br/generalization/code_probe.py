"""Diagnose internal code localization using explicitly manual DEV code crops."""

import json
import time
from pathlib import Path

from paddleocr import TextRecognition

from sonar_vision.evaluation.final_photos import read_case, write_case
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "sonar_br/datasets/manifests/generalization-20261116.ocr.json"
OUT = ROOT / "sonar_br/artifacts/generalization-20261116/code-probe"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    model = Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"
    if not model.exists():
        raise ValueError("Cached recognizer required; no implicit download")
    binding = {
        "reference": sha256(REF),
        "runner": sha256(Path(__file__)),
        "model_files": {f.name: sha256(f) for f in model.iterdir() if f.is_file()},
        "scope": "manual code crop DEV only; NOT automatic pipeline or generalization",
    }
    saved = OUT / "binding.json"
    if saved.exists() and json.loads(saved.read_text()) != binding:
        raise ValueError("Preserve probe")
    saved.write_text(json.dumps(binding, indent=2) + "\n")
    engine = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(model),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    for ref in json.loads(REF.read_text())["records"]:
        if ref["split"] != "development_only" or ref["id"] == "35046766":
            raise ValueError("No test crops")
        path = OUT / (ref["id"] + ".json")
        if read_case(path, sha256(saved)) is not None:
            continue
        if sha256(Path(ref["image"])) != ref["crop_sha256"]:
            raise ValueError("Crop changed")
        start = time.perf_counter()
        pred = list(engine.predict(input=ref["image"], batch_size=1))
        ms = (time.perf_counter() - start) * 1000
        texts = [{"text": str(r["rec_text"]), "confidence": float(r["rec_score"])} for r in pred]
        row = {
            "id": ref["id"],
            "expected_code": ref["expected_code"],
            "readings": texts,
            "exact": any(
                r["text"] == ref["expected_code"] and r["confidence"] >= 0.7 for r in texts
            ),
            "ms": ms,
            "scope": "manual_CODE_crop_DEV_only_not_automatic_ROI",
        }
        write_case(path, row, sha256(saved))
        print(ref["id"], texts, flush=True)


if __name__ == "__main__":
    main()
