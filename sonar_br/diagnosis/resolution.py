"""Known-photo diagnostic: retain source pixels within unchanged 4K area safety budget."""

import json
from pathlib import Path

import numpy as np
from paddleocr import TextRecognition
from PIL import Image, ImageOps

from sonar_br.diagnosis.run import GenericBusDetector
from sonar_br.ocr_experiments.run import case
from sonar_vision.br.hierarchy import HierarchicalDetector
from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment
from sonar_vision.detection.yolo import YoloSignDetector
from sonar_vision.evaluation.final_photos import read_case, write_case
from sonar_vision.ocr.paddle import PaddleEngine
from sonar_vision.training.dataset import sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sonar_br/artifacts/v2-diagnosis/source3200"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ref_path = ROOT / "sonar_br/datasets/manifests/br-v2.review.json"
    binding = {
        "reference": sha256(ref_path),
        "runner": sha256(Path(__file__)),
        "cascade_source": sha256(ROOT / "src/sonar_vision/br/hierarchy.py"),
        "source_policy": (
            "original EXIF-oriented, max side3200, area remains<=3840*2160; "
            "reference boxes scaled by actual x/y factors; no threshold tuning"
        ),
        "bus": ".25/640/max8/crop5percent",
        "sign": ".35/640/NMS.5",
        "OCR": "unchanged E3",
        "scope": "known V2 diagnostic, NOT independent; isolates source pixel retention",
    }
    frozen = OUT / "binding.json"
    if frozen.exists() and json.loads(frozen.read_text()) != binding:
        raise ValueError("Preserve executed resolution diagnostic")
    frozen.write_text(json.dumps(binding, indent=2) + "\n")
    backend = YoloSignDetector(ROOT / "sonar_br/artifacts/train-v1/run/weights/best.pt")
    backend.model.overrides.update(device="cpu", iou=0.5, rect=True, quantize=None, batch=1)
    cascade = HierarchicalDetector(GenericBusDetector(), backend)
    rec = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    engine = ExperimentalOCR(PaddleEngine(), OCRExperiment(spatial="recognize"), rec)
    for index, original in enumerate(json.loads(ref_path.read_text())["records"]):
        if original["split"] != "test":
            continue
        r = dict(original)
        with Image.open(ROOT / "sonar_br/datasets/inbox/images" / r["filename"]) as source:
            im = ImageOps.exif_transpose(source).convert("RGB")
            im.thumbnail((3200, 3200))
            image = np.asarray(im)[:, :, ::-1].copy()
        sx, sy = image.shape[1] / r["width"], image.shape[0] / r["height"]
        r["boxes"] = [
            b
            | {"xyxy": [b["xyxy"][0] * sx, b["xyxy"][1] * sy, b["xyxy"][2] * sx, b["xyxy"][3] * sy]}
            for b in r["boxes"]
        ]
        r["scope"] = "known_diagnostic_source3200"
        for name in ["baseline", "hierarchy"] if index % 2 == 0 else ["hierarchy", "baseline"]:
            path = OUT / "cases" / f"{name}-{r['id']}.json"
            if read_case(path, sha256(frozen)) is not None:
                continue
            row = case(image, r, backend if name == "baseline" else cascade, engine, False)
            row["source_shape"] = list(image.shape)
            row["reference_scale"] = [sx, sy]
            row["bus_trace"] = cascade.trace if name == "hierarchy" else None
            write_case(path, row, sha256(frozen))
            print(name, r["id"], row["detection"], flush=True)


if __name__ == "__main__":
    main()
