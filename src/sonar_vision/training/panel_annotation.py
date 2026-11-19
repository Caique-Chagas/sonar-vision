"""Image-grounded proposals are NOT ground truth. Export requires a separate review."""

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sonar_vision.training.dataset import sha256


def checked_box(values: Any, width: int, height: int) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("Esperadas quatro coordenadas xyxy")
    box = [float(v) for v in values]
    x1, y1, x2, y2 = box
    if not all(math.isfinite(v) for v in box) or not (0 <= x1 < x2 <= width):
        raise ValueError("Caixa inválida na largura")
    if not 0 <= y1 < y2 <= height:
        raise ValueError("Caixa inválida na altura")
    return box


def propose_panels(image: np.ndarray[Any, Any], seeds: list[list[float]]) -> dict[str, Any]:
    """Search local dark connected components around text seeds, without padding GT.

    A component must contain the seed and extend it measurably. Windshield/roof
    components often fail the aspect/height gates. Surviving components still need
    visual correction: this heuristic has no calibrated semantic confidence.
    """
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Imagem BGR uint8 necessária")
    height, width = image.shape[:2]
    if width * height > 20_000_000 or len(seeds) > 64:
        raise ValueError("Orçamento de pré-anotação excedido")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    proposals = []
    for index, seed in enumerate(seeds):
        x1, y1, x2, y2 = checked_box(seed, width, height)
        sw, sh = x2 - x1, y2 - y1
        lx, ly = max(0, int(x1 - 2 * sw)), max(0, int(y1 - 1.5 * sh))
        rx, ry = min(width, math.ceil(x2 + 2 * sw)), min(height, math.ceil(y2 + 1.5 * sh))
        local = gray[ly:ry, lx:rx]
        candidates: list[dict[str, Any]] = []
        for threshold in (45, 75, 105):
            mask: np.ndarray[Any, Any] = (local < threshold).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                ax, ay, bx, by = lx + cx, ly + cy, lx + cx + cw, ly + cy + ch
                # A clipped local component gives no evidence of a panel boundary.
                if cx == 0 or cy == 0 or cx + cw == local.shape[1] or cy + ch == local.shape[0]:
                    continue
                if not (ax <= x1 and ay <= y1 and bx >= x2 and by >= y2):
                    continue
                expansion = cw * ch / (sw * sh)
                if not (1.08 <= expansion <= 6 and 2 <= cw / ch <= 18 and ch <= 3 * sh):
                    continue
                candidate = {
                    "xyxy": [ax, ay, bx, by],
                    "method": "dark_component_contains_text_seed",
                    "threshold": threshold,
                    "seed_area_ratio": expansion,
                    "approved": False,
                    "semantic_confidence": None,
                }
                if not any(c["xyxy"] == candidate["xyxy"] for c in candidates):
                    candidates.append(candidate)
        candidates.sort(key=lambda c: c["seed_area_ratio"])
        proposals.append({"seed_index": index, "seed_xyxy": seed, "candidates": candidates[:3]})
    return {"schema_version": "1", "training_eligible": False, "proposals": proposals}


def export_reviewed(review_path: Path, output: Path) -> dict[str, Any]:
    """Only explicit per-image visual/rights review can create new YOLO labels.

    Refuses overwritten output and unreviewed proposals. Source images/labels are
    untouched. Bounding boxes are supplied by the reviewer, not inferred here.
    """
    if output.exists():
        raise ValueError("Saída já existe; preservar versão anterior")
    review = json.loads(review_path.read_text())
    if review.get("geometry_policy") != "whole_main_front_destination_display":
        raise ValueError("Política geométrica não aprovada")
    records = review.get("records", [])
    if not records:
        raise ValueError("Revisão vazia")
    validated = []
    groups: dict[str, str] = {}
    image_hashes: set[str] = set()
    for record in records:
        if record.get("reviewed") is not True or not record.get("visual_evidence"):
            raise ValueError("Revisão visual explícita obrigatória")
        if record.get("rights_verified") is not True or not record.get("rights_evidence"):
            raise ValueError("Procedência não verificada")
        if not record.get("author") or not record.get("license"):
            raise ValueError("Autor e licença obrigatórios")
        if not str(record.get("source", "")).startswith("https://"):
            raise ValueError("Fonte HTTPS necessária")
        group, split = record.get("group_id"), record.get("split")
        if not group or not record.get("group_evidence") or split not in {"train", "val", "test"}:
            raise ValueError("Grupo, evidência e split explícitos obrigatórios")
        if group in groups and groups[group] != split:
            raise ValueError("Grupo cruza splits")
        groups[group] = split
        image_path = (review_path.parent / record["image"]).resolve(strict=True)
        digest = sha256(image_path)
        if digest != record.get("image_sha256") or digest in image_hashes:
            raise ValueError("Imagem alterada ou duplicada")
        image_hashes.add(digest)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Imagem ilegível")
        height, width = image.shape[:2]
        boxes = record.get("boxes", [])
        if not boxes and record.get("confirmed_negative") is not True:
            raise ValueError("Ausência de boxes não comprova negativo")
        lines = []
        for item in boxes:
            if item.get("approved") is not True or not item.get("boundary_evidence"):
                raise ValueError("Proposta automática não aprovada")
            x1, y1, x2, y2 = checked_box(item["xyxy"], width, height)
            lines.append(
                f"0 {(x1 + x2) / 2 / width:.8f} {(y1 + y2) / 2 / height:.8f} "
                f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
            )
        validated.append((record, image_path, lines))
    output.mkdir(parents=True)
    with (output / "manifest.csv").open("w") as stream:
        fields = ["image", "label", "group_id", "source", "license", "split"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, (record, image_path, lines) in enumerate(validated):
            label = output / f"{i:05d}.txt"
            label.write_text("\n".join(lines) + ("\n" if lines else ""))
            writer.writerow(
                {k: record[k] for k in fields if k not in {"image", "label"}}
                | {"image": str(image_path), "label": label.name}
            )
    receipt = {
        "review_sha256": sha256(review_path),
        "images": len(validated),
        "boxes": sum(len(lines) for _, _, lines in validated),
        "reviewer": review.get("reviewer"),
        "independent_human_review": review.get("independent_human_review", False),
        "geometry_policy": review["geometry_policy"],
    }
    (output / "review-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt
