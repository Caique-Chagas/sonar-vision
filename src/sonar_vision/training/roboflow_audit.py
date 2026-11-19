"""Audit semantic YOLO exports without assuming word boxes are whole signs."""

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

from sonar_vision.training.dataset import dhash, origin_family, sha256


def class_names(root: Path) -> dict[int, str]:
    config = yaml.safe_load((root / "data.yaml").read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuração YAML inválida")
    names = config.get("names")
    if isinstance(names, list):
        result = dict(enumerate(str(n) for n in names))
    elif isinstance(names, dict):
        result = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValueError("Dataset sem names válidos")
    if (
        not result
        or set(result) != set(range(len(result)))
        or any(not n.strip() for n in result.values())
    ):
        raise ValueError("Classes inconsistentes")
    if config.get("nc", len(result)) != len(result):
        raise ValueError("nc diverge de names")
    return result


def read_annotations(path: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    """Preserve polygon vertices; envelopes are descriptive until geometry review."""
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if class_id not in names or not all(math.isfinite(v) for v in coords):
            raise ValueError("Classe ou coordenada inválida")
        points = None
        if len(parts) == 5:
            x, y, width, height = coords
            if width <= 0 or height <= 0:
                raise ValueError("Bounding box sem área")
            xmin, xmax = x - width / 2, x + width / 2
            ymin, ymax = y - height / 2, y + height / 2
            geometry = "box"
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            points = list(zip(coords[::2], coords[1::2], strict=True))
            xs, ys = coords[::2], coords[1::2]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            area = (
                abs(
                    sum(
                        a[0] * b[1] - b[0] * a[1]
                        for a, b in zip(points, points[1:] + points[:1], strict=True)
                    )
                )
                / 2
            )
            if area <= 1e-10:
                raise ValueError("Polígono degenerado")
            x, y, width, height = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
            geometry = "polygon"
        else:
            raise ValueError("Anotação YOLO inválida")
        if xmin < -1e-5 or ymin < -1e-5 or xmax > 1.00001 or ymax > 1.00001:
            raise ValueError("Anotação fora da imagem")
        result.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "xywh": [x, y, width, height],
                "source_geometry": geometry,
                "source_polygon": points,
            }
        )
    return result


def audit_export(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    names = class_names(root)
    images = sorted(
        p
        for split in ["train", "valid", "val", "test"]
        for p in (root / split / "images").glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not 1 <= len(images) <= 5000:
        raise ValueError("Dataset vazio ou acima do limite de 5000 imagens")
    records = []
    counts: Counter[str] = Counter()
    for image in images:
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise ValueError(f"Label ausente; não presumir negativo: {image.name}")
        if not image.resolve().is_relative_to(root) or not label.resolve().is_relative_to(root):
            raise ValueError("Arquivo fora do export")
        boxes = read_annotations(label, names)
        pixels = cv2.imread(str(image))
        if pixels is None:
            raise ValueError(f"Imagem corrompida: {image.name}")
        height, width = pixels.shape[:2]
        for box in boxes:
            counts[box["class_name"]] += 1
        records.append(
            {
                "image": image.relative_to(root).as_posix(),
                "label": label.relative_to(root).as_posix(),
                "split": image.relative_to(root).parts[0],
                "image_sha256": sha256(image),
                "label_sha256": sha256(label),
                "origin_family": origin_family(image.stem),
                "dhash": dhash(pixels),
                "width": width,
                "height": height,
                "boxes": [
                    box | {"pixels_wh": [box["xywh"][2] * width, box["xywh"][3] * height]}
                    for box in boxes
                ],
            }
        )
    pairs = []
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            exact = a["image_sha256"] == b["image_sha256"]
            near = (a["dhash"] ^ b["dhash"]).bit_count() <= 3
            family = a["origin_family"] == b["origin_family"]
            if exact or near or family:
                pairs.append(
                    {
                        "a": a["image"],
                        "b": b["image"],
                        "exact": exact,
                        "near": near,
                        "family": family,
                        "cross_split": a["split"] != b["split"],
                    }
                )
    return {
        "images": len(records),
        "classes": names,
        "class_box_counts": dict(counts),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "negative_images": sum(not r["boxes"] for r in records),
        "annotation_geometry_counts": dict(
            Counter(b["source_geometry"] for r in records for b in r["boxes"])
        ),
        "suspected_correlated_pairs": pairs,
        "cross_split_pairs": sum(p["cross_split"] for p in pairs),
        "geometry_reviewed": False,
        "training_eligible": False,
        "reason": "Manual geometry and sequence review required; class text proves neither",
        "records": records,
    }


def convert_reviewed(root: Path, review_path: Path, output: Path) -> Path:
    """Only individually reviewed whole-sign boxes can become destination_sign.

    Original labels/classes are retained in the conversion receipt. Output feeds
    existing prepare_dataset(), which rechecks hashes/groups/near-duplicate splits.
    """
    if output.exists():
        raise ValueError("Saída existe; preserve o experimento")
    root = root.resolve(strict=True)
    audit = audit_export(root)
    review = json.loads(review_path.read_text())
    if (
        review.get("source_url")
        != "https://universe.roboflow.com/cakoworspace/deteccao-letreiro-onibus-gqjt2"
        or review.get("license") != "CC BY 4.0"
    ):
        raise ValueError("Fonte/licença não auditadas")
    if (
        review.get("rights_reviewed") is not True
        or not str(review.get("rights_evidence", "")).strip()
    ):
        raise ValueError("Procedência/direitos das fotografias precisam de evidência revisada")
    indexed = {r["image"]: r for r in audit["records"]}
    accepted = []
    seen = set()
    for item in review.get("images", []):
        record = indexed.get(item.get("image"))
        if record is None or item["image"] in seen:
            raise ValueError("Imagem desconhecida ou repetida na revisão")
        seen.add(item["image"])
        if (
            item.get("reviewed") is not True
            or not isinstance(item.get("group_id"), str)
            or not item["group_id"].strip()
            or not isinstance(item.get("group_evidence"), str)
            or not item["group_evidence"].strip()
        ):
            raise ValueError("Revisão geométrica e agrupamento explícitos obrigatórios")
        if any(item.get(k) != record[k] for k in ["image_sha256", "label_sha256"]):
            raise ValueError("Hashes divergem da revisão; reaudite os dados")
        selected = item.get("boxes", [])
        if not selected and item.get("confirmed_negative") is not True:
            raise ValueError("Ausência de caixas não comprova negativo")
        if item.get("confirmed_negative") is True and record["boxes"]:
            raise ValueError("Negativo contradiz anotações existentes")
        indices = set()
        for box in selected:
            i = box.get("index")
            if type(i) is not int or i in indices or not 0 <= i < len(record["boxes"]):
                raise ValueError("Índice de caixa inválido")
            indices.add(i)
            if box.get("geometry") != "whole_destination_sign":
                raise ValueError("Palavra/fragmento não pode virar letreiro inteiro")
        accepted.append((item, record, sorted(indices)))
    if not accepted:
        raise ValueError("Nenhuma imagem revisada")
    # Exact duplicates/families/near duplicates cannot receive independent group IDs.
    groups = {a[0]["image"]: a[0]["group_id"] for a in accepted}
    for pair in audit["suspected_correlated_pairs"]:
        if pair["a"] in groups and pair["b"] in groups and groups[pair["a"]] != groups[pair["b"]]:
            raise ValueError("Duplicatas/sequências correlacionadas precisam compartilhar grupo")
    output.mkdir(parents=True)
    rows, provenance = [], []
    for index, (item, record, selected_indices) in enumerate(accepted):
        path = output / f"{index:05d}.txt"
        path.write_text(
            "".join(
                "0 " + " ".join(str(v) for v in record["boxes"][i]["xywh"]) + "\n"
                for i in selected_indices
            )
        )
        rows.append(
            {
                "image": str(root / record["image"]),
                "label": path.name,
                "group_id": item["group_id"],
                "source": review["source_url"],
                "license": review["license"],
            }
        )
        provenance.append(
            {
                "image": record["image"],
                "image_sha256": record["image_sha256"],
                "original_label": record["label"],
                "original_label_sha256": record["label_sha256"],
                "selected_original_boxes": [record["boxes"][i] for i in selected_indices],
            }
        )
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "conversion-audit.json").write_text(
        json.dumps(
            {
                "source": review["source_url"],
                "license": review["license"],
                "review_sha256": sha256(review_path),
                "records": provenance,
                "training_eligible": False,
                "next_gate": "prepare_dataset + sequence/split review before any training",
            },
            indent=2,
        )
    )
    return manifest
