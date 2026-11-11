"""Auditable YOLO dataset creation. Reject duplicate/near-duplicate split leakage."""

import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def origin_family(filename: str) -> str:
    return re.split(r"\.rf\.", filename)[0]


def dhash(image: np.ndarray[Any, Any]) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = cv2.resize(gray, (9, 8))
    return int("".join("1" if bit else "0" for bit in (pixels[:, 1:] > pixels[:, :-1]).flat), 2)


def validate_labels(path: Path, class_count: int) -> list[tuple[int, float, float, float, float]]:
    output = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Rótulo YOLO inválido: {path.name}")
        class_id = int(parts[0])
        x, y, w, h = (float(value) for value in parts[1:])
        if not 0 <= class_id < class_count or not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"Classe ou coordenada inválida: {path.name}")
        if (
            not 0 < w <= 1
            or not 0 < h <= 1
            or x - w / 2 < -1e-5
            or y - h / 2 < -1e-5
            or x + w / 2 > 1.00001
            or y + h / 2 > 1.00001
        ):
            raise ValueError(f"Bounding box fora da imagem: {path.name}")
        output.append((class_id, x, y, w, h))
    return output


def prepare_dataset(
    manifest: Path,
    output: Path,
    *,
    seed: int = 42,
    split_assurance: str = "origin_group_only_sequence_unknown",
) -> dict[str, Any]:
    """Manifest: image,label,group_id,source,license (paths relative to CSV)."""
    if output.exists():
        raise ValueError("Diretório de saída já existe; preserve o checkpoint e use outro caminho")
    rows = list(csv.DictReader(manifest.open()))
    if not rows or not {"image", "label", "group_id", "source", "license"} <= rows[0].keys():
        raise ValueError("Manifesto precisa de image,label,group_id,source,license")
    groups = sorted({row["group_id"] for row in rows})
    if len(groups) < 3 or "" in groups:
        raise ValueError("São necessários pelo menos três grupos de origem não vazios")
    random.Random(seed).shuffle(groups)
    nval = max(1, round(len(groups) * 0.15))
    ntest = max(1, round(len(groups) * 0.15))
    assignments = {
        group: ("test" if i < ntest else "val" if i < ntest + nval else "train")
        for i, group in enumerate(groups)
    }
    if "train" not in assignments.values():
        raise ValueError("Divisão sem grupo de treino")
    seen_hashes: dict[str, str] = {}
    seen_origins: dict[str, str] = {}
    prepared: list[dict[str, Any]] = []
    for row in rows:
        image_path = (manifest.parent / row["image"]).resolve(strict=True)
        label_path = (manifest.parent / row["label"]).resolve(strict=True)
        if not row["license"].strip() or not row["source"].startswith("https://"):
            raise ValueError("Fonte HTTPS e licença explícita são obrigatórias")
        if len(rows) > 5000:
            raise ValueError("Limite do preparador MVP: 5000 imagens")
        labels = validate_labels(label_path, 1)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Imagem ilegível: {image_path.name}")
        split = assignments[row["group_id"]]
        digest = sha256(image_path)
        origin = origin_family(image_path.name)
        for key, index in ((digest, seen_hashes), (origin, seen_origins)):
            if key in index and index[key] != split:
                raise ValueError(
                    "Vazamento entre splits: imagem duplicada ou família de aumentação"
                )
            index[key] = split
        prepared.append(
            row
            | {
                "image_path": image_path,
                "label_path": label_path,
                "sha256": digest,
                "label_sha256": sha256(label_path),
                "dhash": dhash(image),
                "split": split,
                "boxes": len(labels),
            }
        )
    for i, a in enumerate(prepared):
        for b in prepared[i + 1 :]:
            if a["split"] != b["split"] and (a["dhash"] ^ b["dhash"]).bit_count() <= 3:
                raise ValueError(
                    f"Possível quase duplicata entre splits: {a['image']} / {b['image']}; "
                    "agrupe as sequências antes de treinar"
                )
    output.mkdir(parents=True)
    exported = []
    for i, row in enumerate(prepared):
        split = row["split"]
        for folder in ("images", "labels"):
            (output / split / folder).mkdir(parents=True, exist_ok=True)
        filename = f"{i:05d}{row['image_path'].suffix.lower()}"
        shutil.copy2(row["image_path"], output / split / "images" / filename)
        shutil.copy2(row["label_path"], output / split / "labels" / f"{i:05d}.txt")
        exported.append(
            {k: v for k, v in row.items() if k not in {"image_path", "label_path"}}
            | {"prepared_image": f"{split}/images/{filename}"}
        )
    data = {
        "path": str(output.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "destination_sign"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data))
    report = {
        "schema_version": "1.0",
        "seed": seed,
        "manifest_sha256": sha256(manifest),
        "split_assurance": split_assurance,
        "counts": dict(Counter(row["split"] for row in prepared)),
        "groups": len(groups),
        "images": len(rows),
        "classes": ["destination_sign"],
        "near_duplicate_hamming_threshold": 3,
        "records": exported,
    }
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def verify_dataset(data_yaml: Path) -> dict[str, Any]:
    """Recheck hashes rather than trusting a stale audit before training/evaluation."""
    root = data_yaml.resolve().parent
    audit: dict[str, Any] = json.loads((root / "audit.json").read_text())
    config = yaml.safe_load(data_yaml.read_text())
    if set(config) != {"path", "train", "val", "test", "names"}:
        raise ValueError("Configuração de dados contém campos não auditados")
    if Path(config["path"]).resolve() != root or any(
        config[split] != f"{split}/images" for split in ("train", "val", "test")
    ):
        raise ValueError("Caminhos divergem dos splits auditados")
    if config.get("names") != {0: "destination_sign"}:
        raise ValueError("Dataset não contém a classe única destination_sign")
    for record in audit["records"]:
        image = root / record["prepared_image"]
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        if sha256(image) != record["sha256"] or sha256(label) != record["label_sha256"]:
            raise ValueError("Dataset alterado após auditoria; gere nova versão")
        validate_labels(label, 1)
    return audit
