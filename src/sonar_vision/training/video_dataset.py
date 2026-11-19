"""Two-class video export with explicit rights, visual-review and group gates."""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

from sonar_vision.br.curation import canonical_author
from sonar_vision.training.dataset import dhash, sha256, validate_labels
from sonar_vision.training.panel_annotation import checked_box

CLASSES = {0: "bus", 1: "destination_sign"}


def review_records(review: dict[str, Any], permissions: dict[str, Any]) -> list[dict[str, Any]]:
    rows = review.get("images", [])
    if not rows:
        raise ValueError("Nenhuma imagem com revisão visual aprovada")
    validated = []
    verified_videos: set[str] = set()
    groups: dict[tuple[str, str], str] = {}
    digests: set[str] = set()
    hashes: list[tuple[int, str]] = []
    for row in rows:
        split = row.get("split")
        permit = permissions.get(row.get("video_sha256"), {})
        usage = "train_ML" if split == "train" else "evaluate_ML"
        private = (
            permit.get("usage_basis") == "user_directed_private_experiment"
            and permit.get("private_experiment_recorded") is True
            and permit.get("public_demo") is False
            and permit.get("redistribution") is False
        )
        if not private:
            if permit.get(usage) is not True or permit.get("rights_reviewed") is not True:
                raise PermissionError("UsoML não autorizado/revisado para a fonte")
            evidence = Path(permit.get("evidence_file", "__missing__"))
            if (
                not permit.get("author")
                or not permit.get("source_url", "").startswith("https://")
                or not evidence.is_file()
                or sha256(evidence) != permit.get("evidence_sha256")
            ):
                raise PermissionError("Documento de direitos/autoria/fonte ausente ou alterado")
            if (
                permit.get("provider") == "pexels"
                and permit.get("specific_ML_grant_reviewed") is not True
            ):
                raise PermissionError("Pexels exige comprovação compatível específica para ML")
        else:
            record = Path(permit.get("user_instruction_record", "__missing__"))
            if not record.is_file() or sha256(record) != permit.get("user_instruction_sha256"):
                raise PermissionError(
                    "Orientação privada precisa de registro íntegro; não é licença"
                )
        video_sha = row["video_sha256"]
        if video_sha not in verified_videos:
            video = Path(permit.get("video_file", "__missing__"))
            if not video.is_file() or sha256(video) != video_sha:
                raise ValueError("Vídeo de origem ausente ou modificado")
            verified_videos.add(video_sha)
        if split not in {"train", "val", "test"}:
            raise ValueError("Split explícito obrigatório")
        if row.get("reviewed") is not True or row.get("all_objects_reviewed") is not True:
            raise ValueError("Caixas e cobertura de todos os objetos precisam de revisão")
        if row.get("ignore_regions"):
            raise ValueError("YOLO padrão não exporta ignore-regions: resolva/exclua o frame")
        for key in ["capture_group", "vehicle_group"]:
            value = row.get(key)
            if not value or str(value).startswith("PENDING"):
                raise ValueError("Grupos de captura/veículo precisam de adjudicação")
            identity = (key, value)
            if identity in groups and groups[identity] != split:
                raise ValueError("Grupo ou veículo cruza splits")
            groups[identity] = split
        if permit.get("author"):
            author_key = ("author", canonical_author(permit["author"]))
            if author_key in groups and groups[author_key] != split:
                raise ValueError(
                    "Fotógrafo cruza splits; adjudique independência antes de exportar"
                )
            groups[author_key] = split
        source = Path(row["image"])
        digest = sha256(source)
        if digest != row.get("image_sha256") or digest in digests:
            raise ValueError("Imagem alterada ou duplicada")
        digests.add(digest)
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError("Imagem ilegível")
        fingerprint = dhash(image)
        if any(s != split and (fingerprint ^ h).bit_count() <= 3 for h, s in hashes):
            raise ValueError("Quase duplicata entre splits")
        hashes.append((fingerprint, split))
        height, width = image.shape[:2]
        labels = []
        for box in row.get("boxes", []):
            cls = box["class_id"]
            if type(cls) is not int or cls not in CLASSES:
                raise ValueError("Classe precisa ser0bus ou1destination_sign")
            x1, y1, x2, y2 = checked_box(box["xyxy"], width, height)
            labels.append(
                f"{cls} {(x1 + x2) / 2 / width:.8f} {(y1 + y2) / 2 / height:.8f} "
                f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
            )
        validated.append({**row, "labels": labels})
    counts = Counter(r["split"] for r in validated)
    if not counts["train"] or not counts["val"]:
        raise ValueError("Treino e validação precisam ter grupos distintos revisados")
    for split in ["train", "val"]:
        present = {
            int(label.split()[0]) for r in validated if r["split"] == split for label in r["labels"]
        }
        if present != set(CLASSES):
            raise ValueError("As duas classes precisam existir em treino e validação")
    return validated


def export_video_dataset(review_path: Path, permission_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Saída existente: preservar versão anterior")
    review, permits = json.loads(review_path.read_text()), json.loads(permission_path.read_text())
    rows = review_records(review, permits)
    output.mkdir(parents=True)
    artifacts = []
    for index, row in enumerate(rows):
        split = row["split"]
        images, labels = output / "images" / split, output / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        image = images / f"{index:05d}.jpg"
        shutil.copyfile(row["image"], image)
        if sha256(image) != row["image_sha256"]:
            raise ValueError("Imagem mudou durante exportação")
        label = labels / f"{index:05d}.txt"
        label.write_text("\n".join(row["labels"]) + "\n")
        validate_labels(label, 2)
        artifacts += [{"path": str(p.resolve()), "sha256": sha256(p)} for p in [image, label]]
    config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASSES,
    }
    if any(r["split"] == "test" for r in rows):
        config["test"] = "images/test"
    (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    audit = {
        "classes": CLASSES,
        "counts": dict(Counter(r["split"] for r in rows)),
        "review_sha256": sha256(review_path),
        "permission_sha256": sha256(permission_path),
        "review_file": str(review_path.resolve()),
        "permission_file": str(permission_path.resolve()),
        "config_sha256": sha256(output / "data.yaml"),
        "files": artifacts,
        "independent_test_claim": False,
        "status": "REVIEWED_EXPORT",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def verify_video_export(data: Path) -> dict[str, Any]:
    audit: dict[str, Any] = json.loads((data.parent / "audit.json").read_text())
    review_path, permits = Path(audit["review_file"]), Path(audit["permission_file"])
    if (
        sha256(data) != audit["config_sha256"]
        or sha256(review_path) != audit["review_sha256"]
        or sha256(permits) != audit["permission_sha256"]
    ):
        raise ValueError("Configuração, revisão ou direitos alterados após exportação")
    review_records(json.loads(review_path.read_text()), json.loads(permits.read_text()))
    for row in audit["files"]:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("Dados exportados foram alterados")
    return audit
