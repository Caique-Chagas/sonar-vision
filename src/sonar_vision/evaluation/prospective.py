"""Audit inputs and pair outputs without inventing truth or temporal evidence."""

import json
from pathlib import Path
from typing import Any

from sonar_vision.core.contracts import Box
from sonar_vision.tracking.iou import iou
from sonar_vision.training.dataset import sha256


def verify_prospective_inputs(protocol: Path, reference: Path, root: Path) -> dict[str, Any]:
    spec = json.loads(protocol.read_text())
    refs: dict[str, Any] = json.loads(reference.read_text())
    if refs.get("protocol_sha256") != sha256(protocol):
        raise ValueError("Referência não pertence ao protocolo congelado")
    if refs.get("reviewed_before_model_outputs") is not True:
        raise ValueError("Gabarito anterior à inferência necessário")
    for relative, digest in spec["source_hashes"].items():
        if sha256(root / relative) != digest:
            raise ValueError("Implementação congelada mudou")
    for profile in spec["profiles"].values():
        if sha256(root / profile["weights"]) != profile["sha256"]:
            raise ValueError("Pesos congelados mudaram")
    ids, hashes = set(), set()
    for record in refs["records"]:
        if record.get("rights_reviewed") is not True:
            raise ValueError("Direitos não revisados")
        if not record.get("split", "").startswith("prospective_test"):
            raise ValueError("Imagem de desenvolvimento/treino no teste prospectivo")
        if record["id"] in ids or record["image_sha256"] in hashes:
            raise ValueError("Imagem repetida na avaliação")
        ids.add(record["id"])
        hashes.add(record["image_sha256"])
        if sha256((reference.parent / record["image"]).resolve()) != record["image_sha256"]:
            raise ValueError("Imagem mudou após anotação")
        if record.get("expected_code") and not record.get("ground_truth_method"):
            raise ValueError("Código sem evidência independente")
    return refs


def match_panels(
    predictions: list[dict[str, Any]],
    references: list[Box],
    ignore_regions: list[Box],
    threshold: float = 0.5,
) -> dict[str, int]:
    matched: set[int] = set()
    false_positives, ignored = 0, 0
    for prediction in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
        box = Box(**prediction["box"])
        scores = [
            (iou(box, ref), index) for index, ref in enumerate(references) if index not in matched
        ]
        best = max(scores, default=(0.0, -1))
        if best[0] >= threshold:
            matched.add(best[1])
        elif any(
            region.x1 <= (box.x1 + box.x2) / 2 <= region.x2
            and region.y1 <= (box.y1 + box.y2) / 2 <= region.y2
            for region in ignore_regions
        ):
            ignored += 1
        else:
            false_positives += 1
    return {
        "tp": len(matched),
        "fn": len(references) - len(matched),
        "fp": false_positives,
        "ignored_unassessed": ignored,
    }


def summarize_paired(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile = {name: [r for r in records if r["profile"] == name] for name in ["A", "D"]}
    identifiers = [{r["id"] for r in rows} for rows in by_profile.values()]
    if not identifiers[0] or identifiers[0] != identifiers[1]:
        raise ValueError("Comparação precisa das mesmas imagens em A e D")
    output = {}
    for name, rows in by_profile.items():
        if len(rows) != len({r["id"] for r in rows}):
            raise ValueError("Exemplo repetido não aumenta denominador")
        if any(r.get("confirmed_events", 0) for r in rows):
            raise ValueError("Fotografias isoladas não comprovam confirmação temporal")
        coded = [r for r in rows if r.get("expected_code")]
        output[name] = {
            "images": len(rows),
            "coded_images": len(coded),
            "exact_codes": sum(r["code_exact"] for r in coded),
            "eligible_exact_codes": sum(r["eligible_code_exact"] for r in coded),
            "complete_code_accuracy": sum(r["code_exact"] for r in coded) / len(coded)
            if coded
            else None,
            "confirmed_events": 0,
            "panel_tp": sum(r["panel_metrics"]["tp"] for r in rows),
            "panel_fn": sum(r["panel_metrics"]["fn"] for r in rows),
            "panel_fp": sum(r["panel_metrics"]["fp"] for r in rows),
            "unassessed_predictions": sum(r["panel_metrics"]["ignored_unassessed"] for r in rows),
        }
    return output
