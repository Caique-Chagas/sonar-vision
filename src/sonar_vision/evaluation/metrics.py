"""Explicit denominators; missing classes produce null metrics rather than fake zeros."""

from collections.abc import Sequence
from typing import Any


def route_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Each record is one bus encounter: expected/predicted full code or None for negative."""
    if not records:
        raise ValueError("Avaliação exige encontros anotados independentes")
    ids = [r["encounter_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Cada encontro deve aparecer uma única vez")
    positives = [r for r in records if r["expected"] is not None]
    negatives = [r for r in records if r["expected"] is None]
    wrong_confirmations = [
        r for r in records if r["predicted"] is not None and r["predicted"] != r["expected"]
    ]
    latency = [float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]
    return {
        "encounters": len(records),
        "positive_encounters": len(positives),
        "negative_encounters": len(negatives),
        "route_accuracy": sum(r["predicted"] == r["expected"] for r in positives) / len(positives)
        if positives
        else None,
        "wrong_confirmation_rate_all_encounters": len(wrong_confirmations) / len(records),
        "false_positive_rate_negative_encounters": sum(
            r["predicted"] is not None for r in negatives
        )
        / len(negatives)
        if negatives
        else None,
        "correct_rejection_rate_negative_encounters": sum(r["predicted"] is None for r in negatives)
        / len(negatives)
        if negatives
        else None,
        "mean_encounter_latency_ms": sum(latency) / len(latency) if latency else None,
    }
