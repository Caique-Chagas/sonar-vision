"""Source-identity/capture safeguards across separately resized dataset generations."""

from collections.abc import Mapping, Sequence
from typing import Any


def validate_lineage(
    records: Sequence[Mapping[str, Any]],
    prior: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Prevent resize bypass, historical test contamination and train/val migration.

    This complements pixel-hash/near-duplicate and review gates; it does not
    certify rights, annotations or capture-day metadata supplied by a reviewer.
    Historical train/val data may retain their role, never become independent test.
    """
    roles: dict[str, set[str]] = {}
    for record in prior:
        for key in ("id", "group_id"):
            value = record.get(key)
            if value:
                roles.setdefault(f"{key}:{value}", set()).add(str(record["split"]))
    assigned: dict[str, str] = {}
    ids: set[str] = set()
    for record in records:
        source_id, group, split = (record.get(k) for k in ("id", "group_id", "split"))
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(group, str)
            or not group
            or split not in {"train", "val", "test"}
        ):
            raise ValueError("Source identity, capture group and explicit split required")
        if source_id in ids:
            raise ValueError("Repeated source identity, even if image bytes differ")
        ids.add(source_id)
        if group in assigned and assigned[group] != split:
            raise ValueError("Capture group crosses splits")
        assigned[group] = split
        history = roles.get(f"id:{source_id}", set()) | roles.get(f"group_id:{group}", set())
        if history and split == "test":
            raise ValueError("Historical source/capture cannot become new independent test")
        if history - {split}:
            raise ValueError("Historical evaluation reservation or train/val role conflict")
    return {"images": len(ids), "capture_groups": len(assigned)}
