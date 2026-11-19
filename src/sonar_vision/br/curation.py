"""Bounded, resumable public-source acquisition with persistent HTTP stop conditions."""

import fcntl
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class CollectionStopped(RuntimeError):
    pass


class CollectionBudget(RuntimeError):
    pass


class BoundedSourceClient:
    def __init__(self, folder: Path, client: httpx.Client, *, max_requests: int = 32):
        if not 1 <= max_requests <= 64:
            raise ValueError("Invalid request budget")
        self.folder, self.client, self.maximum = folder, client, max_requests
        folder.mkdir(parents=True, exist_ok=True)
        self.ledger = folder / "http-ledger.json"
        self.stop = folder / "http-stop.json"
        self.entries = json.loads(self.ledger.read_text()) if self.ledger.exists() else []

    def resume_once_after_retry(self) -> None:
        """Explicit single resumption; preserve stop evidence and never bypass403.

        A valid seconds Retry-After and at least60seconds of cooldown are required.
        A repeated429 cannot be cleared again by this collector.
        """
        history = self.folder / "http-resume.json"
        if not self.stop.exists() or history.exists():
            raise CollectionStopped("No further resumption permitted")
        stopped = json.loads(self.stop.read_text())
        retry = str(stopped.get("retry_after", ""))
        if stopped.get("status") != 429 or not retry.isdecimal():
            raise CollectionStopped("Restriction requires a different external condition")
        elapsed = time.time() - self.stop.stat().st_mtime
        if elapsed < max(60, int(retry)):
            raise CollectionStopped("Retry-After/cooldown has not elapsed")
        stopped["cooldown_seconds"] = elapsed
        history.write_text(json.dumps(stopped, indent=2) + "\n")
        self.stop.rename(self.folder / "http-stop-before-single-resume.json")

    def fetch(self, url: str, *, max_bytes: int) -> bytes:
        with (self.folder / "http.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.entries = json.loads(self.ledger.read_text()) if self.ledger.exists() else []
            return self._fetch(url, max_bytes=max_bytes)

    def _fetch(self, url: str, *, max_bytes: int) -> bytes:
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or parts.hostname
            not in {"commons.wikimedia.org", "upload.wikimedia.org", "thumb.wikimedia.org"}
            or parts.username
            or parts.password
            or parts.port not in {None, 443}
        ):
            raise ValueError("Unapproved public-source URL")
        if not 1 <= max_bytes <= 12_000_000:
            raise ValueError("Invalid byte budget")
        if self.stop.exists():
            raise CollectionStopped("Source stopped; inspect persisted status and Retry-After")
        if len(self.entries) >= self.maximum:
            raise CollectionBudget("Persistent request budget exhausted")
        if self.entries and self.ledger.exists():
            remaining = 3 - (time.time() - self.ledger.stat().st_mtime)
            if remaining > 0:
                time.sleep(remaining)
        entry: dict[str, object] = {"url": url, "status": None, "retry_after": None}
        self.entries.append(entry)
        self.ledger.write_text(json.dumps(self.entries, indent=2) + "\n")
        with self.client.stream("GET", url) as response:
            entry.update(
                status=response.status_code, retry_after=response.headers.get("Retry-After")
            )
            self.ledger.write_text(json.dumps(self.entries, indent=2) + "\n")
            if response.status_code in {403, 429}:
                self.stop.write_text(json.dumps(entry, indent=2) + "\n")
                raise CollectionStopped("HTTP restriction; no retry or identity change")
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise CollectionBudget("Response exceeds byte budget")
            return bytes(body)


def canonical_author(author: str) -> str:
    """Normalize documented photographer aliases, never route text or missing dates."""
    import unicodedata

    value = " ".join(
        unicodedata.normalize("NFKD", author).encode("ascii", "ignore").decode().casefold().split()
    )
    if "mariordo" in value:
        return "mariordo"
    if "rafael delazari" in value:
        return "rafael delazari"
    return value


def assess_targeted_coverage(records: list[dict[str, object]]) -> dict[str, object]:
    """Coverage screen only; passing it never replaces split/rights/export audits."""
    from sonar_vision.training.panel_annotation import checked_box

    positive_groups: set[str] = set()
    small = integrated = codes = panels = incomplete = 0
    hashes: set[str] = set()
    for record in records:
        if record.get("reviewed") is not True or record.get("rights_verified") is not True:
            raise ValueError("Explicit visual and rights review required")
        digest = str(record.get("image_sha256", ""))
        if not digest or digest in hashes:
            raise ValueError("Missing/duplicate image identity")
        hashes.add(digest)
        boxes = record.get("boxes")
        if not isinstance(boxes, list):
            raise ValueError("Explicit box list required")
        if not boxes:
            if record.get("confirmed_negative") is not True:
                raise ValueError("Empty boxes do not establish a negative")
            continue
        width, height = int(str(record["width"])), int(str(record["height"]))
        if min(width, height) <= 0:
            raise ValueError("Invalid image dimensions")
        group = str(record.get("group_id", ""))
        if not group:
            raise ValueError("Capture group required")
        positive_groups.add(group)
        integrated += record.get("windshield_integrated") is True
        codes += bool(record.get("expected_code"))
        incomplete += record.get("exhaustive_annotation") is not True
        for box in boxes:
            if not isinstance(box, dict) or box.get("approved") is not True:
                raise ValueError("Unreviewed proposal cannot establish panel coverage")
            xyxy = checked_box(box["xyxy"], width, height)
            small += (xyxy[3] - xyxy[1]) * 640 / max(width, height) < 20
            panels += 1
    counts = {
        "positive_groups": len(positive_groups),
        "small_panels": small,
        "windshield_images": integrated,
        "complete_code_images": codes,
    }
    minimum = {
        "positive_groups": 12,
        "small_panels": 8,
        "windshield_images": 8,
        "complete_code_images": 6,
    }
    reasons = [
        f"{name}: {counts[name]} < research gate {value}"
        for name, value in minimum.items()
        if counts[name] < value
    ]
    if incomplete:
        reasons.append(f"{incomplete} frames require exhaustive annotation review")
    return {
        "coverage_passed": not reasons,
        "counts": counts,
        "panels": panels,
        "incomplete_frames": incomplete,
        "reasons": reasons,
        "scope": (
            "Engineering coverage screen, not statistical sufficiency or training authorization"
        ),
    }
