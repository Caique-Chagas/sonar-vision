"""Catalog-aware normalization. OCR substitutions remain suggestions, never certainty."""

import re
import unicodedata
from dataclasses import dataclass

FULL_CODE = re.compile(
    r"(?<![A-Z0-9])([0-9]{3}[A-Z0-9]|[0-9]{3,5}|[A-Z][0-9]{3})-([0-9]{2})(?![A-Z0-9])"
)
SHORT_CODE = re.compile(r"(?<![A-Z0-9])([0-9]{3}[A-Z0-9]|[0-9]{3,5})(?![A-Z0-9-])")


def clean_code(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).upper()
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", "", value)


@dataclass(frozen=True)
class NormalizedLine:
    raw: str
    candidates: tuple[str, ...]
    method: str
    ambiguous: bool

    @property
    def route(self) -> str | None:
        return self.candidates[0] if len(self.candidates) == 1 and not self.ambiguous else None


class LineNormalizer:
    def __init__(self, routes: set[str] | None = None):
        self.routes = {clean_code(route) for route in routes or set()}

    def normalize(self, text: str) -> NormalizedLine:
        # Keep word boundaries for extraction, allow spaces inside an otherwise valid code.
        normalized = unicodedata.normalize("NFKC", text).upper()
        normalized = normalized.replace("–", "-").replace("—", "-").replace("−", "-")
        normalized = re.sub(r"\s*-\s*", "-", normalized)
        compact = clean_code(text)
        if compact in self.routes:
            return NormalizedLine(text, (compact,), "catalog_exact", False)
        exact = {f"{m[0]}-{m[1]}" for m in FULL_CODE.findall(normalized)}
        if not exact and FULL_CODE.fullmatch(compact):
            exact.add(compact)
        # Also recognize catalog formats not covered by the Brazilian example pattern.
        tokens = set(re.findall(r"[A-Z0-9]+(?:-[A-Z0-9]+)*", normalized))
        exact |= tokens & self.routes
        if exact:
            return NormalizedLine(text, tuple(sorted(exact)), "exact", len(exact) != 1)
        shorts = set(SHORT_CODE.findall(normalized))
        if compact.isalnum():
            shorts.add(compact)
        candidates = {r for r in self.routes if r.split("-")[0] in shorts}
        if candidates:
            # Suffix and direction were not seen: do not infer them from a catalog.
            return NormalizedLine(text, tuple(sorted(candidates)), "abbreviated", True)
        translated = compact.translate(str.maketrans("OIS", "015"))
        suggestions = {r for r in self.routes if clean_code(r) == translated}
        return NormalizedLine(text, tuple(sorted(suggestions)), "ocr_suggestion", True)
