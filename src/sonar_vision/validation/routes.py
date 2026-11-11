from dataclasses import dataclass

from sonar_vision.core.contracts import MatchStatus
from sonar_vision.validation.normalizer import clean_code


@dataclass(frozen=True)
class MatchResult:
    status: MatchStatus
    reason: str


class RouteMatcher:
    def __init__(self, allowed_routes: set[str] | None = None):
        self.allowed = None if allowed_routes is None else {clean_code(r) for r in allowed_routes}

    def match(self, detected: str | None, requested: str) -> MatchResult:
        if detected is None:
            return MatchResult(MatchStatus.UNCERTAIN, "Leitura insuficiente ou ambígua")
        if self.allowed is not None and detected not in self.allowed:
            return MatchResult(MatchStatus.UNCERTAIN, "Linha ausente do catálogo contextual")
        if detected == clean_code(requested):
            return MatchResult(MatchStatus.MATCH, "Código completo solicitado identificado")
        return MatchResult(MatchStatus.NON_MATCH, "Código completo de outra linha identificado")
