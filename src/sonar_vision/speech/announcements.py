from sonar_vision.core.contracts import MatchStatus, VisionEvent


def announcement(event: VisionEvent) -> str:
    prefix = "Simulação. " if event.simulated else ""
    if event.match_status == MatchStatus.MATCH:
        return (
            prefix
            + f"Ônibus da linha {event.route_detected} identificado. Linha solicitada confirmada."
        )
    if event.match_status == MatchStatus.NON_MATCH:
        return prefix + f"Ônibus da linha {event.route_detected} identificado. É outra linha."
    return prefix + "Não foi possível confirmar a identificação deste ônibus."


class AnnouncementGate:
    """Bounded deduplication within a session, per track and semantic result."""

    def __init__(self) -> None:
        self.seen: set[tuple[int, str | None, MatchStatus]] = set()
        self.last: dict[int, MatchStatus] = {}

    def accept(self, event: VisionEvent) -> bool:
        key = (event.track_id, event.route_detected, event.match_status)
        previous = self.last.get(event.track_id)
        self.last[event.track_id] = event.match_status
        # Report loss of consensus even if the initial uncertainty was announced.
        # Confirmed semantic results remain deduplicated across the session.
        lost_consensus = event.match_status == MatchStatus.UNCERTAIN and previous in {
            MatchStatus.MATCH,
            MatchStatus.NON_MATCH,
        }
        if key in self.seen and not lost_consensus:
            return False
        self.seen.add(key)
        return True
