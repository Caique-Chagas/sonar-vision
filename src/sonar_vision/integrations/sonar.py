"""Replaceable operational context. Fixtures do not assert official route existence."""

import json
from importlib.resources import files
from typing import Protocol


class RouteProvider(Protocol):
    def allowed_routes(self, stop_id: str) -> set[str] | None: ...


class FixtureRouteProvider:
    def allowed_routes(self, stop_id: str) -> set[str]:
        data = json.loads(files("sonar_vision").joinpath("fixtures/routes.json").read_text())
        return set(data["routes"])


class NoOperationalContext:
    def allowed_routes(self, stop_id: str) -> None:
        return None
