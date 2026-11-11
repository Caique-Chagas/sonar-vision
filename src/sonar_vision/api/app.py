from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from sonar_vision import __version__
from sonar_vision.api.schemas import SessionRequest
from sonar_vision.api.sessions import Session, SessionManager
from sonar_vision.core.config import Settings
from sonar_vision.core.contracts import VisionEvent, to_legacy
from sonar_vision.core.logging import configure_logging
from sonar_vision.processing.sources import SourceError, validate_url


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or Settings()
    manager = SessionManager(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        manager.close()

    app = FastAPI(
        title="SONAR VISION", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.state.manager = manager
    web = str(files("sonar_vision").joinpath("web"))
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.middleware("http")
    async def local_security(request: Request, call_next: Any) -> Response:
        # No cross-origin commands against a localhost service, including DNS rebinding hosts.
        host = request.headers.get("host", "").split(":")[0]
        origin = request.headers.get("origin")
        if host not in {"localhost", "127.0.0.1", "testserver"}:
            return Response("Host não permitido", status_code=400)
        if origin and origin not in {
            f"http://{request.headers.get('host')}",
            f"https://{request.headers.get('host')}",
        }:
            return Response("Origem não permitida", status_code=403)
        if request.method == "POST":
            try:
                content_length = int(request.headers.get("content-length", "-1"))
            except ValueError:
                return Response("Content-Length inválido", status_code=400)
            if content_length < 0:
                return Response("Content-Length obrigatório", status_code=411)
            if content_length > 16384:
                return Response("Payload excede 16 KiB", status_code=413)
        response: Response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    def get_session(session_id: str) -> Session:
        try:
            return manager.get(session_id)
        except KeyError as exc:
            raise HTTPException(404, "Sessão não encontrada ou expirada") from exc

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(f"{web}/index.html")

    @app.get("/docs", include_in_schema=False)
    def api_docs() -> FileResponse:
        return FileResponse(f"{web}/api.html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "academic_status": "PARTIALLY_COMPLETED",
            "custom_model_available": settings.model_path.is_file(),
            "simulation_available": True,
            "model_note": "Presença de arquivo não comprova treinamento nem qualidade",
        }

    @app.post("/api/sessions", status_code=202)
    def start_session(request: SessionRequest) -> dict[str, Any]:
        if request.mode == "real":
            try:
                validate_url(request.source_url or "", settings.allowed_hosts)
            except SourceError as exc:
                raise HTTPException(422, str(exc)) from exc
            if not settings.model_path.is_file():
                raise HTTPException(503, "Pesos customizados ausentes; consulte docs/TRAINING.md")
        try:
            return manager.create(request).snapshot()
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc

    @app.get("/api/sessions/{session_id}")
    def session_state(session_id: str) -> dict[str, Any]:
        return get_session(session_id).snapshot()

    @app.delete("/api/sessions/{session_id}", status_code=202)
    def stop_session(session_id: str) -> dict[str, str]:
        get_session(session_id).stop.set()
        return {"status": "stop_requested"}

    @app.get("/api/sessions/{session_id}/frame")
    def latest_frame(session_id: str) -> Response:
        session = get_session(session_id)
        with session.lock:
            jpeg = session.jpeg
        if jpeg is None:
            return Response(status_code=204)
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/api/sessions/{session_id}/events", response_model=list[VisionEvent])
    def session_events(session_id: str) -> list[VisionEvent]:
        return [
            VisionEvent(**{k: v for k, v in e.items() if k != "announcement"})
            for e in get_session(session_id).snapshot()["events"]
        ]

    @app.get("/api/sessions/{session_id}/legacy")
    def legacy_events(session_id: str) -> list[dict[str, Any]]:
        return [to_legacy(e).model_dump(mode="json") for e in session_events(session_id)]

    return app


app = create_app()
