"""Raw MP4 streaming upload avoids multipart buffering and filename-based paths."""

import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from sonar_vision.api.video_jobs import VideoJob, VideoJobManager
from sonar_vision.processing.local_video import MAX_UPLOAD, LocalVideoError, validate_upload_name


def video_router(manager: VideoJobManager) -> APIRouter:
    router = APIRouter(prefix="/api/video-demo")

    def get(id: str) -> VideoJob:
        try:
            return manager.get(id)
        except KeyError as exc:
            raise HTTPException(404, "Análise não encontrada ou expirada") from exc

    @router.post("/jobs", status_code=202)
    async def upload(
        request: Request,
        filename: str = "",
        profile: Literal["baseline", "diagnostic1280"] = "baseline",
    ) -> dict[str, Any]:
        try:
            validate_upload_name(
                filename,
                request.headers.get("content-type", ""),
                request.headers.get("x-sonar-media-permission", ""),
            )
        except LocalVideoError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not manager.weights.is_file():
            raise HTTPException(503, "Pesos BR V1 ausentes; consulte o guia de demonstração")
        try:
            job = manager.reserve()
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc
        try:
            transferred = 0
            async with asyncio.timeout(30):
                with (job.folder / "source.mp4").open("xb") as stream:
                    async for chunk in request.stream():
                        transferred += len(chunk)
                        if transferred > MAX_UPLOAD:
                            raise HTTPException(413, "Máximo64MiB")
                        stream.write(chunk)
            if transferred < 16:
                raise HTTPException(422, "Arquivo vazio ou truncado")
            with (job.folder / "source.mp4").open("rb") as stream:
                if stream.read(12)[4:8] != b"ftyp":
                    raise HTTPException(422, "Conteúdo não corresponde a MP4")
            job.data["filename"] = filename
            job.data["profile"] = profile
            (job.folder / "job-config.json").write_text(json.dumps({"profile": profile}))
            manager.start(job)
            return job.snapshot()
        except TimeoutError as exc:
            manager.discard_upload(job)
            raise HTTPException(408, "Upload excedeu30segundos") from exc
        except BaseException:
            manager.discard_upload(job)
            raise

    @router.get("/jobs/{id}")
    def status(id: str) -> dict[str, Any]:
        return get(id).snapshot()

    @router.delete("/jobs/{id}", status_code=202)
    def cancel(id: str) -> dict[str, str]:
        job = get(id)
        if job.state != "COMPLETED":
            job.stop.set()
        return {"status": "cancel_requested"}

    @router.get("/examples/{id}")
    def example(id: str) -> FileResponse:
        if id not in {"curitiba", "saopaulo", "rotas", "rotas-approach"}:
            raise HTTPException(404, "Exemplo desconhecido")
        file = manager.root.parent / "practical-examples" / (id + "-real.mp4")
        if not file.is_file():
            raise HTTPException(404, "Exemplo local ainda não preparado")
        return FileResponse(file, media_type="video/mp4", filename=file.name)

    @router.get("/static-evidence/{kind}")
    def static(kind: str) -> FileResponse:
        names = {
            "image": ("static-br-v1.jpg", "image/jpeg"),
            "report": ("static-br-v1.json", "application/json"),
        }
        if kind not in names:
            raise HTTPException(404, "Evidência não encontrada")
        name, mime = names[kind]
        file = manager.root.parent / "practical-examples" / name
        if not file.is_file():
            raise HTTPException(404, "Evidência estática ainda não preparada")
        return FileResponse(file, media_type=mime)

    @router.get("/jobs/{id}/frame")
    def frame(id: str) -> Response:
        job = get(id)
        with job.lock:
            jpeg = job.jpeg
        return Response(jpeg, media_type="image/jpeg") if jpeg else Response(status_code=204)

    @router.get("/jobs/{id}/files/{kind}")
    def download(id: str, kind: str) -> FileResponse:
        job = get(id)
        if job.state != "COMPLETED":
            raise HTTPException(409, "Resultado ainda não disponível")
        names = {
            "video": ("annotated.mp4", "video/mp4"),
            "report": ("report.json", "application/json"),
            "frames": ("frames.jsonl", "application/x-ndjson"),
            "detection": ("first-detection.jpg", "image/jpeg"),
        }
        if kind not in names:
            raise HTTPException(404, "Arquivo não encontrado")
        name, mime = names[kind]
        if not (job.folder / name).is_file():
            raise HTTPException(404, "Não há frame com detecção nesta análise")
        return FileResponse(
            job.folder / name,
            media_type=mime,
            filename=name,
            content_disposition_type="inline" if kind == "video" else "attachment",
        )

    return router
