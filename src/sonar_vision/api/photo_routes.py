"""Photograph jobs share the video manager's single native-process budget."""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from sonar_vision.api.video_jobs import VideoJobManager
from sonar_vision.processing.local_video import LocalVideoError
from sonar_vision.processing.photograph import MAX_IMAGE, validate_image_name


def photo_router(manager: VideoJobManager) -> APIRouter:
    router = APIRouter(prefix="/api/video-demo/photo")

    @router.get("/example")
    def example() -> FileResponse:
        path = manager.root.parent / "practical-examples/photograph-real.jpg"
        if not path.is_file():
            raise HTTPException(404, "Fotografia de exemplo ausente")
        return FileResponse(path, media_type="image/jpeg", filename="photograph-real.jpg")

    @router.post("/jobs", status_code=202)
    async def upload(request: Request, filename: str = "") -> dict[str, Any]:
        mime = request.headers.get("content-type", "").split(";")[0].lower()
        try:
            validate_image_name(filename, mime, request.headers.get("x-sonar-media-permission", ""))
        except LocalVideoError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not manager.weights.is_file():
            raise HTTPException(503, "Pesos BR V1 ausentes")
        try:
            job = manager.reserve()
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc
        try:
            count = 0
            async with asyncio.timeout(30):
                with (job.folder / "source-image").open("xb") as stream:
                    async for chunk in request.stream():
                        count += len(chunk)
                        if count > MAX_IMAGE:
                            raise HTTPException(413, "Fotografia excede12MiB")
                        stream.write(chunk)
            with (job.folder / "source-image").open("rb") as stream:
                head = stream.read(8)
            magic = b"\xff\xd8\xff" if mime == "image/jpeg" else b"\x89PNG\r\n\x1a\n"
            if count < 16 or not head.startswith(magic):
                raise HTTPException(422, "Conteúdo não corresponde ao formato da fotografia")
            job.data.update(filename=filename, mode="static_photograph")
            (job.folder / "job-config.json").write_text(json.dumps({"mode": "static_photograph"}))
            manager.start(job)
            return job.snapshot()
        except TimeoutError as exc:
            manager.discard_upload(job)
            raise HTTPException(408, "Upload excedeu30segundos") from exc
        except BaseException:
            manager.discard_upload(job)
            raise

    @router.get("/jobs/{id}/files/{kind}")
    def result(id: str, kind: str) -> FileResponse:
        try:
            job = manager.get(id)
        except KeyError as exc:
            raise HTTPException(404, "Análise não encontrada") from exc
        if job.state != "COMPLETED" or job.data.get("mode") != "static_photograph":
            raise HTTPException(409, "Resultado de fotografia ainda não disponível")
        names = {
            "original": ("photo-original.png", "image/png"),
            "annotated": ("photo-annotated.jpg", "image/jpeg"),
            "report": ("photo-report.json", "application/json"),
        }
        if kind.startswith("roi-") and kind[4:].isdigit() and 0 <= int(kind[4:]) < 16:
            names[kind] = (f"roi-{int(kind[4:])}.png", "image/png")
        if kind not in names or not (job.folder / names[kind][0]).is_file():
            raise HTTPException(404, "Arquivo não encontrado")
        name, mime = names[kind]
        return FileResponse(job.folder / name, media_type=mime)

    return router
