"""Prepare every frame once; render and encode at the source clock, never interpolate."""

import hashlib
import inspect
import json
import math
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import cv2

from sonar_vision.desktop.profiles import Profile
from sonar_vision.desktop.render import render
from sonar_vision.processing.roi import Frame
from sonar_vision.training.dataset import sha256

ROOT = Path("artifacts/demo/sonar-video-v3/final-presentation")
BR = Path("sonar_br/artifacts/train-v1/run/weights/best.pt")
BUS = Path("artifacts/demo/sonar-video-v3/training/full-f5829c9ab4/run/weights/best.pt")


@dataclass(frozen=True)
class ClipSpec:
    id: str
    source: str
    start_frame: int
    frames: int
    city: str


INBOX = "sonar_br/datasets/inbox/videos/"
SPECS = {
    "santos-a": ClipSpec("santos-a", INBOX + "11916524_1920_1080_60fps.mp4", 390, 600, "SANTOS A"),
    "santos-b": ClipSpec("santos-b", INBOX + "16606194_1920_1080_60fps.mp4", 0, 600, "SANTOS B"),
    "sao-paulo": ClipSpec("sao-paulo", INBOX + "videoplayback.mp4", 6570, 300, "SÃO PAULO"),
}


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def inspect_media(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("Vídeo ausente: " + str(path))
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        if (
            not ok
            or not math.isfinite(fps)
            or not 0.1 <= fps <= 120
            or count < 1
            or not 2 <= w <= 3840
            or not 2 <= h <= 2160
        ):
            raise ValueError("Vídeo inválido/não decodificável: " + str(path))
        return dict(fps=fps, frames=count, width=w, height=h, duration_seconds=count / fps)
    finally:
        cap.release()


def inference_pixels(frame: Frame) -> Frame:
    h, w = frame.shape[:2]
    scale = min(1, 1600 / max(h, w))
    return cast(
        Frame, cv2.resize(frame, (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)))
    )


class TimedOCR:
    def __init__(self, inner: Any):
        self.inner = inner
        self.seconds = 0.0
        self.calls = 0

    def read(self, roi: Frame) -> Any:
        start = time.perf_counter()
        try:
            return self.inner.read(roi)
        finally:
            self.seconds += time.perf_counter() - start
            self.calls += 1


class Encoder:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        import imageio_ffmpeg

        self.path = path
        self.log = path.with_suffix(".ffmpeg.log").open("wb")
        rate = str(Fraction(fps).limit_denominator(100000))
        self.process = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                rate,
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-threads",
                "2",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=self.log,
        )

    def write(self, frame: Frame) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Encoder sem pipe")
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        code = self.process.wait(timeout=120)
        self.log.close()
        if code:
            raise ValueError(f"FFmpeg exit{code}; ver {self.path.with_suffix('.ffmpeg.log')}")


def verify_prepared(directory: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "READY":
        raise ValueError("Pré-processamento incompleto")
    for row in manifest["files"]:
        path = directory / row["name"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError("Artefato alterado: " + str(path))
    return manifest


def prepare_clip(
    spec: ClipSpec,
    directory: Path,
    *,
    analyzer_factory: Callable[[], Any] | None = None,
    model_files: dict[str, Path] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if (directory / "manifest.json").exists():
        return verify_prepared(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "frames"
    cache.mkdir(exist_ok=True)
    started = time.perf_counter()
    source = Path(spec.source)
    meta = inspect_media(source)
    if spec.start_frame < 0 or spec.frames < 1 or spec.start_frame + spec.frames > meta["frames"]:
        raise ValueError("Intervalo fora do vídeo")
    models = model_files if model_files is not None else dict(bus=BUS, sign=BR)
    binding = dict(
        spec=asdict(spec),
        source_sha256=sha256(source),
        source_metadata=meta,
        weights={k: dict(path=str(p), sha256=sha256(p)) for k, p in models.items()},
        imgsz=960,
        confidence=0.35,
        iou=0.5,
        all_frames=True,
        interpolation=False,
        analyzer="V3 bus + BRV1 sign + E3/desktop direct fallback",
        implementation_sha256=sha256(Path(__file__)),
        renderer_sha256=sha256(Path(inspect.getfile(render))),
        evidence_kind="software_fixture" if analyzer_factory else "real_model_inference",
    )
    binding_file = directory / "binding.json"
    if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
        raise ValueError("Configuração mudou; preserve cache e escolha outro diretório")
    if not binding_file.exists():
        atomic_json(binding_file, binding)
    analyzer = None
    timed = None
    rows = []
    reused = 0
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, spec.start_frame)
    try:
        for index in range(spec.frames):
            ok, raw = cap.read()
            if not ok:
                raise ValueError(f"Frame corrompido: {index}")
            frame = inference_pixels(cast(Frame, raw))
            digest = hashlib.sha256(frame.tobytes()).hexdigest()
            path = cache / f"{index:06d}.json"
            if path.exists():
                record = json.loads(path.read_text())
                reused += 1
                if record["input_sha256"] != digest:
                    raise ValueError("Pixels/cache desalinhados")
            else:
                if analyzer is None:
                    if analyzer_factory:
                        analyzer = analyzer_factory()
                    else:
                        from sonar_vision.desktop.engine import DesktopOCR
                        from sonar_vision.desktop.v3 import make_analyzer

                        analyzer = make_analyzer(BR, 960, BUS)
                        analyzer.ocr = DesktopOCR(analyzer.ocr)
                    timed = TimedOCR(analyzer.ocr)
                    analyzer.ocr = timed
                assert timed is not None
                timed.seconds = 0.0
                timed.calls = 0
                before = time.perf_counter()
                _, row = analyzer.analyze(frame)
                row["stage_timings"] = dict(
                    row.get("stage_timings", {}),
                    ocr_seconds=timed.seconds,
                    ocr_calls=timed.calls,
                    analysis_seconds=time.perf_counter() - before,
                )
                record = dict(
                    index=index,
                    source_index=spec.start_frame + index,
                    timestamp_seconds=index / meta["fps"],
                    source_timestamp_seconds=(spec.start_frame + index) / meta["fps"],
                    input_sha256=digest,
                    row=row,
                )
                atomic_json(path, record)
            if record["index"] != index or record["source_index"] != spec.start_frame + index:
                raise ValueError("Índice/cache desalinhado")
            rows.append(record)
            if index % 60 == 0 or index == spec.frames - 1:
                atomic_json(
                    directory / "progress.json",
                    dict(stage="inference", completed=index + 1, total=spec.frames, reused=reused),
                )
                if progress:
                    progress(
                        f"{spec.id}: {index + 1}/{spec.frames} frames reais; "
                        f"cache reutilizado{reused}"
                    )
    finally:
        cap.release()
    inference_wall = time.perf_counter() - started
    profile = Profile(
        spec.id,
        spec.city,
        "V3 ônibus + BR V1 letreiro",
        str(BR),
        str(source),
        960,
        "Vídeo pessoal do usuário",
        "Declarado pelo usuário",
        "ML autorizado",
        "Sem confirmação temporal",
    )
    token = uuid4().hex[:8]
    ann_temp = directory / f"annotated-{token}.mp4"
    ref_temp = directory / f"reference-{token}.mp4"
    ann = Encoder(ann_temp, 1280, 720, meta["fps"])
    ref = Encoder(ref_temp, meta["width"], meta["height"], meta["fps"])
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, spec.start_frame)
    render_seconds = 0.0
    encoding_start = time.perf_counter()
    try:
        for record in rows:
            ok, raw = cap.read()
            if not ok:
                raise ValueError("Frame indisponível na composição")
            frame = inference_pixels(cast(Frame, raw))
            if hashlib.sha256(frame.tobytes()).hexdigest() != record["input_sha256"]:
                raise ValueError("Previsões não pertencem ao frame da composição")
            before = time.perf_counter()
            canvas = render(
                profile,
                dict(
                    frame=frame,
                    index=record["index"],
                    row=record["row"],
                    presentation=True,
                    source_fps=meta["fps"],
                ),
                "Pré-processado",
                False,
            )
            render_seconds += time.perf_counter() - before
            ann.write(canvas)
            ref.write(cast(Frame, raw))
    finally:
        cap.release()
        ann.close()
        ref.close()
    for temp, name in [(ann_temp, "annotated.mp4"), (ref_temp, "reference.mp4")]:
        final = directory / name
        if final.exists():
            final.rename(directory / (name + ".unsealed-" + token))
        os.replace(temp, final)
    with (directory / "predictions.jsonl").open("w") as stream:
        for record in rows:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    final_meta = inspect_media(directory / "annotated.mp4")
    reference_meta = inspect_media(directory / "reference.mp4")
    duration = spec.frames / meta["fps"]
    if any(
        abs(m["fps"] - meta["fps"]) > 0.01
        or m["frames"] != spec.frames
        or abs(m["duration_seconds"] - duration) > 1 / meta["fps"]
        for m in [final_meta, reference_meta]
    ):
        raise ValueError("FPS/duração/contagem alterados na codificação")
    timings = {
        name: sum(r["row"].get("stage_timings", {}).get(name, 0.0) for r in rows)
        for name in [
            "yolo_bus_seconds",
            "yolo_sign_seconds",
            "ocr_seconds",
            "ocr_calls",
            "analysis_seconds",
        ]
    }
    manifest = dict(
        status="READY",
        spec=asdict(spec),
        binding=binding,
        source=meta,
        annotated=final_meta,
        reference=reference_meta,
        duration_seconds=duration,
        all_frames_inferred=True,
        interpolation=False,
        reused_frames=reused,
        counts=dict(
            frames=spec.frames,
            bus=sum(bool(r["row"].get("buses")) for r in rows),
            panel=sum(bool(r["row"]["findings"]) for r in rows),
            OCR_text=sum(any(f["readings"] for f in r["row"]["findings"]) for r in rows),
        ),
        raw_texts=sorted(
            {v["text"] for r in rows for f in r["row"]["findings"] for v in f["readings"]}
        ),
        stage_seconds=timings,
        render_seconds=render_seconds,
        inference_wall_seconds=inference_wall,
        encoding_wall_seconds=time.perf_counter() - encoding_start,
        total_wall_seconds=time.perf_counter() - started,
        independent_accuracy_claim=False,
        temporal_confirmation=False,
        files=[
            dict(name=name, sha256=sha256(directory / name))
            for name in ["binding.json", "annotated.mp4", "reference.mp4", "predictions.jsonl"]
        ],
    )
    atomic_json(directory / "manifest.json", manifest)
    if progress:
        progress(f"{spec.id}: READY, {spec.frames}frames, {meta['fps']:.6f}FPS, {duration:.6f}s")
    return manifest
