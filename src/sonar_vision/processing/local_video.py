"""Bounded local-file video analysis; every overlay comes from current-frame inference."""

import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import cv2

from sonar_vision.detection.yolo import Detector
from sonar_vision.ocr.paddle import OCREngine
from sonar_vision.processing.roi import Frame, extract_roi, validate_frame
from sonar_vision.validation.normalizer import LineNormalizer

MAX_UPLOAD = 64 * 1024 * 1024
MAX_SECONDS = 60
MAX_WALL_SECONDS = 300
SAMPLE_FPS = 5


def demo_imgsz(profile: str) -> int:
    if profile not in {"baseline", "diagnostic1280"}:
        raise LocalVideoError("Perfil de demonstração desconhecido")
    return 1280 if profile == "diagnostic1280" else 640


class LocalVideoError(ValueError):
    pass


def validate_upload_name(filename: str, media_type: str, consent: str) -> None:
    if (
        not filename
        or len(filename) > 180
        or any(c in filename for c in ["/", "\\"])
        or any(ord(c) < 32 for c in filename)
        or Path(filename).suffix.lower() != ".mp4"
        or media_type.split(";")[0].lower() != "video/mp4"
    ):
        raise LocalVideoError("Selecione um arquivo MP4 válido")
    if consent != "confirmed":
        raise LocalVideoError("Confirme a autorização para processar o arquivo")


def inspect_video(path: Path) -> dict[str, Any]:
    if not 16 <= path.stat().st_size <= MAX_UPLOAD:
        raise LocalVideoError("Arquivo vazio ou maior que64MiB")
    with path.open("rb") as stream:
        head = stream.read(64)
    if head[4:8] != b"ftyp":
        raise LocalVideoError("Conteúdo não corresponde a um contêiner MP4")
    # Set in the isolated worker, never give a user URL to a decoder.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,pipe|enable_drefs;0"
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or not all(math.isfinite(x) for x in [fps, count, width, height]):
            raise LocalVideoError("Vídeo não decodificável")
        if not (0.1 <= fps <= 60 and count >= 1 and 2 <= width <= 3840 and 2 <= height <= 2160):
            raise LocalVideoError("Resolução ou FPS fora dos limites: até3840x2160 e60FPS")
        duration = count / fps
        if duration > MAX_SECONDS or count > 3600:
            raise LocalVideoError("Duração máxima:60segundos")
        ok, frame = capture.read()
        if not ok:
            raise LocalVideoError("Primeiro frame corrompido")
        validate_frame(cast(Frame, frame))
        return {
            "fps": fps,
            "frames": int(count),
            "width": int(width),
            "height": int(height),
            "duration_seconds": duration,
        }
    finally:
        capture.release()


class DemoFrameAnalyzer:
    """No requested route, reference code or manual box enters this analysis."""

    def __init__(self, detector: Detector, ocr: OCREngine, *, static: bool = False):
        self.detector, self.ocr = detector, ocr
        self.normalizer = LineNormalizer()
        self.static = static

    def analyze(self, frame: Frame) -> tuple[Frame, dict[str, Any]]:
        validate_frame(frame)
        annotated = frame.copy()
        start = time.perf_counter()
        findings: list[dict[str, Any]] = []
        for detection in self.detector.detect(frame):
            if detection.label != "destination_sign":
                continue
            error = None
            try:
                readings = self.ocr.read(extract_roi(frame, detection.box))
            except Exception:
                readings = []
                error = "OCR_FAILURE"
            if len(readings) > 32:
                readings = readings[:32]
                error = "OCR_READING_LIMIT"
            codes = {
                route
                for r in readings
                if r.confidence >= 0.7
                for route in [self.normalizer.normalize(r.text).route]
                if route
            }
            code = next(iter(codes)) if len(codes) == 1 and not error else None
            finding = {
                "box": detection.box.model_dump(),
                "detector_confidence": detection.confidence,
                "readings": [r.model_dump() for r in readings],
                "code": code,
                "state": "CODE_CANDIDATE" if code else "UNCERTAIN",
                "ocr_error": error,
            }
            findings.append(finding)
            box = detection.box
            cv2.rectangle(
                annotated, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 220, 220), 2
            )
            cv2.putText(
                annotated,
                f"YOLO {detection.confidence:.3f} | {code or 'INCERTO'}",
                (int(box.x1), max(20, int(box.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 220),
                1,
            )
        candidate = any(f["code"] for f in findings)
        state = "CODE_CANDIDATE" if candidate else ("UNCERTAIN" if findings else "NO_PANEL")
        # Enlarge only the display after inference so low-resolution clips retain
        # readable captions. Reported boxes remain in inference-frame coordinates.
        if annotated.shape[1] < 800:
            scale = 800 / annotated.shape[1]
            annotated = cast(
                Frame,
                cv2.resize(annotated, (800, max(2, int(annotated.shape[0] * scale) // 2 * 2))),
            )
        panel = cv2.copyMakeBorder(annotated, 0, 140, 0, 0, cv2.BORDER_CONSTANT, value=(22, 27, 35))
        labels = {
            "CODE_CANDIDATE": "Codigo candidato - sem confirmacao operacional",
            "UNCERTAIN": "Leitura incerta",
            "NO_PANEL": "Painel nao localizado",
        }
        lines = [
            "SONAR BR V1 | FOTOGRAFIA ESTATICA | NAO E VIDEO"
            if self.static
            else "SONAR BR V1 | VIDEO GRAVADO | FRAMES AMOSTRADOS | SEM CONFIRMACAO",
            labels[state],
        ]
        for f in findings[:2]:
            raw = " | ".join(r["text"] for r in f["readings"])
            scores = ",".join(f"{r['confidence']:.3f}" for r in f["readings"][:4])
            lines.append(f"OCR: {raw[:100] or '(sem texto)'} | scores OCR: {scores or '-'}")
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=15)
        for i, line in enumerate(lines):
            draw.text((12, annotated.shape[0] + 12 + i * 27), line, font=font, fill=(240, 240, 240))
        import numpy as np

        panel = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return cast(Frame, panel), {
            "findings": findings,
            "state": state,
            "inference_seconds": time.perf_counter() - start,
            "temporal_confirmation": False,
            "modality": "static_photograph" if self.static else "recorded_video",
        }


def analyze_video(
    path: Path, folder: Path, analyzer: DemoFrameAnalyzer, publish: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    import imageio_ffmpeg

    started = time.perf_counter()
    metadata = inspect_video(path)
    fps = min(SAMPLE_FPS, metadata["fps"])
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    writer = None
    interim, result = folder / "encoding.mp4", folder / "annotated.mp4"
    metrics = {
        "frames_processed": 0,
        "frames_with_detection": 0,
        "frames_with_code_candidate": 0,
        "uncertain_frames": 0,
        "frames_with_OCR_text": 0,
        "OCR_calls": 0,
        "OCR_failures": 0,
    }
    inference_seconds = 0.0
    last_sample = -1
    source_index = 0
    first_detection_timestamp = None
    first_detection = None
    try:
        with (folder / "frames.jsonl").open("w") as log:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if source_index < metadata["frames"] - 1:
                        raise LocalVideoError("Decodificação interrompida antes do fim anunciado")
                    break
                if source_index >= 3600 or source_index / metadata["fps"] >= MAX_SECONDS:
                    raise LocalVideoError("Frames reais excedem o limite de duração")
                validate_frame(cast(Frame, frame))
                sample = int(source_index * fps / metadata["fps"])
                timestamp = source_index / metadata["fps"]
                source_index += 1
                if sample == last_sample:
                    continue
                last_sample = sample
                scale = min(1, 1600 / max(frame.shape[:2]))
                width, height = (
                    max(2, int(frame.shape[1] * scale) // 2 * 2),
                    max(2, int(frame.shape[0] * scale) // 2 * 2),
                )
                frame = cast(Frame, cv2.resize(frame, (width, height)))
                annotated, row = analyzer.analyze(frame)
                if row["findings"] and first_detection_timestamp is None:
                    if not cv2.imwrite(str(folder / "first-detection.jpg"), annotated):
                        raise LocalVideoError("Não foi possível salvar o frame detectado")
                    first_detection_timestamp = timestamp
                    first_detection = {"state": row["state"], "findings": row["findings"]}
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(interim),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        fps,
                        (annotated.shape[1], annotated.shape[0]),
                    )
                    if not writer.isOpened():
                        raise LocalVideoError("Encoder intermediário indisponível")
                writer.write(annotated)
                metrics["frames_processed"] += 1
                metrics["frames_with_detection"] += bool(row["findings"])
                metrics["frames_with_code_candidate"] += row["state"] == "CODE_CANDIDATE"
                metrics["uncertain_frames"] += row["state"] != "CODE_CANDIDATE"
                metrics["frames_with_OCR_text"] += any(f["readings"] for f in row["findings"])
                metrics["OCR_calls"] += len(row["findings"])
                metrics["OCR_failures"] += sum(f["ocr_error"] is not None for f in row["findings"])
                inference_seconds += row["inference_seconds"]
                row.update(source_frame=source_index - 1, timestamp_seconds=timestamp)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                if log.tell() > 32 * 1024 * 1024 or interim.stat().st_size > 128 * 1024 * 1024:
                    raise LocalVideoError("Limite de armazenamento do processamento excedido")
                ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                publish(
                    {
                        "kind": "frame",
                        "metrics": dict(metrics),
                        "metadata": metadata,
                        "progress": min(0.95, source_index / metadata["frames"] * 0.95),
                        "jpeg": jpeg.tobytes() if ok else None,
                        "latest": row,
                    }
                )
                if time.perf_counter() - started > MAX_WALL_SECONDS - 45:
                    raise LocalVideoError("Prazo máximo de processamento excedido")
        if not metrics["frames_processed"]:
            raise LocalVideoError("Nenhum frame válido")
    finally:
        capture.release()
        if writer:
            writer.release()
    publish({"kind": "encoding"})
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(interim),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(result),
        ],
        check=True,
        timeout=45,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.stat().st_size > 128 * 1024 * 1024:
        raise LocalVideoError("Resultado excede limite de armazenamento")
    report = {
        "mode": "recorded_video_offline_sampled",
        "model_version": analyzer.detector.model_version,
        "OCR": "PaddleOCR original or E3, configured by worker",
        "metadata": metadata,
        "analysis_fps": fps,
        **metrics,
        "inference_seconds": inference_seconds,
        "effective_inference_fps": metrics["frames_processed"] / inference_seconds
        if inference_seconds
        else 0,
        "total_seconds": time.perf_counter() - started,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        "accuracy": None,
        "ground_truth_used": False,
        "temporal_confirmation": False,
        "audio": "removed; human narration recorded separately",
        "first_detection_timestamp_seconds": first_detection_timestamp,
        "first_detection": first_detection,
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    interim.unlink(missing_ok=True)
    return report


def create_demo_analyzer(
    weights: Path, imgsz: int = 640, *, static: bool = False
) -> DemoFrameAnalyzer:
    """Same BR V1/E3 configuration for both modalities; no reference data."""
    for name in ["en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_det"]:
        if not (Path.home() / ".paddlex/official_models" / name).is_dir():
            raise LocalVideoError("Modelos OCR locais ausentes; consulte o guia")
    from paddleocr import TextRecognition

    from sonar_vision.br.ocr_optimization import ExperimentalOCR, OCRExperiment
    from sonar_vision.detection.yolo import YoloSignDetector
    from sonar_vision.ocr.paddle import PaddleEngine

    detector = YoloSignDetector(weights, confidence=0.35, imgsz=imgsz)
    detector.model.overrides.update(device="cpu", iou=0.5, rect=True, quantize=None, batch=1)
    baseline = PaddleEngine()
    rec = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(Path.home() / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    return DemoFrameAnalyzer(
        detector, ExperimentalOCR(baseline, OCRExperiment(spatial="recognize"), rec), static=static
    )


def video_worker(folder_name: str, weights_name: str, output: Any) -> None:
    """Spawn boundary limits native decoder/model lifetime and client-visible errors."""
    folder = Path(folder_name)
    started = time.perf_counter()
    try:
        if os.name == "posix":
            os.setsid()
        config = folder / "job-config.json"
        settings = json.loads(config.read_text()) if config.exists() else {"profile": "baseline"}
        if settings.get("mode") == "static_photograph":
            from sonar_vision.processing.photograph import analyze_photograph, decode_photograph

            image = decode_photograph(folder / "source-image")
            analyzer = create_demo_analyzer(Path(weights_name), static=True)
            report = analyze_photograph(image, folder, analyzer)
            report["total_seconds_including_models"] = time.perf_counter() - started
            (folder / "photo-report.json").write_text(json.dumps(report, indent=2) + "\n")
            output.put({"kind": "done", "report": report}, timeout=10)
            return
        inspect_video(folder / "source.mp4")
        profile = settings["profile"]
        imgsz = demo_imgsz(profile)
        analyzer = create_demo_analyzer(Path(weights_name), imgsz)
        initialization = time.perf_counter() - started
        report = analyze_video(
            folder / "source.mp4", folder, analyzer, lambda m: output.put(m, timeout=10)
        )
        report.update(
            profile=profile,
            OCR="E3: PaddleOCR original + spatial recognition en_PP-OCRv5_mobile_rec",
            detector_parameters={
                "confidence": 0.35,
                "imgsz": imgsz,
                "iou": 0.5,
                "rect": True,
                "device": "cpu",
                "batch": 1,
                "tiles": False,
                "roi_padding": 0,
            },
            model_initialization_seconds=initialization,
            total_seconds_including_models=time.perf_counter() - started,
        )
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        output.put({"kind": "done", "report": report}, timeout=10)
    except Exception as exc:
        output.put(
            {
                "kind": "error",
                "category": type(exc).__name__,
                "error": str(exc)
                if isinstance(exc, LocalVideoError)
                else "Falha ao carregar modelos, decodificar ou codificar vídeo",
            },
            timeout=5,
        )
