"""Single native model process; E3 and optional direct recognition on predicted ROIs."""

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import cv2

from sonar_vision.core.contracts import Box, OCRReading
from sonar_vision.desktop.v3 import make_analyzer
from sonar_vision.processing.roi import Frame, extract_roi


class DesktopOCR:
    def __init__(self, e3: Any):
        self.e3 = e3
        self.calls: list[str] = []

    def read(self, roi: Frame) -> list[OCRReading]:
        self.calls.append("E3")
        readings: list[OCRReading] = list(self.e3.read(roi))
        if not readings:
            self.calls[-1] = "PP-OCRv5 direto — experimental (E3 vazio)"
            readings = [
                OCRReading(text=str(r["rec_text"]), confidence=float(r["rec_score"]))
                for r in self.e3.recognizer.predict(input=[roi], batch_size=1)
                if str(r["rec_text"]).strip()
            ]
        return readings


def process_frames(
    weights: str, imgsz: int, incoming: Any, outgoing: Any, bus_weights: str | None = None
) -> None:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        analyzer = make_analyzer(Path(weights), imgsz, Path(bus_weights) if bus_weights else None)
        ocr = DesktopOCR(analyzer.ocr)
        analyzer.ocr = ocr
        outgoing.put({"kind": "ready", "model_version": analyzer.detector.model_version})
        while True:
            message = incoming.get()
            if message is None:
                break
            index, timestamp, frame = message
            ocr.calls.clear()
            annotated, row = analyzer.analyze(frame)
            for finding, source in zip(row["findings"], ocr.calls, strict=True):
                finding["recognizer"] = source
            roi = (
                extract_roi(frame, Box.model_validate(row["findings"][0]["box"]))
                if row["findings"]
                else None
            )
            outgoing.put(
                {
                    "kind": "frame",
                    "index": index,
                    "timestamp": timestamp,
                    "annotated": annotated,
                    "frame": frame,
                    "row": row,
                    "roi": roi,
                }
            )
    except Exception as exc:
        outgoing.put({"kind": "error", "error": type(exc).__name__ + ": " + str(exc)[:240]})


class Engine:
    def __init__(self, weights: Path, imgsz: int, bus_weights: Path | None = None):
        ctx = mp.get_context("spawn")
        self.incoming = ctx.Queue(maxsize=1)
        self.outgoing = ctx.Queue(maxsize=1)
        self.process = ctx.Process(
            target=process_frames,
            args=(
                str(weights),
                imgsz,
                self.incoming,
                self.outgoing,
                str(bus_weights) if bus_weights else None,
            ),
            daemon=True,
        )
        self.process.start()

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        for channel in [self.incoming, self.outgoing]:
            channel.cancel_join_thread()
            channel.close()


def next_sample(cap: Any, index: int, source_fps: float) -> tuple[int, float, Frame] | None:
    """Sequential decode; no stale bounding boxes placed on skipped frames."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        current = index
        index += 1
        if current and int(current * 5 / source_fps) == int((current - 1) * 5 / source_fps):
            continue
        h, w = frame.shape[:2]
        scale = min(1, 1600 / max(h, w))
        frame = cv2.resize(
            frame, (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2))
        )
        return current, current / source_fps, frame
