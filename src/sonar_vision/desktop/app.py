"""Cocoa main-thread GUI, bounded single-process inference, reproducible receipts."""

import argparse
import hashlib
import json
import os
import queue
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from sonar_vision.desktop.engine import Engine, next_sample
from sonar_vision.desktop.profiles import Profile, profiles, validate_local
from sonar_vision.desktop.render import render
from sonar_vision.processing.local_video import inspect_video

TITLE = "SONAR VISION — LIVE COMPUTER VISION DEMO"


class Session:
    def __init__(self, root: Path, profile: Profile, output: Path):
        self.profile, self.output = profile, output
        output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.result: dict[str, Any] | None = None
        self.status = "Carregando modelo local…" if profile.video else profile.limitation
        self.ready, self.inflight, self.eof = False, False, False
        self.available_at = 0.0
        self.error: str | None = None
        self.cap: Any = None
        self.engine: Engine | None = None
        self.writer: Any = None
        self.log = (output / "frames.jsonl").open("w")
        self.meta: dict[str, Any] = {}
        self.stats: dict[str, Any] = {
            "profile": profile.record(),
            "frames_processed": 0,
            "displayed_new_frame_times": [],
            "frames_with_detection": 0,
            "frames_with_bus_detection": None,
            "frames_with_OCR_text": 0,
            "OCR_calls": 0,
            "direct_OCR_calls": 0,
            "eligible_codes": [],
            "raw_texts": [],
            "inference_seconds": 0.0,
            "first_detection_frame": None,
            "temporal_confirmation": False,
            "independent_accuracy": None,
            "model_loaded": False,
            "error": None,
        }
        validate_local(root, profile)
        if profile.video:
            path = root / profile.video
            self.meta = inspect_video(path)
            self.stats["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.stats["metadata"] = self.meta
            self.cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
            self.engine = Engine(
                root / profile.weights,
                profile.imgsz,
                root / profile.bus_weights if profile.bus_weights else None,
            )
        else:
            self.stats["unavailable"] = profile.limitation

    def tick(self, paused: bool) -> bool:
        if not self.engine or self.error or paused:
            return False
        try:
            msg = self.engine.outgoing.get_nowait()
        except queue.Empty:
            msg = None
        if msg:
            if msg["kind"] == "error":
                self.error = msg["error"]
                self.status = "ERRO: " + self.error
                return False
            if msg["kind"] == "ready":
                self.ready = True
                self.stats.update(model_loaded=True, model_version=msg["model_version"])
            if msg["kind"] == "frame":
                self.result = msg
                self.inflight = False
                self.available_at = time.monotonic() + 0.2
                row = msg["row"]
                findings = row["findings"]
                self.stats["frames_processed"] += 1
                self.stats["frames_with_detection"] += bool(findings)
                if "buses" in row:
                    self.stats["frames_with_bus_detection"] = (
                        self.stats["frames_with_bus_detection"] or 0
                    ) + bool(row["buses"])
                self.stats["OCR_calls"] += len(findings)
                self.stats["direct_OCR_calls"] += sum(
                    f["recognizer"].startswith("PP-OCR") for f in findings
                )
                self.stats["frames_with_OCR_text"] += any(f["readings"] for f in findings)
                self.stats["inference_seconds"] += row["inference_seconds"]
                for f in findings:
                    self.stats["raw_texts"] += [r["text"] for r in f["readings"]]
                    if f["code"]:
                        self.stats["eligible_codes"].append(f["code"])
                self.log.write(
                    json.dumps(
                        {"frame": msg["index"], "timestamp": msg["timestamp"], **row},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self.log.flush()
                self.status = "Frame inferido; próximo será analisado sem reutilizar caixas"
                canvas = render(self.profile, self.result, self.status, False)
                if self.writer is None:
                    self.writer = cv2.VideoWriter(
                        str(self.output / "encoding.mp4"),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        5,
                        (1280, 720),
                    )
                    if not self.writer.isOpened():
                        raise RuntimeError("Encoder não disponível")
                self.writer.write(canvas)
                if findings and self.stats["first_detection_frame"] is None:
                    self.stats["first_detection_frame"] = msg["index"]
                    cv2.imwrite(str(self.output / "first-detection-window.png"), canvas)
                cv2.imwrite(str(self.output / "last-window.png"), canvas)
                return True
        if (
            self.ready
            and not self.inflight
            and not self.eof
            and time.monotonic() >= self.available_at
        ):
            sample = next_sample(
                self.cap, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)), self.meta["fps"]
            )
            if sample is None:
                self.eof = True
            else:
                self.engine.incoming.put(sample, timeout=1)
                self.inflight = True
        if time.monotonic() - self.started > 180:
            self.error = "Orçamento de180segundos excedido"
        return False

    def close(self) -> dict[str, Any]:
        if self.engine:
            self.engine.close()
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
            import imageio_ffmpeg

            try:
                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-nostdin",
                        "-v",
                        "error",
                        "-i",
                        str(self.output / "encoding.mp4"),
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
                        str(self.output / "annotated.mp4"),
                    ],
                    check=True,
                    timeout=30,
                )
                (self.output / "encoding.mp4").unlink()
            except (subprocess.SubprocessError, OSError) as exc:
                self.stats["export_error"] = type(exc).__name__
        self.log.close()
        self.stats.update(error=self.error, total_seconds=time.monotonic() - self.started)
        seconds = self.stats["inference_seconds"]
        self.stats["effective_analysis_fps"] = (
            self.stats["frames_processed"] / seconds if seconds else 0
        )
        self.stats["raw_texts"] = sorted(set(self.stats["raw_texts"]))
        self.stats["eligible_codes"] = sorted(set(self.stats["eligible_codes"]))
        (self.output / "report.json").write_text(
            json.dumps(self.stats, ensure_ascii=False, indent=2) + "\n"
        )
        return self.stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["presentation", "live"], default="presentation")
    parser.add_argument(
        "city",
        nargs="?",
        default="best",
        choices=[
            "best",
            "all",
            "sao-paulo",
            "curitiba",
            "bangkok",
            "santos-a",
            "santos-b",
            "sao-paulo-v3",
            "santos-v3",
            "santos-yolo",
        ],
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Janela real; termina após os perfis solicitados"
    )
    parser.add_argument(
        "--test-controls",
        action="store_true",
        help="Atalhos programados no mesmo dispatch da GUI, sem simular inferência",
    )
    sound = parser.add_mutually_exclusive_group()
    sound.add_argument("--audio", dest="audio", action="store_true")
    sound.add_argument("--mute", dest="audio", action="store_false")
    parser.set_defaults(audio=True)
    parser.add_argument("--presentation-version", choices=["final", "historical"], default="final")
    args = parser.parse_args()
    if args.mode == "presentation" and args.city in {
        "best",
        "all",
        "sao-paulo",
        "santos-a",
        "santos-b",
        "santos-yolo",
    }:
        from sonar_vision.desktop.catalog import ROOT as FINAL_ROOT
        from sonar_vision.desktop.catalog import final_directories
        from sonar_vision.desktop.player import run_presentation

        directories = final_directories() if args.presentation_version == "final" else None
        selected = "sao-paulo" if directories and args.city == "best" else args.city
        run_presentation(
            selected,
            smoke=args.smoke,
            test_controls=args.test_controls,
            prepared_directories=directories,
            output_root=FINAL_ROOT if directories else None,
            audio=args.audio,
        )
        return
    root = Path.cwd()
    import imageio_ffmpeg
    from PIL import ImageFont

    ImageFont.load_default(size=19)
    imageio_ffmpeg.get_ffmpeg_exe()
    if "COCOA" not in cv2.getBuildInformation() and "QT" not in cv2.getBuildInformation():
        raise SystemExit("OpenCV sem interface gráfica; restaure opencv-python com Cocoa")
    for name in ["PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec"]:
        if not all(
            (Path.home() / ".paddlex/official_models" / name / file).is_file()
            for file in ["inference.json", "inference.pdiparams", "inference.yml"]
        ):
            raise SystemExit("Cache OCR ausente; inicialização offline cancelada: " + name)
    choices = profiles()
    if args.city not in {"best", "all"} and args.city not in choices:
        raise SystemExit("Perfil V3 ainda não validado/ativado; use um perfil histórico")
    for profile in choices.values():
        validate_local(root, profile)
    out = root / "artifacts/demo/final-desktop"
    out.mkdir(parents=True, exist_ok=True)
    (out / "profiles.json").write_text(
        json.dumps([p.record() for p in choices.values()], ensure_ascii=False, indent=2) + "\n"
    )
    order = (
        (
            ["santos-a", "santos-b", "sao-paulo-v3"]
            if "santos-a" in choices
            else ["curitiba", "sao-paulo", "bangkok"]
        )
        if args.city == "best"
        else (list(choices) if args.city == "all" else [args.city])
    )
    if args.city == "best" and "santos-yolo" in choices:
        order = ["santos-yolo"]
    run = out / ("run-" + uuid4().hex[:10])
    run.mkdir()
    events = (run / "events.jsonl").open("w")
    reports = []

    def event(action: str, city: str, **extra: Any) -> None:
        events.write(json.dumps({"action": action, "city": city, **extra}) + "\n")
        events.flush()

    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, 1280, 720)
    position, serial = 0, 0
    session = None
    paused = False
    started = time.monotonic()
    schedule = [
        (6, 32),
        (7, 32),
        (9, ord("r")),
        (13, ord("3")),
        (16, ord("1")),
        (19, ord("2")),
        (21, ord("n")),
        (24, ord("s")),
        (26, ord("q")),
    ]
    try:
        while True:
            if session is None:
                selected = order[position]
                canvas = render(choices[selected], None, "Preparando inferência local…", False)
                cv2.imshow(TITLE, canvas)
                cv2.waitKey(1)
                session = Session(root, choices[selected], run / f"{serial:02d}-{selected}")
                serial += 1
                event(
                    "city_started_clear_previous_state",
                    selected,
                    PID=os.getpid(),
                    window_visible=cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE),
                )
                if session.profile.video is None:
                    cv2.imwrite(
                        str(session.output / "unavailable-window.png"),
                        render(session.profile, None, session.status, False),
                    )
            changed = session.tick(paused)
            if changed:
                session.stats["displayed_new_frame_times"].append(time.perf_counter())
            canvas = render(session.profile, session.result, session.status, paused)
            displays = session.stats["displayed_new_frame_times"]
            interval = displays[-1] - displays[0] if len(displays) > 1 else 0
            live_fps = (len(displays) - 1) / interval if interval else None
            session.stats["measured_display_fps"] = live_fps
            caption = f"LIVE {live_fps:.2f} FPS" if live_fps else "LIVE: aferindo FPS"
            cv2.putText(
                canvas, caption, (1000, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1
            )
            cv2.imshow(TITLE, canvas)
            key = cv2.waitKey(35) & 0xFF
            if 65 <= key <= 90:
                key += 32
            if args.test_controls and schedule and time.monotonic() - started >= schedule[0][0]:
                _, key = schedule.pop(0)
                event("scripted_control", session.profile.id, key=key)
            if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1 or key in [27, ord("q")]:
                event("quit", session.profile.id)
                break
            if key == 32:
                paused = not paused
                event("pause" if paused else "resume", session.profile.id)
            if key == ord("s"):
                cv2.imwrite(str(session.output / ("saved-" + uuid4().hex[:6] + ".png")), canvas)
                event("saved_current_displayed_frame", session.profile.id)
            switch = None
            if ord("1") <= key < ord("1") + len(choices):
                switch = list(choices)[key - ord("1")]
            elif key == ord("r"):
                switch = session.profile.id
                event("restart", switch)
            elif key == ord("n"):
                switch = list(choices)[(list(choices).index(session.profile.id) + 1) % len(choices)]
            finished = (
                bool(session.error)
                or session.eof
                or (
                    session.profile.video is None
                    and (args.smoke or len(order) > 1)
                    and time.monotonic() - session.started > 3
                )
            )
            if session.eof and not args.smoke and not args.test_controls and not paused:
                paused = True
                session.status = "Fim do vídeo — R reinicia; N próximo"
                event("end_of_video_hold", session.profile.id)
                cv2.imwrite(
                    str(session.output / "end-of-video-window.png"),
                    render(session.profile, session.result, session.status, True),
                )
            if finished and not paused and not args.test_controls:
                if args.smoke and position == len(order) - 1:
                    event("smoke_completed", session.profile.id)
                    break
                position = (position + 1) % len(order)
                switch = order[position]
            if switch:
                reports.append(session.close())
                session = None
                paused = False
                if switch not in order:
                    order.append(switch)
                position = order.index(switch)
    finally:
        if session:
            reports.append(session.close())
        events.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        (run / "summary.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
        print("Desktop evidence:", run, flush=True)
    if args.smoke and any(r.get("error") for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
