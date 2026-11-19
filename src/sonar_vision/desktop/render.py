"""Video-first YOLO canvas; only detections of the displayed frame are drawn."""

import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sonar_vision.desktop.profiles import Profile
from sonar_vision.processing.roi import Frame

BUS_COLOR = (36, 227, 109)
SIGN_COLOR = (255, 214, 41)


@lru_cache(maxsize=3)
def ui_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def place(canvas: Frame, image: Frame, x: int, y: int, w: int, h: int) -> None:
    scale = min(w / image.shape[1], h / image.shape[0])
    iw, ih = max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))
    resized = cv2.resize(image, (iw, ih))
    canvas[
        y + (h - ih) // 2 : y + (h - ih) // 2 + ih, x + (w - iw) // 2 : x + (w - iw) // 2 + iw
    ] = resized


def label(draw: Any, text: str, x: float, y: float, color: tuple[int, int, int]) -> None:
    font = ui_font(17)
    bounds = draw.textbbox((0, 0), text, font=font)
    width, height = bounds[2] + 10, bounds[3] - bounds[1] + 10
    left = max(0, min(int(x), 1280 - width))
    top = max(36, min(int(y), 680 - height))
    draw.rectangle((left, top, left + width, top + height), fill=(12, 17, 20))
    draw.text((left + 5, top + 3 - bounds[1]), text, font=font, fill=color)


def render(profile: Profile, result: dict[str, Any] | None, status: str, paused: bool) -> Frame:
    canvas = np.full((720, 1280, 3), (22, 22, 22), dtype=np.uint8)
    row = result["row"] if result else {"findings": []}
    scale, offset_x, offset_y = 1.0, 0.0, 0.0
    if result is not None:
        frame = result.get("frame")
        if frame is None:
            # Compatibility for older receipts; current worker always supplies
            # untouched pixels and never uses a previous frame's annotations.
            frame = result["annotated"][:-140]
        scale = min(1280 / frame.shape[1], 720 / frame.shape[0])
        offset_x = (1280 - int(frame.shape[1] * scale)) // 2
        offset_y = (720 - int(frame.shape[0] * scale)) // 2
        place(canvas, frame, 0, 0, 1280, 720)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    def coordinates(box: dict[str, float]) -> tuple[float, float, float, float]:
        return (
            offset_x + box["x1"] * scale,
            offset_y + box["y1"] * scale,
            offset_x + box["x2"] * scale,
            offset_y + box["y2"] * scale,
        )

    technical = bool(result and result.get("technical"))
    for bus in row.get("buses", []):
        x1, y1, x2, y2 = coordinates(bus["box"])
        draw.rectangle((x1, y1, x2, y2), outline=BUS_COLOR, width=3)
        if not technical:
            label(draw, f"ÔNIBUS {bus['confidence']:.3f}", x1, y1 - 28, BUS_COLOR)
    for finding in row["findings"]:
        x1, y1, x2, y2 = coordinates(finding["box"])
        draw.rectangle((x1, y1, x2, y2), outline=SIGN_COLOR, width=3)
        if not technical:
            label(draw, f"LETREIRO {finding['detector_confidence']:.3f}", x1, y1 - 28, SIGN_COLOR)
            raw = " | ".join(dict.fromkeys(r["text"] for r in finding["readings"]))
            text = "OCR: " + (raw or "sem texto reconhecido")
            for i, line in enumerate(textwrap.wrap(text, 68)[:3]):
                label(draw, line, x1, y2 + 5 + i * 28, SIGN_COLOR)
    if technical and result is not None:
        from sonar_vision.desktop.label_layout import draw_labels

        result["label_layout"] = draw_labels(draw, row, coordinates, ui_font(17))
    draw.rectangle((0, 0, 1280, 33), fill=(12, 17, 20))
    presentation = bool(result and result.get("presentation"))
    heading = (
        (
            "YOLO + OCR — inferência real pré-processada | "
            + profile.city
            + " | V3 ônibus + BR V1 letreiro"
        )
        if presentation
        else (f"SONAR | {profile.city} | {profile.model}")
    )
    draw.text((12, 6), heading, font=ui_font(17), fill="white")
    if result is None:
        text = "AGUARDANDO INFERÊNCIA" if profile.video else profile.limitation
        for i, line in enumerate(textwrap.wrap(text, 90)):
            draw.text((30, 310 + 30 * i), line, font=ui_font(19), fill="white")
    index = result["index"] if result else "—"
    state = "INCERTO · sem confirmação temporal" if result else status[:90]
    if technical and result is not None:
        evidence = result["evidence"]
        state = evidence["state"] + " · linha operacional não verificada"
        if evidence["stable_codes"]:
            state += " · " + ", ".join(c["code"] for c in evidence["stable_codes"])
    draw.rectangle((0, 680, 1280, 720), fill=(12, 17, 20))
    rate = (
        f"{result['source_fps']:.2f} FPS · todos os frames inferidos"
        if presentation and result
        else "5 frames/s amostrados"
    )
    draw.text(
        (12, 682),
        f"{state} | frame {index} | {'PAUSADO' if paused else rate}",
        font=ui_font(15),
        fill="white",
    )
    draw.text(
        (12, 702),
        (
            "ESPAÇO pausa · R reinicia · N próximo · 1 SP · 2 Santos A · "
            "3 Santos B · S salva · " + ("M áudio · " if technical else "") + "Q/ESC sair"
            if presentation
            else "ESPAÇO pausa · R reinicia · N próximo · S salva · Q/ESC sair · 8 composição V3+V1"
        ),
        font=ui_font(13),
        fill="#b8cdd2",
    )
    return cast(Frame, cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
