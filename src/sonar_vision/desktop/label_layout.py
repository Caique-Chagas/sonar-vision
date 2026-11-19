"""Place prediction labels around current panel ROIs; never move prediction boxes."""

import textwrap
from collections.abc import Callable
from typing import Any

Rect = tuple[float, float, float, float]


def intersects(a: Rect, b: Rect) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


class LabelLayout:
    def __init__(self, panels: list[Rect]):
        self.panels = panels
        self.occupied: list[Rect] = []

    def allocate(self, anchor: Rect, width: int, height: int) -> Rect | None:
        if width > 1272 or height > 636:
            return None
        x1, y1, x2, y2 = anchor
        positions = [(x1, y1 - height - 7), (x1, y2 + 7), (x2 + 8, y1), (x1 - width - 8, y1)]
        positions += [(x, y) for y in range(40, 650, 32) for x in range(8, 1250, 80)]
        for x, y in positions:
            left, top = max(4, min(x, 1276 - width)), max(37, min(y, 677 - height))
            rect = (left, top, left + width, top + height)
            if not any(intersects(rect, other) for other in self.panels + self.occupied):
                self.occupied.append(rect)
                return rect
        return None


def draw_labels(
    draw: Any, row: dict[str, Any], coordinates: Callable[..., Rect], font: Any
) -> dict[str, Any]:
    panels = [coordinates(f["box"]) for f in row["findings"]]
    layout = LabelLayout(panels)
    labels: list[dict[str, Any]] = []
    requests = []
    for f, box in zip(row["findings"], panels, strict=True):
        raw = " | ".join(dict.fromkeys(r["text"] for r in f["readings"]))
        lines = [f"LETREIRO {f['detector_confidence']:.3f} · detecção"]
        lines += textwrap.wrap("OCR: " + (raw or "sem texto reconhecido"), 38)[:3]
        requests.append((box, lines, (255, 214, 41), "panel"))
    for bus in row.get("buses", []):
        requests.append(
            (
                coordinates(bus["box"]),
                [f"ÔNIBUS {bus['confidence']:.3f} · detecção"],
                (36, 227, 109),
                "bus",
            )
        )
    skipped = 0
    for anchor, lines, color, kind in requests:
        width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) + 12
        height = len(lines) * 23 + 8
        rect = layout.allocate(anchor, width, height)
        if rect is None:
            skipped += 1
            continue
        x, y, _, _ = rect
        draw.rectangle(rect, fill=(12, 17, 20))
        for i, line in enumerate(lines):
            draw.text((x + 6, y + 3 + i * 23), line, font=font, fill=color)
        labels.append(dict(kind=kind, rect=rect, anchor=anchor, lines=lines))
    return dict(
        labels=labels,
        skipped=skipped,
        panel_occlusions=sum(intersects(tuple(item["rect"]), p) for item in labels for p in panels),
        label_overlaps=sum(
            intersects(tuple(a["rect"]), tuple(b["rect"]))
            for i, a in enumerate(labels)
            for b in labels[i + 1 :]
        ),
        boxes_origin="current_frame_detector",
        box_coordinates_modified=False,
    )
