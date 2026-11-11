"""Render the actual Markdown report to an A4 PDF, without remote assets."""

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]


def inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#225577">\1</link>', text)
    return text


def main() -> None:
    source = ROOT / "docs/REPORT.md"
    destination = ROOT / "reports/SONAR_VISION_FIAP.pdf"
    destination.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            spaceAfter=8,
            alignment=TA_LEFT,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableBody", fontName="Helvetica", fontSize=8, leading=11, wordWrap="CJK"
        )
    )
    story = []
    lines = source.read_text().splitlines()
    index = 0
    in_code = False
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not line.strip():
            continue
        image = re.fullmatch(r"!\[.*?\]\(([^)]+)\)", line)
        if image:
            path = (source.parent / image[1]).resolve()
            if path.is_file() and path.is_relative_to(ROOT):
                figure = Image(str(path))
                factor = min(16 * cm / figure.imageWidth, 10 * cm / figure.imageHeight)
                figure.drawWidth = figure.imageWidth * factor
                figure.drawHeight = figure.imageHeight * factor
                story.extend([figure, Spacer(1, 8)])
            continue
        if line.startswith("|"):
            table_lines = [line]
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = [
                [
                    Paragraph(inline(cell.strip()), styles["TableBody"])
                    for cell in row.strip("|").split("|")
                ]
                for row in table_lines
                if not re.fullmatch(r"[|\-: ]+", row)
            ]
            table = Table(rows, colWidths=[16 * cm / len(rows[0])] * len(rows[0]), repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf0")),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9baeb7")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([table, Spacer(1, 10)])
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["Heading1"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["Heading2"]))
        else:
            if not in_code:
                while (
                    index < len(lines)
                    and lines[index].strip()
                    and not re.match(r"\d+\. ", lines[index])
                    and not lines[index].startswith(("#", "|", "- ", "```", "!["))
                ):
                    line += " " + lines[index]
                    index += 1
            story.append(Paragraph(inline(line), styles["ReportBody"]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#4b616c"))
        canvas.drawString(2.5 * cm, 1.5 * cm, "SONAR VISION | FIAP 2026.2 | Protótipo acadêmico")
        canvas.drawRightString(18.5 * cm, 1.5 * cm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(destination),
        pagesize=(21 * cm, 29.7 * cm),
        rightMargin=2.5 * cm,
        leftMargin=2.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2.2 * cm,
        title="SONAR VISION — Relatório técnico FIAP 2026.2",
        author="Equipe SONAR VISION — FIAP",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination)


if __name__ == "__main__":
    main()
