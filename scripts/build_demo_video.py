"""Render an honest narrated pilot overview using real evidence and macOS TTS.

The simulation chapter is explicitly labeled. No temporal route confirmation
is fabricated from repeated still images. All inputs and outputs stay local.
"""

import json
import math
import shutil
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/demo"
BUILD = OUT / "build"
SIZE = (1600, 900)
BG = "#101a20"
TEXT = "#f2f4ef"
MUTED = "#b7c3c7"
ACCENT = "#f6cb79"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)


def font(size: int) -> ImageFont.FreeTypeFont:
    from matplotlib import get_data_path

    return ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), size)


def slide(title: str, label: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    draw.text((65, 30), "SONAR VISION", font=font(23), fill=ACCENT)
    draw.text((65, 88), title, font=font(42), fill=TEXT)
    draw.line((65, 155, 1535, 155), fill="#34444d", width=2)
    draw.text((65, 840), label, font=font(19), fill=MUTED)
    return image, draw


def lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int = 62,
    size: int = 28,
    color: str = TEXT,
) -> int:
    x, y = xy
    for paragraph in text.split("\n"):
        for row in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), row, font=font(size), fill=color)
            y += size + 13
        y += 8
    return y


def paste(image: Image.Image, path: Path, rectangle: tuple[int, int, int, int]) -> None:
    x, y, w, h = rectangle
    source = Image.open(path).convert("RGB")
    fitted = ImageOps.contain(source, (w, h))
    image.paste(fitted, (x + (w - fitted.width) // 2, y + (h - fitted.height) // 2))


def inference_evidence() -> dict:
    from sonar_vision.detection.yolo import YoloSignDetector
    from sonar_vision.ocr.paddle import PaddleEngine
    from sonar_vision.processing.roi import extract_roi
    from sonar_vision.training.dataset import verify_dataset

    verify_dataset(ROOT / "data/bangkok-pilot/data.yaml")
    path = ROOT / "data/bangkok-pilot/test/images/00068.jpg"
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError("Restaure a imagem de teste auditada antes de gerar o vídeo")
    detector = YoloSignDetector(ROOT / "models/best.pt")
    engine = PaddleEngine()
    detections = detector.detect(frame)
    if not detections:
        raise RuntimeError(
            "O modelo não detectou a imagem selecionada; não usar resultado fictício"
        )
    records = []
    for i, detection in enumerate(detections):
        crop = extract_roi(frame, detection.box)
        readings = engine.read(crop)
        crop_path = BUILD / f"real-roi-{i}.png"
        cv2.imwrite(str(crop_path), crop)
        box = detection.box
        cv2.rectangle(
            frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 230, 255), 2
        )
        records.append(
            {
                "detection": detection.model_dump(),
                "ocr": [reading.model_dump() for reading in readings],
                "crop": str(crop_path.relative_to(ROOT)),
            }
        )
    cv2.imwrite(str(BUILD / "real-inference.png"), frame)
    result = {
        "scope": "selected held-out single image, not a new accuracy benchmark",
        "source_image": str(path.relative_to(ROOT)),
        "model_version": detector.model_version,
        "records": records,
        "temporal_confirmation": False,
        "attribution": "SAINT70239 / Bangkok-Bus-Dataset / CC BY 4.0",
    }
    (ROOT / "reports/demo-inference.json").write_text(json.dumps(result, indent=2))
    return result


def subtitles(
    narrations: list[str], audio_durations: list[float], segments: list[Path]
) -> list[float]:
    """Approximate cue timing within each actual chapter, explicitly marked for review."""

    def stamp(seconds: float) -> str:
        millis = round(seconds * 1000)
        hours, millis = divmod(millis, 3600000)
        minutes, millis = divmod(millis, 60000)
        seconds_whole, millis = divmod(millis, 1000)
        return f"{hours:02}:{minutes:02}:{seconds_whole:02},{millis:03}"

    cues = []
    offset = 0.0
    encoded_durations = []
    for narration, duration, segment in zip(narrations, audio_durations, segments, strict=True):
        words = narration.split()
        consumed = 0
        for index in range(0, len(words), 12):
            chunk = " ".join(words[index : index + 12])
            start = offset + duration * consumed / len(words)
            consumed += len(chunk.split())
            end = offset + duration * consumed / len(words)
            caption = "\n".join(textwrap.wrap(chunk, 52))
            cues.append(f"{len(cues) + 1}\n{stamp(start)} --> {stamp(end)}\n{caption}\n")
        capture = cv2.VideoCapture(str(segment))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        if not fps:
            raise RuntimeError("Capítulo não decodifica para gerar legendas")
        encoded_durations.append(frames / fps)
        offset += frames / fps
    (OUT / "pilot-narrated.srt").write_text("\n".join(cues))
    return encoded_durations


def main() -> None:
    import imageio_ffmpeg

    if shutil.which("say") is None:
        raise SystemExit(
            "Esta ferramenta de narração usa say no macOS; forneça áudio "
            "equivalente em outros sistemas"
        )
    BUILD.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((ROOT / "reports/evaluation/detector_metrics.json").read_text())
    evidence = inference_evidence()
    if evidence["model_version"].split(":")[-1] != metrics["model_sha256"]:
        raise RuntimeError("Pesos e métricas não pertencem ao mesmo experimento")
    chapters = []

    image, draw = slide("Informação para apoiar a autonomia", "PROTÓTIPO ACADÊMICO · FIAP 2026.2")
    lines(
        draw,
        "Identificação visual de linhas de ônibus\nDetector customizado, OCR "
        "e validação conservadora.\nEstado: PARTIALLY_COMPLETED",
        (65, 205),
        64,
        33,
    )
    lines(
        draw,
        "Caique Chagas Nitta · RM551943\nRodrigo Resende Viana · RM551057\n"
        "Gabriel Torres Luiz · RM98600\nOtavio Vitoriano Da Silva · RM552012\n"
        "Eduardo Fedeli · RM550132",
        (65, 475),
        85,
        23,
        MUTED,
    )
    chapters.append(
        (
            image,
            "O Sonar Vision é um protótipo acadêmico de apoio à identificação de linhas de ônibus. "
            "Este vídeo apresenta o treino piloto e exemplos reais de detector e "
            "OCR em imagens estrangeiras. "
            "A interface simulada será identificada. O sistema não confirma "
            "destino, sentido ou condições de embarque.",
        )
    )

    image, draw = slide(
        "Treinamento customizado comprovado", "GRÁFICO REAL · dados de Bangkok, CC BY 4.0"
    )
    paste(image, ROOT / "reports/training/results.png", (65, 180, 1470, 450))
    lines(
        draw,
        "YOLO11n · 12 épocas · Apple M1 / MPS · 84 treino / 18 validação / 18 teste",
        (65, 660),
        94,
        25,
    )
    metric_text = "    ".join(
        [
            f"mAP50: {metrics['map50']:.3f}",
            f"Recall: {metrics['recall']:.3f}",
            f"Precision: {metrics['precision']:.3f}",
        ]
    )
    lines(
        draw,
        metric_text,
        (65, 715),
        100,
        28,
        ACCENT,
    )
    chapters.append(
        (
            image,
            "O modelo YOLO onze nano foi realmente ajustado durante doze épocas, "
            "em Apple M um com MPS. "
            "Foram usadas cento e vinte famílias de imagens: oitenta e quatro "
            "para treino, dezoito para validação e dezoito para teste. "
            "O teste reservado obteve mape cinquenta de zero vírgula seis quatro "
            "quatro e recall de zero vírgula quatro sete um. "
            "O recall ainda é limitado e não comprova desempenho brasileiro.",
        )
    )

    image, draw = slide(
        "Detector e OCR reais em uma imagem de teste",
        "FRAME ÚNICO · sem confirmação temporal ou de linha brasileira",
    )
    paste(image, BUILD / "real-inference.png", (65, 180, 640, 620))
    record = evidence["records"][0]
    paste(image, ROOT / record["crop"], (790, 210, 650, 180))
    raw = " | ".join(reading["text"] for reading in record["ocr"]) or "Sem texto reconhecido"
    details_text = "\n".join(
        [
            f"Texto devolvido pelo OCR: {raw}",
            f"Score do detector: {record['detection']['confidence']:.3f}",
            "Scores separados, não calibrados.",
            "Resultado temporal: UNCERTAIN",
        ]
    )
    lines(
        draw,
        details_text,
        (790, 435),
        39,
        26,
    )
    chapters.append(
        (
            image,
            "Aqui a detecção e o OCR foram executados novamente para produzir esta evidência. "
            "A caixa amarela vem do modelo customizado, e o recorte mostrado é a "
            "região prevista, não uma caixa inventada. "
            f"O texto retornado pelo OCR nesta imagem foi {raw}. "
            "É uma imagem única de Bangkok. Nenhuma confirmação temporal foi "
            "fabricada pela repetição dessa fotografia.",
        )
    )

    image, draw = slide(
        "Interface acessível — SIMULAÇÃO",
        "SIMULAÇÃO EXPLÍCITA · o clipe de navegação está disponível separadamente",
    )
    paste(image, OUT / "simulation-desktop.png", (65, 180, 730, 625))
    lines(
        draw,
        "MATCH, NON_MATCH e UNCERTAIN\nHistórico, teclado e mensagens "
        "textuais.\nA voz depende do navegador.\nEsta cena usa leituras "
        "artificiais: não comprova OCR de 971D-10.",
        (850, 220),
        35,
        27,
    )
    chapters.append(
        (
            image,
            "A interface oferece histórico, mensagens textuais e anúncio por voz. "
            "Esta captura é explicitamente uma simulação, com leituras artificiais. "
            "Ela demonstra os estados de correspondência, outra linha e "
            "incerteza, mas não prova que o OCR leu nove sete um dê dez. "
            "A narração deste vídeo foi sintetizada localmente e não é um teste "
            "acústico da voz no navegador.",
        )
    )

    image, draw = slide(
        "Qualidade e próximos passos",
        "Sem autorização de embarque · validação brasileira ainda pendente",
    )
    lines(
        draw,
        "104 testes aprovados; lint, tipos e instalação verificados.\nOCR: "
        "8/9 recortes legíveis; outros 8/17 excluídos.\nVídeos reais "
        "processados sem código confirmado.\nFaltam encontros brasileiros "
        "anotados e revisão humana de acessibilidade.",
        (65, 200),
        85,
        31,
    )
    lines(
        draw,
        "Créditos: SAINT70239 / Bangkok-Bus-Dataset, CC BY 4.0.\nImagens de "
        "ônibus: recorte, caixas previstas e redimensionamento.\nCurvas de "
        "treino e interface: artefatos deste projeto.\nFontes e limitações "
        "completas: docs/DATA_SOURCES.md e docs/REPORT.md.",
        (65, 570),
        100,
        22,
        MUTED,
    )
    chapters.append(
        (
            image,
            "O checkpoint aprovou cento e quatro testes de software. "
            "O OCR acertou oito de nove recortes legíveis, enquanto oito outros "
            "recortes não tinham transcrição confiável. "
            "Nos vídeos reais explorados não houve confirmação de código. "
            "A próxima etapa é avaliar encontros brasileiros anotados e revisar "
            "áudio e tecnologias assistivas. "
            "As imagens do dataset Bangkok são de Saint sete zero dois três "
            "nove, com licença Creative Commons atribuição quatro ponto zero. "
            "A entrega continua parcialmente concluída.",
        )
    )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    segments = []
    durations = []
    for index, (image, narration) in enumerate(chapters):
        base = BUILD / f"chapter-{index + 1}"
        image.save(base.with_suffix(".png"))
        base.with_suffix(".txt").write_text(narration)
        run(
            [
                "say",
                "-v",
                "Luciana",
                "-r",
                "160",
                "-f",
                str(base.with_suffix(".txt")),
                "-o",
                str(base.with_suffix(".aiff")),
            ]
        )
        audio_info = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(base.with_suffix(".aiff"))],
            capture_output=True,
            text=True,
            check=False,
        )
        import re

        match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", audio_info.stderr)
        if not match:
            raise RuntimeError("Duração da narração indisponível")
        duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
        durations.append(duration)
        output = base.with_suffix(".mp4")
        run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-framerate",
                "20",
                "-i",
                str(base.with_suffix(".png")),
                "-i",
                str(base.with_suffix(".aiff")),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-tune",
                "stillimage",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
        segments.append(output)

    concat = BUILD / "concat.txt"
    concat.write_text("".join(f"file '{path.name}'\n" for path in segments))
    destination = OUT / "pilot-narrated.mp4"
    run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )

    transcript = "# Narração do vídeo piloto\n\n"
    for i, (_, narration) in enumerate(chapters):
        transcript += f"## Capítulo {i + 1}\n\n{narration}\n\n"
    (ROOT / "docs/VIDEO_NARRATION.md").write_text(transcript.rstrip() + "\n")
    encoded_durations = subtitles([text for _, text in chapters], durations, segments)
    audio = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(destination),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64)
    rms = math.sqrt(float(np.mean(samples * samples))) if samples.size else 0
    if rms == 0:
        raise RuntimeError("Áudio vazio ou silencioso")
    capture = cv2.VideoCapture(str(destination))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    ok, _ = capture.read()
    capture.release()
    if not ok or not fps:
        raise RuntimeError("Vídeo final não decodifica")
    result = {
        "status": "PASSED",
        "scope": "narrated pilot evidence, not Brazilian temporal validation",
        "video": "artifacts/demo/pilot-narrated.mp4",
        "width": SIZE[0],
        "height": SIZE[1],
        "duration_seconds": count / fps,
        "fps": fps,
        "audio_rms_pcm16": rms,
        "audio_voice": "macOS Luciana pt-BR",
        "chapter_audio_seconds": durations,
        "audio_acoustically_reviewed": False,
        "simulation_explicitly_labeled": True,
        "bytes": destination.stat().st_size,
        "subtitle": "artifacts/demo/pilot-narrated.srt",
        "subtitle_timing": "approximate word-proportional chapter timing; human review pending",
        "encoded_chapter_seconds": encoded_durations,
    }
    (ROOT / "reports/demo-video.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
