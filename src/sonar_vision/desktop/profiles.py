"""Curated cached footage only; no invented Bangkok substitute."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    id: str
    city: str
    model: str
    weights: str
    video: str | None
    imgsz: int
    source: str
    author: str
    license: str
    limitation: str
    bus_weights: str | None = None

    def record(self) -> dict[str, object]:
        return asdict(self)


def profiles() -> dict[str, Profile]:
    br = "sonar_br/artifacts/train-v1/run/weights/best.pt"
    result = {
        "sao-paulo": Profile(
            "sao-paulo",
            "SÃO PAULO",
            "SONAR BR V1",
            br,
            "artifacts/demo/final-desktop/media/sao-paulo-real.mp4",
            640,
            "https://commons.wikimedia.org/wiki/File:Apartes_Grandes_Reportagens_-_SP_24h_-_Bloco_1-3.webm",
            "Câmara Municipal de São Paulo",
            "CC BY 3.0",
            (
                "Ônibus ao fundo de entrevista noturna; letreiro pequeno. "
                "Sem reconhecimento positivo."
            ),
        ),
        "curitiba": Profile(
            "curitiba",
            "CURITIBA",
            "SONAR BR V1 — diagnóstico 1280",
            br,
            "artifacts/demo/practical-examples/rotas-approach-real.mp4",
            1280,
            "https://commons.wikimedia.org/wiki/File:Rotas_do_Brasil_-_Curitiba.webm",
            "TV BrasilGov",
            "CC BY 3.0",
            (
                "Painel parcial; perfil experimental. "
                "Não é avaliação independente ou confirmação temporal."
            ),
        ),
        "bangkok": Profile(
            "bangkok",
            "BANGKOK",
            "YOLO Bangkok original",
            "models/best.pt",
            None,
            640,
            "https://commons.wikimedia.org/wiki/File:Walking_in_BANGKOK_-_Thailand_-_Modern_City_Center_Tour_(2019)_-_4K_60fps_(UHD).webm",
            "POPtravel (candidato não transferido)",
            "CC BY 3.0 (candidato)",
            (
                "SEM VÍDEO AUTORIZADO EM CACHE. Fonte anterior bloqueada HTTP429; "
                "nenhuma inferência Bangkok em vídeo."
            ),
        ),
    }
    # Delayed import avoids render/profile initialization cycles. An absent
    # validated private configuration preserves every historical profile.
    from sonar_vision.desktop.v3 import active_configuration

    active = active_configuration()
    if active:
        for key, item in active["profiles"].items():
            result[key] = Profile(
                key,
                item["city"],
                item.get("model", "SONAR VIDEO V3 — privado"),
                item.get("weights", active["weights"]),
                item["video"],
                960,
                item["source"],
                "Não verificado",
                "Direitos não verificados; somente experimento local privado",
                "Não publicar. Amostra pequena; sem confirmação temporal de linha.",
            )
        moving = Path("artifacts/demo/sonar-video-v3/visual-yolo-style/media/santos-moving.mp4")
        if moving.is_file():
            result["santos-yolo"] = Profile(
                "santos-yolo",
                "SANTOS — VÍDEO REAL",
                "V3 ônibus + BR V1 letreiro | composição experimental",
                br,
                str(moving),
                960,
                "Vídeo pessoal; declaração do usuário",
                "Não identificado",
                "ML autorizado pelo usuário",
                "Composição de dois detectores; não é sucesso do V3 sozinho.",
                bus_weights=active["weights"],
            )
    return result


def validate_local(root: Path, profile: Profile) -> None:
    if profile.bus_weights and not (root / profile.bus_weights).is_file():
        raise ValueError(f"Pesos de ônibus ausentes: {profile.bus_weights}")
    if not (root / profile.weights).is_file():
        raise ValueError(f"Pesos ausentes: {profile.weights}")
    if profile.video and not (root / profile.video).is_file():
        raise ValueError(f"Vídeo ausente: {profile.video}")
