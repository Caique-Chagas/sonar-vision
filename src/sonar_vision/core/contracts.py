"""Versioned, conservative integration contracts. Scores retain their own meaning."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class MatchStatus(StrEnum):
    MATCH = "MATCH"
    NON_MATCH = "NON_MATCH"
    UNCERTAIN = "UNCERTAIN"


class Box(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    x1: float = Field(ge=0, allow_inf_nan=False)
    y1: float = Field(ge=0, allow_inf_nan=False)
    x2: float = Field(gt=0, allow_inf_nan=False)
    y2: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box precisa ter área positiva")
        return self


class Detection(BaseModel):
    box: Box
    confidence: Score
    label: Literal["destination_sign", "bus"]


class OCRReading(BaseModel):
    text: str = Field(max_length=512)
    confidence: Score


class VisionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stop_id: str = Field(min_length=1, max_length=100)
    waiting_session_id: str = Field(min_length=1, max_length=100)
    track_id: int = Field(ge=1)
    route_requested: str = Field(min_length=1, max_length=32)
    route_detected: str | None = None
    destination_detected: None = None
    detection_confidence: Score | None = None
    ocr_confidence: Score | None = None
    match_status: MatchStatus = MatchStatus.UNCERTAIN
    frames_considered: int = Field(ge=0)
    supporting_frames: int = Field(ge=0)
    source: Literal["video_stream", "local_video", "simulation"]
    model_version: str
    status: Literal["CONFIRMED", "UNCERTAIN"]
    raw_texts: list[str] = Field(default_factory=list, max_length=32)
    reason: str
    simulated: bool = False

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp precisa incluir fuso horário")
        if self.supporting_frames > self.frames_considered:
            raise ValueError("supporting_frames excede frames_considered")
        if self.status == "CONFIRMED":
            if not self.route_detected or self.supporting_frames < 3:
                raise ValueError("Confirmação exige linha e pelo menos três frames distintos")
            if self.detection_confidence is None or self.ocr_confidence is None:
                raise ValueError("Confirmação exige scores reais separados")
        elif self.match_status != MatchStatus.UNCERTAIN or self.route_detected is not None:
            raise ValueError("Leitura incerta não pode confirmar correspondência ou linha")
        if self.match_status == MatchStatus.MATCH and self.route_detected != self.route_requested:
            raise ValueError("MATCH exige igualdade entre códigos completos")
        if (
            self.match_status == MatchStatus.NON_MATCH
            and self.route_detected == self.route_requested
        ):
            raise ValueError("NON_MATCH exige códigos diferentes")
        if self.simulated != (self.source == "simulation"):
            raise ValueError("Eventos simulados precisam ter source=simulation")
        return self


class LegacyEvent(BaseModel):
    espera_id: str
    linha_lida: str | None
    confianca: float | None
    horario_da_passagem: datetime
    tempo_parado: None = None
    confianca_tipo: Literal["ocr_mean_uncalibrated"] = "ocr_mean_uncalibrated"
    simulado: bool


def to_legacy(event: VisionEvent) -> LegacyEvent:
    return LegacyEvent(
        espera_id=event.waiting_session_id,
        linha_lida=event.route_detected,
        confianca=event.ocr_confidence,
        horario_da_passagem=event.timestamp,
        simulado=event.simulated,
    )
