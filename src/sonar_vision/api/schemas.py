import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sonar_vision.validation.normalizer import FULL_CODE, clean_code


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route_requested: str = Field(min_length=1, max_length=32)
    source_url: str | None = Field(default=None, max_length=2048)
    mode: Literal["real", "simulation"] = "real"
    stop_id: str = Field(default="parada-piloto", min_length=1, max_length=100)
    waiting_session_id: str = Field(default="espera-demo", min_length=1, max_length=100)
    fps: float = Field(default=5, ge=0.5, le=15)
    max_frames: int = Field(default=150, ge=1, le=1200)
    max_seconds: float = Field(default=30, ge=1, le=180)

    @field_validator("route_requested")
    @classmethod
    def route_format(cls, value: str) -> str:
        value = clean_code(value)
        if not FULL_CODE.fullmatch(value) and not re.fullmatch(r"[A-Z0-9]{2,6}-[0-9]{1,3}", value):
            raise ValueError("Informe código completo, por exemplo 971D-10")
        return value

    @model_validator(mode="after")
    def source_required(self) -> "SessionRequest":
        if self.mode == "real" and not self.source_url:
            raise ValueError("Inferência real exige URL de vídeo")
        return self
