from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SONAR_", env_file=".env", extra="ignore")
    model_path: Path = Path("models/best.pt")
    ocr_backend: Literal["paddle"] = "paddle"
    video_hosts: str = ""
    max_sessions: int = Field(default=2, ge=1, le=4)
    session_retention: int = Field(default=20, ge=2, le=100)
    min_votes: int = Field(default=3, ge=3, le=12)
    window_frames: int = Field(default=8, ge=3, le=24)
    min_ratio: float = Field(default=0.75, ge=0.5, le=1)
    min_ocr: float = Field(default=0.7, ge=0, le=1)
    min_detection: float = Field(default=0.35, ge=0, le=1)
    evidence_seconds: float = Field(default=3, gt=0, le=10)
    min_span_seconds: float = Field(default=0.3, ge=0.1, le=5)

    @model_validator(mode="after")
    def feasible_window(self) -> Self:
        if self.min_votes > self.window_frames or self.min_span_seconds > self.evidence_seconds:
            raise ValueError("Janela temporal não comporta os limiares de confirmação")
        return self

    @property
    def allowed_hosts(self) -> set[str]:
        return {host.strip().lower() for host in self.video_hosts.split(",") if host.strip()}
