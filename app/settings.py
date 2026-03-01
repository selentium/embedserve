from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    MODEL_ID: str = "stub-model"
    MODEL_REVISION: str = "milestone1-stub"
    LOG_LEVEL: str = "INFO"
    MAX_INPUTS_PER_REQUEST: int = Field(default=64, ge=1)

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in _VALID_LOG_LEVELS:
            msg = f"Unsupported LOG_LEVEL: {value}"
            raise ValueError(msg)
        return normalized
