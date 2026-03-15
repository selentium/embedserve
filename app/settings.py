from __future__ import annotations

import re

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
_VALID_DEVICE_PATTERN = re.compile(r"^(cpu|cuda(?::\d+)?)$")
_VALID_DTYPE_VALUES = {"float32", "float16", "bfloat16"}
_VALID_OUTPUT_DTYPE_VALUES = {"float32", "float16"}
_VALID_REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_REVISION: str = "826711e54e001c83835913827a843d8dd0a1def9"
    LOG_LEVEL: str = "INFO"
    MAX_INPUTS_PER_REQUEST: int = Field(default=64, ge=1)
    MAX_BATCH_SIZE: int = Field(default=128, ge=1)
    MAX_BATCH_TOKENS: int = Field(default=8192, ge=1)
    BATCH_TIMEOUT_MS: int = Field(default=2, ge=1)
    MAX_BATCH_QUEUE_SIZE: int = Field(default=1024, ge=1)
    BATCH_REQUEST_TIMEOUT_MS: int = Field(default=5000, ge=1)
    DEVICE: str = "cpu"
    DTYPE: str = "float32"
    MAX_LENGTH: int = Field(default=512, ge=1)
    TRUNCATE: bool = True
    NORMALIZE_EMBEDDINGS: bool = True
    OUTPUT_DTYPE: str = "float32"

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in _VALID_LOG_LEVELS:
            msg = f"Unsupported LOG_LEVEL: {value}"
            raise ValueError(msg)
        return normalized

    @field_validator("DEVICE")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.lower()
        if _VALID_DEVICE_PATTERN.fullmatch(normalized) is None:
            msg = f"Unsupported DEVICE: {value}"
            raise ValueError(msg)
        return normalized

    @field_validator("DTYPE")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in _VALID_DTYPE_VALUES:
            msg = f"Unsupported DTYPE: {value}"
            raise ValueError(msg)
        return normalized

    @field_validator("OUTPUT_DTYPE")
    @classmethod
    def validate_output_dtype(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in _VALID_OUTPUT_DTYPE_VALUES:
            msg = f"Unsupported OUTPUT_DTYPE: {value}"
            raise ValueError(msg)
        return normalized

    @field_validator("MODEL_REVISION")
    @classmethod
    def validate_model_revision(cls, value: str) -> str:
        if _VALID_REVISION_PATTERN.fullmatch(value) is None:
            msg = "MODEL_REVISION must be a 40-character hexadecimal commit hash"
            raise ValueError(msg)
        return value
