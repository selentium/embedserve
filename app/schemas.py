from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.readiness import ReadyReason


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: list[str] = Field(min_length=1)

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, value: list[str]) -> list[str]:
        if any(item.strip() == "" for item in value):
            msg = "Input strings must not be blank or whitespace-only"
            raise ValueError(msg)
        return value


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class UsageInfo(BaseModel):
    tokens: int


class EmbedResponse(BaseModel):
    data: list[EmbeddingItem]
    model: str
    revision: str
    dim: int
    usage: UsageInfo


class HealthResponse(BaseModel):
    status: Literal["ok"]


class TokenizationSettingsInfo(BaseModel):
    max_length: int
    truncate: bool


class BatchingSettingsInfo(BaseModel):
    max_batch_size: int
    max_batch_tokens: int
    batch_timeout_ms: int
    max_batch_queue_size: int
    batch_request_timeout_ms: int


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    mode: Literal["model"]
    model: str
    revision: str
    device: str
    dtype: str
    tokenization: TokenizationSettingsInfo
    batching: BatchingSettingsInfo


class NotReadyResponse(BaseModel):
    status: Literal["not_ready"]
    mode: Literal["model"]
    model: str
    revision: str
    device: str
    dtype: str
    reason: ReadyReason
    detail: str
    tokenization: TokenizationSettingsInfo
    batching: BatchingSettingsInfo
