from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    mode: Literal["stub"]
