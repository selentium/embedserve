from __future__ import annotations

from hashlib import sha256
from typing import Protocol

from app.schemas import EmbeddingItem, EmbedResponse, UsageInfo

STUB_EMBEDDING_DIM = 8
_UINT32_MAX = 0xFFFFFFFF


class Embedder(Protocol):
    def embed(self, inputs: list[str]) -> EmbedResponse: ...


class StubEmbedder:
    def __init__(self, *, model: str, revision: str, dim: int = STUB_EMBEDDING_DIM) -> None:
        self._model = model
        self._revision = revision
        self._dim = dim

    def embed(self, inputs: list[str]) -> EmbedResponse:
        data = [
            EmbeddingItem(index=index, embedding=self._embed_text(text))
            for index, text in enumerate(inputs)
        ]
        return EmbedResponse(
            data=data,
            model=self._model,
            revision=self._revision,
            dim=self._dim,
            usage=UsageInfo(tokens=0),
        )

    def _embed_text(self, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()
        values: list[float] = []

        for offset in range(0, self._dim * 4, 4):
            chunk = digest[offset : offset + 4]
            raw_value = int.from_bytes(chunk, byteorder="big", signed=False)
            scaled = (raw_value / _UINT32_MAX) * 2.0 - 1.0
            values.append(round(scaled, 6))

        return values
