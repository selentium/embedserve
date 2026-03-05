from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.engine.embedder import Embedder, RuntimeInitializationError, build_transformer_embedder
from app.readiness import ReadyReason
from app.settings import Settings


@dataclass(frozen=True)
class RuntimeState:
    ready: bool
    mode: Literal["model"]
    model_id: str
    revision: str
    device: str
    dtype: str
    reason: ReadyReason | None
    detail: str | None
    embedder: Embedder | None


RuntimeInitializer = Callable[[Settings], RuntimeState]


def initialize_runtime(settings: Settings) -> RuntimeState:
    embedder = build_transformer_embedder(settings)

    try:
        embedder.embed(["warmup"])
    except RuntimeInitializationError as exc:
        if exc.reason == "tokenizer_incompatible":
            raise
        raise RuntimeInitializationError("warmup_failed", exc.detail) from exc
    except Exception as exc:
        raise RuntimeInitializationError("warmup_failed", str(exc)) from exc

    return RuntimeState(
        ready=True,
        mode="model",
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
        device=embedder.device,
        dtype=embedder.dtype,
        reason=None,
        detail=None,
        embedder=embedder,
    )


def build_unready_runtime(
    settings: Settings,
    *,
    reason: ReadyReason,
    detail: str,
) -> RuntimeState:
    return RuntimeState(
        ready=False,
        mode="model",
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
        device=settings.DEVICE,
        dtype=settings.DTYPE,
        reason=reason,
        detail=detail,
        embedder=None,
    )
