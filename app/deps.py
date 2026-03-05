from __future__ import annotations

from typing import cast

from fastapi import Request

from app.metrics import AppMetrics
from app.runtime import RuntimeState
from app.settings import Settings


async def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


async def get_metrics(request: Request) -> AppMetrics:
    return cast(AppMetrics, request.app.state.metrics)


async def get_runtime(request: Request) -> RuntimeState:
    return cast(RuntimeState, request.app.state.runtime)
