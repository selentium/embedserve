from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.routing import BaseRoute
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.deps import get_embedder, get_metrics, get_settings
from app.engine.embedder import Embedder, StubEmbedder
from app.logging import configure_logging
from app.metrics import (
    CONTENT_TYPE_LATEST,
    AppMetrics,
    create_metrics,
    mark_ready,
    observe_http_request,
    render_metrics,
    touch_http_metrics,
)
from app.schemas import EmbedRequest, EmbedResponse, HealthResponse, ReadyResponse
from app.settings import Settings


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    if isinstance(route, BaseRoute):
        route_path = getattr(route, "path", None)
        if isinstance(route_path, str):
            return route_path
    return "unmatched"


def _request_too_large_error(
    limit: int, actual: int, payload: EmbedRequest
) -> RequestValidationError:
    return RequestValidationError(
        [
            {
                "type": "too_long",
                "loc": ("body", "inputs"),
                "msg": f"List should have at most {limit} items after validation, not {actual}",
                "input": payload.inputs,
                "ctx": {"field_type": "List", "max_length": limit, "actual_length": actual},
            }
        ]
    )


class RequestContextMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        app_logger: logging.Logger,
        access_logger: logging.Logger,
    ) -> None:
        self.app = app
        self.app_logger = app_logger
        self.access_logger = access_logger

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        request = Request(scope)
        started_at = perf_counter()
        status_code = 500
        response_started = False
        request_cancelled = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started, status_code

            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}

            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except asyncio.CancelledError:
            request_cancelled = True
            raise
        except Exception:
            if response_started:
                raise

            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error"},
            )
            status_code = response.status_code
            self.app_logger.exception(
                "Unhandled server error",
                extra={
                    "event": "unhandled_exception",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                },
            )
            await response(scope, receive, send_with_request_id)
        finally:
            if not request_cancelled:
                duration_seconds = perf_counter() - started_at
                duration_ms = round(duration_seconds * 1000, 3)
                route = _route_template(request)
                metrics = cast(AppMetrics, request.app.state.metrics)
                observe_http_request(
                    metrics,
                    method=request.method,
                    route=route,
                    status_code=status_code,
                    duration_seconds=duration_seconds,
                )

                self.access_logger.info(
                    "Request completed",
                    extra={
                        "event": "http_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "duration_ms": duration_ms,
                    },
                )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    configure_logging(settings.LOG_LEVEL)

    metrics = create_metrics()
    mark_ready(metrics, mode="stub")

    app.state.settings = settings
    app.state.metrics = metrics
    app.state.embedder = StubEmbedder(
        model=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
    )

    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app_logger = logging.getLogger("embedserve.app")
    access_logger = logging.getLogger("embedserve.access")
    app.add_middleware(
        RequestContextMiddleware,
        app_logger=app_logger,
        access_logger=access_logger,
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse)
    async def readyz() -> ReadyResponse:
        return ReadyResponse(status="ready", mode="stub")

    @app.get("/metrics")
    async def metrics_endpoint(
        metrics: Annotated[AppMetrics, Depends(get_metrics)],
    ) -> Response:
        touch_http_metrics(metrics, method="GET", route="/metrics", status_code=200)
        return Response(content=render_metrics(metrics), media_type=CONTENT_TYPE_LATEST)

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(
        payload: EmbedRequest,
        settings: Annotated[Settings, Depends(get_settings)],
        embedder: Annotated[Embedder, Depends(get_embedder)],
    ) -> EmbedResponse:
        actual_inputs = len(payload.inputs)
        if actual_inputs > settings.MAX_INPUTS_PER_REQUEST:
            raise _request_too_large_error(
                limit=settings.MAX_INPUTS_PER_REQUEST,
                actual=actual_inputs,
                payload=payload,
            )
        return embedder.embed(payload.inputs)

    return app


app = create_app()
