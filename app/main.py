from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.routing import BaseRoute
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.batching import (
    BatcherQueueFullError,
    BatcherShuttingDownError,
    BatchInferenceError,
    DynamicBatcher,
)
from app.deps import get_batcher, get_metrics, get_runtime, get_settings
from app.determinism import DeterminismPolicyState, apply_determinism_policy
from app.engine.embedder import RuntimeInitializationError
from app.logging import configure_logging
from app.metrics import (
    CONTENT_TYPE_LATEST,
    AppMetrics,
    create_metrics,
    observe_http_request,
    render_metrics,
    set_ready_state,
    touch_http_metrics,
)
from app.runtime import RuntimeInitializer, RuntimeState, build_unready_runtime, initialize_runtime
from app.schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    NotReadyResponse,
    ReadyResponse,
)
from app.settings import Settings

_DETAIL_BATCH_QUEUE_FULL = "Batch queue is full"
_DETAIL_BATCH_REQUEST_TIMED_OUT = "Embedding request timed out"
_DETAIL_BATCH_SHUTDOWN = "Service is shutting down"
_DETAIL_BATCH_INFERENCE_FAILED = "Embedding inference failed"
_SUBMISSION_POLL_INTERVAL_SECONDS = 0.001


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
                metrics = request.app.state.metrics
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


def _runtime_log_fields(
    settings: Settings,
    runtime: RuntimeState | None = None,
    determinism: DeterminismPolicyState | None = None,
) -> dict[str, object]:
    if determinism is None:
        determinism = DeterminismPolicyState(
            mode="numerical_stability",
            seed=0,
            deterministic_algorithms="unsupported",
        )

    return {
        "model": settings.MODEL_ID,
        "revision": settings.MODEL_REVISION,
        "device": settings.DEVICE if runtime is None else runtime.device,
        "dtype": settings.DTYPE if runtime is None else runtime.dtype,
        "max_length": settings.MAX_LENGTH,
        "truncate": settings.TRUNCATE,
        "normalize_embeddings": settings.NORMALIZE_EMBEDDINGS,
        "output_dtype": settings.OUTPUT_DTYPE,
        "determinism_policy": determinism.mode,
        "determinism_seed": determinism.seed,
        "deterministic_algorithms": determinism.deterministic_algorithms,
    }


async def _await_submission_result(
    future: asyncio.Future[EmbedResponse],
    *,
    timeout_seconds: float,
) -> EmbedResponse:
    deadline = perf_counter() + timeout_seconds
    while True:
        if future.done():
            return future.result()
        if perf_counter() >= deadline:
            raise TimeoutError
        await asyncio.sleep(_SUBMISSION_POLL_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    configure_logging(settings.LOG_LEVEL)

    app_logger = logging.getLogger("embedserve.app")
    metrics = create_metrics()
    set_ready_state(metrics, mode="model", ready=False)

    app.state.settings = settings
    app.state.metrics = metrics

    runtime_initializer = app.state.runtime_initializer
    determinism: DeterminismPolicyState | None = None
    batcher: DynamicBatcher | None = None

    try:
        determinism = apply_determinism_policy()
        runtime = runtime_initializer(settings)
    except ImportError:
        raise
    except Exception as exc:
        if isinstance(exc, RuntimeInitializationError):
            failure = exc
        else:
            failure = RuntimeInitializationError("initialization_failed", str(exc))

        runtime = build_unready_runtime(
            settings,
            reason=failure.reason,
            detail=failure.detail,
        )
        app_logger.exception(
            "Runtime initialization failed",
            extra={
                "event": "runtime_initialization_failed",
                "reason": failure.reason,
                "detail": failure.detail,
                **_runtime_log_fields(settings, determinism=determinism),
            },
        )
    else:
        set_ready_state(metrics, mode="model", ready=True)
        app_logger.info(
            "Runtime initialization succeeded",
            extra={
                "event": "runtime_initialization_succeeded",
                **_runtime_log_fields(settings, runtime, determinism),
            },
        )

    app.state.runtime = runtime
    if runtime.ready and runtime.embedder is not None:
        batcher = DynamicBatcher(
            embedder=runtime.embedder,
            metrics=metrics,
            max_batch_size=settings.MAX_BATCH_SIZE,
            max_batch_tokens=settings.MAX_BATCH_TOKENS,
            batch_timeout_ms=settings.BATCH_TIMEOUT_MS,
            max_batch_queue_size=settings.MAX_BATCH_QUEUE_SIZE,
        )
        await batcher.start()
    app.state.batcher = batcher

    try:
        yield
    finally:
        if batcher is not None:
            await batcher.shutdown()


def create_app(
    *,
    runtime_initializer: RuntimeInitializer = initialize_runtime,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.runtime_initializer = runtime_initializer

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

    @app.get(
        "/readyz",
        response_model=ReadyResponse,
        responses={503: {"model": NotReadyResponse}},
    )
    async def readyz(
        runtime: Annotated[RuntimeState, Depends(get_runtime)],
    ) -> ReadyResponse | JSONResponse:
        if runtime.ready:
            return ReadyResponse(
                status="ready",
                mode=runtime.mode,
                model=runtime.model_id,
                revision=runtime.revision,
                device=runtime.device,
                dtype=runtime.dtype,
            )

        assert runtime.reason is not None
        return JSONResponse(
            status_code=503,
            content=NotReadyResponse(
                status="not_ready",
                mode=runtime.mode,
                model=runtime.model_id,
                revision=runtime.revision,
                device=runtime.device,
                dtype=runtime.dtype,
                reason=runtime.reason,
                detail=runtime.detail or "Initialization failed",
            ).model_dump(),
        )

    @app.get("/metrics")
    async def metrics_endpoint(
        metrics: Annotated[AppMetrics, Depends(get_metrics)],
    ) -> Response:
        touch_http_metrics(metrics, method="GET", route="/metrics", status_code=200)
        return Response(content=render_metrics(metrics), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/embed",
        response_model=EmbedResponse,
        responses={
            503: {
                "description": "Service unavailable",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["detail"],
                            "properties": {"detail": {"type": "string"}},
                        },
                        "examples": {
                            "unready": {
                                "summary": "Model runtime is not ready",
                                "value": {"detail": "Model is not ready: initialization_failed"},
                            },
                            "queue_full": {
                                "summary": "Batch queue overload",
                                "value": {"detail": _DETAIL_BATCH_QUEUE_FULL},
                            },
                            "request_timeout": {
                                "summary": "Batch wait timeout",
                                "value": {"detail": _DETAIL_BATCH_REQUEST_TIMED_OUT},
                            },
                            "shutdown": {
                                "summary": "Service shutdown",
                                "value": {"detail": _DETAIL_BATCH_SHUTDOWN},
                            },
                        },
                    }
                },
            },
            500: {
                "description": "Embedding inference failed",
                "content": {
                    "application/json": {
                        "example": {"detail": _DETAIL_BATCH_INFERENCE_FAILED},
                    }
                },
            },
        },
    )
    async def embed(
        payload: EmbedRequest,
        settings: Annotated[Settings, Depends(get_settings)],
        runtime: Annotated[RuntimeState, Depends(get_runtime)],
        batcher: Annotated[DynamicBatcher | None, Depends(get_batcher)],
        metrics: Annotated[AppMetrics, Depends(get_metrics)],
    ) -> EmbedResponse:
        actual_inputs = len(payload.inputs)
        if actual_inputs > settings.MAX_INPUTS_PER_REQUEST:
            raise _request_too_large_error(
                limit=settings.MAX_INPUTS_PER_REQUEST,
                actual=actual_inputs,
                payload=payload,
            )

        if not runtime.ready or runtime.embedder is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model is not ready: {runtime.reason}",
            )

        if batcher is None:
            metrics.batch_shutdown_rejections_total.inc()
            raise HTTPException(status_code=503, detail=_DETAIL_BATCH_SHUTDOWN)

        try:
            token_counts = await run_in_threadpool(runtime.embedder.preflight, payload.inputs)
        except asyncio.CancelledError:
            raise
        except RequestValidationError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=_DETAIL_BATCH_INFERENCE_FAILED) from exc

        try:
            submission = batcher.submit(payload.inputs, token_counts)
        except BatcherQueueFullError as exc:
            metrics.batch_overload_rejections_total.inc()
            raise HTTPException(status_code=503, detail=_DETAIL_BATCH_QUEUE_FULL) from exc
        except BatcherShuttingDownError as exc:
            metrics.batch_shutdown_rejections_total.inc()
            raise HTTPException(status_code=503, detail=_DETAIL_BATCH_SHUTDOWN) from exc

        timeout_seconds = settings.BATCH_REQUEST_TIMEOUT_MS / 1000.0
        try:
            return await _await_submission_result(
                submission.future, timeout_seconds=timeout_seconds
            )
        except TimeoutError as exc:
            batcher.cancel(submission.job_id)
            metrics.batch_request_timeouts_total.inc()
            raise HTTPException(status_code=503, detail=_DETAIL_BATCH_REQUEST_TIMED_OUT) from exc
        except asyncio.CancelledError:
            batcher.cancel(submission.job_id)
            metrics.batch_request_cancellations_total.inc()
            raise
        except BatcherShuttingDownError as exc:
            metrics.batch_shutdown_rejections_total.inc()
            raise HTTPException(status_code=503, detail=_DETAIL_BATCH_SHUTDOWN) from exc
        except BatchInferenceError as exc:
            raise HTTPException(status_code=500, detail=_DETAIL_BATCH_INFERENCE_FAILED) from exc

    return app


app = create_app()
