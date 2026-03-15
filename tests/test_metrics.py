from __future__ import annotations

import asyncio
import contextlib

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.types import Message

from app.engine.embedder import RuntimeInitializationError
from app.main import create_app
from app.metrics import AppMetrics
from app.runtime import RuntimeState
from app.settings import Settings
from tests.fakes import make_fake_embedder, make_ready_runtime


def _ready_initializer(settings: Settings) -> RuntimeState:
    embedder, _, _ = make_fake_embedder(
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
    )
    return make_ready_runtime(
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
        embedder=embedder,
    )


def _http_request_count(
    metrics: AppMetrics,
    *,
    method: str,
    route: str,
    status_code: str,
) -> float | None:
    for metric in metrics.http_requests_total.collect():
        for sample in metric.samples:
            if sample.name != "embedserve_http_requests_total":
                continue
            if sample.labels == {
                "method": method,
                "route": route,
                "status_code": status_code,
            }:
                return sample.value
    return None


def _app_ready_value(metrics: AppMetrics, *, mode: str) -> float | None:
    for metric in metrics.app_ready.collect():
        for sample in metric.samples:
            if sample.name != "embedserve_app_ready":
                continue
            if sample.labels == {"mode": mode}:
                return sample.value
    return None


def test_metrics_exposes_ready_runtime_state() -> None:
    with TestClient(create_app(runtime_initializer=_ready_initializer)) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert response.headers["X-Request-ID"]

    body = response.text
    assert "embedserve_http_requests_total" in body
    assert "embedserve_http_request_duration_seconds" in body
    assert 'embedserve_app_ready{mode="model"} 1.0' in body
    assert "embedserve_batch_queue_depth" in body
    assert "embedserve_batch_queue_wait_seconds" in body
    assert "embedserve_batch_size" in body
    assert "embedserve_batch_token_count" in body
    assert 'embedserve_batch_flush_total{reason="max_batch_size"} 0.0' in body
    assert 'embedserve_batch_flush_total{reason="max_batch_tokens"} 0.0' in body
    assert 'embedserve_batch_flush_total{reason="timeout"} 0.0' in body
    assert 'embedserve_batch_flush_total{reason="shutdown"} 0.0' in body
    assert "embedserve_batch_overload_rejections_total" in body
    assert "embedserve_batch_request_timeouts_total" in body
    assert "embedserve_batch_request_cancellations_total" in body
    assert "embedserve_batch_shutdown_rejections_total" in body
    assert "embedserve_batch_inference_failures_total" in body
    assert "python_gc_objects_collected_total" in body or "process_virtual_memory_bytes" in body


def test_metrics_exposes_unready_runtime_state() -> None:
    def failing_initializer(settings: Settings) -> RuntimeState:
        raise RuntimeInitializationError("model_load_failed", "weights missing")

    with TestClient(create_app(runtime_initializer=failing_initializer)) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert 'embedserve_app_ready{mode="model"} 0.0' in response.text


def test_metrics_supports_in_process_asgi_transport() -> None:
    async def run_request() -> tuple[int, str]:
        app = create_app(runtime_initializer=_ready_initializer)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client,
        ):
            response = await asyncio.wait_for(client.get("/metrics"), timeout=1)

        return response.status_code, response.headers["X-Request-ID"]

    status_code, request_id = asyncio.run(run_request())

    assert status_code == 200
    assert request_id


def test_cancelled_requests_do_not_record_http_500_metrics() -> None:
    async def run_request() -> float | None:
        app = create_app(runtime_initializer=_ready_initializer)

        @app.get("/cancel-test")
        async def cancel_test() -> dict[str, bool]:
            await asyncio.sleep(10)
            return {"ok": True}

        async with app.router.lifespan_context(app):
            metrics = app.state.metrics
            request_started = False
            sent_messages: list[Message] = []

            async def receive() -> Message:
                nonlocal request_started
                if not request_started:
                    request_started = True
                    return {"type": "http.request", "body": b"", "more_body": False}

                await asyncio.sleep(10)
                return {"type": "http.disconnect"}

            async def send(message: Message) -> None:
                sent_messages.append(message)

            request_task = asyncio.create_task(
                app(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0"},
                        "http_version": "1.1",
                        "method": "GET",
                        "scheme": "http",
                        "path": "/cancel-test",
                        "raw_path": b"/cancel-test",
                        "query_string": b"",
                        "headers": [],
                        "client": ("127.0.0.1", 123),
                        "server": ("testserver", 80),
                    },
                    receive,
                    send,
                )
            )
            await asyncio.sleep(0.05)
            request_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request_task

            assert sent_messages == []
            assert _app_ready_value(metrics, mode="model") == 1

            return _http_request_count(
                metrics,
                method="GET",
                route="/cancel-test",
                status_code="500",
            )

    http_500_count = asyncio.run(run_request())

    assert http_500_count is None


def test_unmatched_routes_use_a_stable_metrics_label() -> None:
    async def run_requests() -> tuple[float | None, float | None, float | None]:
        app = create_app(runtime_initializer=_ready_initializer)
        async with app.router.lifespan_context(app):
            metrics = app.state.metrics
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                first_response = await client.get("/does-not-exist/12345")
                second_response = await client.get("/does-not-exist/67890")

            assert first_response.status_code == 404
            assert second_response.status_code == 404

            return (
                _http_request_count(
                    metrics,
                    method="GET",
                    route="unmatched",
                    status_code="404",
                ),
                _http_request_count(
                    metrics,
                    method="GET",
                    route="/does-not-exist/12345",
                    status_code="404",
                ),
                _http_request_count(
                    metrics,
                    method="GET",
                    route="/does-not-exist/67890",
                    status_code="404",
                ),
            )

    unmatched_count, first_raw_path_count, second_raw_path_count = asyncio.run(run_requests())

    assert unmatched_count == 2
    assert first_raw_path_count is None
    assert second_raw_path_count is None
