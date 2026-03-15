from __future__ import annotations

import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app.engine.embedder import Embedder
from app.main import create_app
from app.runtime import RuntimeState
from app.schemas import EmbeddingItem, EmbedResponse, UsageInfo
from app.settings import Settings
from tests.fakes import make_fake_embedder


def _response_for_inputs(inputs: list[str]) -> EmbedResponse:
    return EmbedResponse(
        data=[
            EmbeddingItem(index=index, embedding=[float(index), float(len(value))])
            for index, value in enumerate(inputs)
        ],
        model="sentence-transformers/all-MiniLM-L6-v2",
        revision="826711e54e001c83835913827a843d8dd0a1def9",
        dim=2,
        usage=UsageInfo(tokens=sum(len(value.split()) + 2 for value in inputs)),
    )


class BlockingEmbedder:
    def __init__(self, *, sleep_seconds: float) -> None:
        self.device = "cpu"
        self.dtype = "float32"
        self.sleep_seconds = sleep_seconds

    def preflight(self, inputs: list[str]) -> list[int]:
        return [len(value.split()) + 2 for value in inputs]

    def embed(self, inputs: list[str]) -> EmbedResponse:
        time.sleep(self.sleep_seconds)
        return _response_for_inputs(inputs)


class PreflightFailureEmbedder:
    device = "cpu"
    dtype = "float32"

    def preflight(self, inputs: list[str]) -> list[int]:
        raise RuntimeError("preflight failed")

    def embed(self, inputs: list[str]) -> EmbedResponse:
        raise AssertionError


class InferenceFailureEmbedder:
    device = "cpu"
    dtype = "float32"

    def preflight(self, inputs: list[str]) -> list[int]:
        return [2 for _ in inputs]

    def embed(self, inputs: list[str]) -> EmbedResponse:
        raise RuntimeError("inference failed")


def _runtime_with_embedder(settings: Settings, embedder: Embedder) -> RuntimeState:
    return RuntimeState(
        ready=True,
        mode="model",
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
        device="cpu",
        dtype="float32",
        reason=None,
        detail=None,
        embedder=embedder,
    )


def test_submit_time_preflight_validation_isolated_from_other_requests() -> None:
    def initializer(settings: Settings) -> RuntimeState:
        embedder, _, _ = make_fake_embedder(
            model_id=settings.MODEL_ID,
            revision=settings.MODEL_REVISION,
            truncate=False,
            effective_max_length=4,
            token_map={"too long": [1, 2, 3, 4, 5], "ok": [1, 2]},
        )
        return _runtime_with_embedder(settings, embedder)

    async def run() -> tuple[int, int]:
        app = create_app(runtime_initializer=initializer)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            valid_task = asyncio.create_task(client.post("/embed", json={"inputs": ["ok"]}))
            invalid_task = asyncio.create_task(client.post("/embed", json={"inputs": ["too long"]}))
            valid, invalid = await asyncio.gather(valid_task, invalid_task)
            return valid.status_code, invalid.status_code

    valid_status, invalid_status = asyncio.run(run())
    assert valid_status == 200
    assert invalid_status == 422


def test_embed_queue_saturation_returns_overload_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_BATCH_SIZE", "1")
    monkeypatch.setenv("MAX_BATCH_QUEUE_SIZE", "1")
    monkeypatch.setenv("BATCH_TIMEOUT_MS", "1000")
    monkeypatch.setenv("BATCH_REQUEST_TIMEOUT_MS", "5000")

    blocking_embedder = BlockingEmbedder(sleep_seconds=0.25)

    def initializer(settings: Settings) -> RuntimeState:
        return _runtime_with_embedder(settings, blocking_embedder)

    async def run() -> tuple[int, str, int, int]:
        app = create_app(runtime_initializer=initializer)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            first_task = asyncio.create_task(client.post("/embed", json={"inputs": ["first"]}))
            await asyncio.sleep(0.05)
            second_task = asyncio.create_task(client.post("/embed", json={"inputs": ["second"]}))
            await asyncio.sleep(0.05)
            overload = await client.post("/embed", json={"inputs": ["third"]})
            first = await first_task
            second = await second_task
            return (
                overload.status_code,
                overload.json()["detail"],
                first.status_code,
                second.status_code,
            )

    overload_status, overload_detail, first_status, second_status = asyncio.run(run())
    assert overload_status == 503
    assert overload_detail == "Batch queue is full"
    assert first_status == 200
    assert second_status == 200


def test_embed_request_timeout_returns_timeout_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_BATCH_SIZE", "1")
    monkeypatch.setenv("BATCH_TIMEOUT_MS", "1000")
    monkeypatch.setenv("BATCH_REQUEST_TIMEOUT_MS", "20")

    blocking_embedder = BlockingEmbedder(sleep_seconds=0.25)

    def initializer(settings: Settings) -> RuntimeState:
        return _runtime_with_embedder(settings, blocking_embedder)

    async def run() -> tuple[int, str, str]:
        app = create_app(runtime_initializer=initializer)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            response = await client.post("/embed", json={"inputs": ["slow"]})
            metrics = await client.get("/metrics")
            return response.status_code, response.json()["detail"], metrics.text

    status_code, detail, metrics_body = asyncio.run(run())
    assert status_code == 503
    assert detail == "Embedding request timed out"
    assert "embedserve_batch_request_timeouts_total 1.0" in metrics_body


def test_embed_shutdown_rejection_returns_shutdown_503() -> None:
    def initializer(settings: Settings) -> RuntimeState:
        embedder, _, _ = make_fake_embedder(
            model_id=settings.MODEL_ID,
            revision=settings.MODEL_REVISION,
        )
        return _runtime_with_embedder(settings, embedder)

    async def run() -> tuple[int, str]:
        app = create_app(runtime_initializer=initializer)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            app.state.batcher._accepting = False
            response = await client.post("/embed", json={"inputs": ["hello"]})
            return response.status_code, response.json()["detail"]

    status_code, detail = asyncio.run(run())
    assert status_code == 503
    assert detail == "Service is shutting down"


def test_embed_preflight_internal_failure_maps_to_500() -> None:
    def initializer(settings: Settings) -> RuntimeState:
        return _runtime_with_embedder(settings, PreflightFailureEmbedder())

    async def run() -> tuple[int, str]:
        app = create_app(runtime_initializer=initializer)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            response = await client.post("/embed", json={"inputs": ["hello"]})
            return response.status_code, response.json()["detail"]

    status_code, detail = asyncio.run(run())
    assert status_code == 500
    assert detail == "Embedding inference failed"


def test_embed_batch_inference_failure_maps_to_500_and_metric() -> None:
    def initializer(settings: Settings) -> RuntimeState:
        return _runtime_with_embedder(settings, InferenceFailureEmbedder())

    async def run() -> tuple[int, str, str]:
        app = create_app(runtime_initializer=initializer)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            response = await client.post("/embed", json={"inputs": ["hello"]})
            metrics = await client.get("/metrics")
            return response.status_code, response.json()["detail"], metrics.text

    status_code, detail, metrics_body = asyncio.run(run())
    assert status_code == 500
    assert detail == "Embedding inference failed"
    assert "embedserve_batch_inference_failures_total 1.0" in metrics_body
