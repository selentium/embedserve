from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

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


class ReorderingPreflightEmbedder:
    device = "cpu"
    dtype = "float32"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.first_preflight_started = threading.Event()
        self.release_first_preflight = threading.Event()

    def preflight(self, inputs: list[str]) -> list[int]:
        text = inputs[0]
        if text == "first":
            self.first_preflight_started.set()
            if not self.release_first_preflight.wait(timeout=5.0):
                raise AssertionError("first preflight was not released")
            time.sleep(0.05)
            return [2]

        if text == "second":
            if not self.first_preflight_started.wait(timeout=5.0):
                raise AssertionError("first preflight did not start")
            self.release_first_preflight.set()
            return [2]

        raise AssertionError(f"unexpected inputs: {inputs}")

    def embed(self, inputs: list[str]) -> EmbedResponse:
        self.calls.append(list(inputs))
        return _response_for_inputs(inputs)


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


def _runtime_unready(settings: Settings) -> RuntimeState:
    return RuntimeState(
        ready=False,
        mode="model",
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
        device="cpu",
        dtype="float32",
        reason="initialization_failed",
        detail="model bootstrap failed",
        embedder=None,
    )


def _metric_value(metrics_text: str, metric_name: str, *, labels: str | None = None) -> str | None:
    suffix = "" if labels is None else f"{{{labels}}}"
    needle = f"{metric_name}{suffix} "
    for line in metrics_text.splitlines():
        if line.startswith(needle):
            return line.removeprefix(needle)
    return None


def test_submit_time_preflight_validation_isolated_from_other_requests() -> None:
    """Ensure one overlength request fails validation without blocking a concurrent valid request from succeeding."""

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
    """Return a 503 overload once the batch queue is saturated while allowing earlier queued work to complete."""
    monkeypatch.setenv("MAX_BATCH_SIZE", "1")
    monkeypatch.setenv("MAX_BATCH_QUEUE_SIZE", "1")
    monkeypatch.setenv("BATCH_TIMEOUT_MS", "1000")
    monkeypatch.setenv("BATCH_REQUEST_TIMEOUT_MS", "5000")

    blocking_embedder = BlockingEmbedder(sleep_seconds=0.25)

    def initializer(settings: Settings) -> RuntimeState:
        return _runtime_with_embedder(settings, blocking_embedder)

    async def run() -> tuple[int, str, int, int, str]:
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
            metrics = await client.get("/metrics")
            return (
                overload.status_code,
                overload.json()["detail"],
                first.status_code,
                second.status_code,
                metrics.text,
            )

    overload_status, overload_detail, first_status, second_status, metrics_body = asyncio.run(run())
    assert overload_status == 503
    assert overload_detail == "Batch queue is full"
    assert first_status == 200
    assert second_status == 200
    assert (
        _metric_value(
            metrics_body,
            "embedserve_request_failures_total",
            labels='reason="overload"',
        )
        == "1.0"
    )


def test_embed_request_timeout_returns_timeout_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map per-request batch timeouts to a 503 response and increment the timeout-related metrics."""
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
    assert (
        _metric_value(
            metrics_body,
            "embedserve_request_failures_total",
            labels='reason="timeout"',
        )
        == "1.0"
    )


def test_embed_shutdown_rejection_returns_shutdown_503() -> None:
    """Return a 503 shutdown response when the batcher has stopped accepting new submissions."""

    def initializer(settings: Settings) -> RuntimeState:
        embedder, _, _ = make_fake_embedder(
            model_id=settings.MODEL_ID,
            revision=settings.MODEL_REVISION,
        )
        return _runtime_with_embedder(settings, embedder)

    async def run() -> tuple[int, str, str]:
        app = create_app(runtime_initializer=initializer)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            app.state.batcher._accepting = False
            response = await client.post("/embed", json={"inputs": ["hello"]})
            metrics = await client.get("/metrics")
            return response.status_code, response.json()["detail"], metrics.text

    status_code, detail, metrics_body = asyncio.run(run())
    assert status_code == 503
    assert detail == "Service is shutting down"
    assert (
        _metric_value(
            metrics_body,
            "embedserve_request_failures_total",
            labels='reason="shutdown"',
        )
        == "1.0"
    )


def test_embed_preflight_internal_failure_maps_to_500() -> None:
    """Map unexpected preflight exceptions to a 500 response and record them as internal request failures."""

    def initializer(settings: Settings) -> RuntimeState:
        return _runtime_with_embedder(settings, PreflightFailureEmbedder())

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
    assert (
        _metric_value(
            metrics_body,
            "embedserve_request_failures_total",
            labels='reason="internal_error"',
        )
        == "1.0"
    )


def test_embed_batch_inference_failure_maps_to_500_and_metric() -> None:
    """Map batch inference crashes to HTTP 500 and increment both batch and request failure metrics."""

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
    assert (
        _metric_value(
            metrics_body,
            "embedserve_request_failures_total",
            labels='reason="internal_error"',
        )
        == "1.0"
    )


def test_embed_validation_and_unready_do_not_increment_operational_failure_counter() -> None:
    """Keep validation 422s and unready 503s out of the operational failure counters reserved for runtime errors."""

    async def run() -> tuple[str | None, str | None]:
        app = create_app(runtime_initializer=_runtime_unready)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            validation_response = await client.post("/embed", json={"inputs": ["   "]})
            unready_response = await client.post("/embed", json={"inputs": ["hello"]})
            metrics = await client.get("/metrics")

        assert validation_response.status_code == 422
        assert unready_response.status_code == 503
        return (
            _metric_value(
                metrics.text,
                "embedserve_request_failures_total",
                labels='reason="internal_error"',
            ),
            _metric_value(
                metrics.text,
                "embedserve_request_failures_total",
                labels='reason="timeout"',
            ),
        )

    internal_error_count, timeout_count = asyncio.run(run())
    assert internal_error_count == "0.0"
    assert timeout_count == "0.0"


def test_embed_preserves_fifo_arrival_order_before_preflight_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve FIFO processing even when a later request finishes preflight before the earlier arrival."""
    monkeypatch.setenv("MAX_BATCH_SIZE", "1")
    monkeypatch.setenv("BATCH_TIMEOUT_MS", "1000")

    embedder = ReorderingPreflightEmbedder()

    def initializer(settings: Settings) -> RuntimeState:
        return _runtime_with_embedder(settings, embedder)

    async def run() -> tuple[int, int]:
        app = create_app(runtime_initializer=initializer)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            first_task = asyncio.create_task(client.post("/embed", json={"inputs": ["first"]}))
            started = await asyncio.to_thread(embedder.first_preflight_started.wait, 5.0)
            assert started

            second_task = asyncio.create_task(client.post("/embed", json={"inputs": ["second"]}))
            first, second = await asyncio.gather(first_task, second_task)
            return first.status_code, second.status_code

    first_status, second_status = asyncio.run(run())
    assert first_status == 200
    assert second_status == 200
    assert embedder.calls == [["first"], ["second"]]


def test_embed_does_not_let_preflight_starve_batch_inference_worker() -> None:
    """Prevent shared preflight threadpool work from starving the batch inference worker under tight executor limits."""
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import asyncio
        import json
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor

        import app.batching as batching_module
        from app.batching import DynamicBatcher
        from app.metrics import create_metrics
        from app.schemas import EmbeddingItem, EmbedResponse, UsageInfo


        class ThreadpoolStarvationEmbedder:
            device = "cpu"
            dtype = "float32"

            def __init__(self) -> None:
                self.first_started = threading.Event()
                self.second_started = threading.Event()
                self.release_first = threading.Event()
                self.release_second = threading.Event()
                self.embed_started = threading.Event()

            def preflight(self, inputs: list[str]) -> list[int]:
                text = inputs[0]
                if text == "first":
                    self.first_started.set()
                    if not self.release_first.wait(timeout=5.0):
                        raise AssertionError("first preflight was not released")
                    return [2]
                if text == "second":
                    self.second_started.set()
                    if not self.release_second.wait(timeout=5.0):
                        raise AssertionError("second preflight was not released")
                    return [2]
                raise AssertionError(f"unexpected inputs: {inputs}")

            def embed(self, inputs: list[str]) -> EmbedResponse:
                self.embed_started.set()
                return EmbedResponse(
                    data=[EmbeddingItem(index=0, embedding=[0.0, float(len(inputs[0]))])],
                    model="test-model",
                    revision="1" * 40,
                    dim=2,
                    usage=UsageInfo(tokens=2),
                )


        async def main() -> None:
            embedder = ThreadpoolStarvationEmbedder()
            executor = ThreadPoolExecutor(max_workers=1)
            batcher = DynamicBatcher(
                embedder=embedder,
                metrics=create_metrics(),
                max_batch_size=1,
                max_batch_tokens=100,
                batch_timeout_ms=1000,
                max_batch_queue_size=16,
            )

            async def run_in_limited_pool(func, *args):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(executor, func, *args)

            original_run_in_threadpool = getattr(batching_module, "run_in_threadpool", None)
            batching_module.run_in_threadpool = run_in_limited_pool
            try:
                await batcher.start()
                first_preflight = asyncio.create_task(
                    run_in_limited_pool(embedder.preflight, ["first"])
                )
                asyncio.create_task(run_in_limited_pool(embedder.preflight, ["second"]))

                started = await asyncio.to_thread(embedder.first_started.wait, 5.0)
                if not started:
                    raise AssertionError("first preflight did not start")
                embedder.release_first.set()
                token_counts = await asyncio.wait_for(first_preflight, timeout=1.0)
                submission = batcher.submit(["first"], token_counts)

                second_started = await asyncio.to_thread(embedder.second_started.wait, 5.0)
                if not second_started:
                    raise AssertionError("second preflight did not start")

                try:
                    response = await asyncio.wait_for(submission.future, timeout=0.2)
                except TimeoutError:
                    print(
                        json.dumps(
                            {"timed_out": True, "embed_started": embedder.embed_started.is_set()}
                        ),
                        flush=True,
                    )
                    os._exit(1)

                print(
                    json.dumps(
                        {
                            "timed_out": False,
                            "embed_started": embedder.embed_started.is_set(),
                            "embedding": response.data[0].embedding,
                        }
                    ),
                    flush=True,
                )
                os._exit(0)
            finally:
                if original_run_in_threadpool is None:
                    del batching_module.run_in_threadpool
                else:
                    batching_module.run_in_threadpool = original_run_in_threadpool


        asyncio.run(main())
        """
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "shared preflight work starved batch inference long enough for the repro to hang",
            pytrace=False,
        )

    assert result.returncode == 0, result.stdout
