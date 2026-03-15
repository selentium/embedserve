from __future__ import annotations

import asyncio
import threading
import time
from contextlib import suppress
from typing import Protocol

import pytest

from app.batching import (
    BatcherQueueFullError,
    BatcherShuttingDownError,
    BatchInferenceError,
    BatchSubmission,
    DynamicBatcher,
)
from app.metrics import AppMetrics, create_metrics
from app.schemas import EmbeddingItem, EmbedResponse, UsageInfo


def _counter_value(metrics: AppMetrics, name: str, labels: dict[str, str]) -> float | None:
    metric_lookup = {
        "batch_flush_total": metrics.batch_flush_total,
    }
    collector = metric_lookup[name]
    for metric in collector.collect():
        for sample in metric.samples:
            if not sample.name.endswith("_total"):
                continue
            if sample.labels == labels:
                return sample.value
    return None


def _gauge_value(metrics: AppMetrics, name: str) -> float | None:
    metric_lookup = {
        "batch_queue_depth": metrics.batch_queue_depth,
    }
    collector = metric_lookup[name]
    for metric in collector.collect():
        for sample in metric.samples:
            if sample.name.endswith("batch_queue_depth"):
                return sample.value
    return None


class RecordingEmbedder:
    def __init__(self) -> None:
        self.device = "cpu"
        self.dtype = "float32"
        self.calls: list[list[str]] = []

    def preflight(self, inputs: list[str]) -> list[int]:
        return [len(value.split()) + 2 for value in inputs]

    def embed(self, inputs: list[str]) -> EmbedResponse:
        self.calls.append(list(inputs))
        return EmbedResponse(
            data=[
                EmbeddingItem(index=index, embedding=[float(index), float(len(value))])
                for index, value in enumerate(inputs)
            ],
            model="test-model",
            revision="1" * 40,
            dim=2,
            usage=UsageInfo(tokens=sum(self.preflight(inputs))),
        )


class SleepingEmbedder(RecordingEmbedder):
    def __init__(self, *, sleep_seconds: float) -> None:
        super().__init__()
        self.sleep_seconds = sleep_seconds

    def embed(self, inputs: list[str]) -> EmbedResponse:
        time.sleep(self.sleep_seconds)
        return super().embed(inputs)


class StartedSleepingEmbedder(RecordingEmbedder):
    def __init__(self, *, sleep_seconds: float) -> None:
        super().__init__()
        self.sleep_seconds = sleep_seconds
        self.started = threading.Event()

    def embed(self, inputs: list[str]) -> EmbedResponse:
        self.started.set()
        time.sleep(self.sleep_seconds)
        return super().embed(inputs)


class EventBlockingEmbedder(RecordingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def embed(self, inputs: list[str]) -> EmbedResponse:
        self.started.set()
        self.release.wait(timeout=5.0)
        return super().embed(inputs)


class NonStoppableBlockingEmbedder(RecordingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.allow_finish = threading.Event()
        self.finished = threading.Event()

    def embed(self, inputs: list[str]) -> EmbedResponse:
        self.started.set()
        try:
            self.allow_finish.wait(timeout=5.0)
            return super().embed(inputs)
        finally:
            self.finished.set()


class InvalidFanOutEmbedder(RecordingEmbedder):
    def embed(self, inputs: list[str]) -> EmbedResponse:
        self.calls.append(list(inputs))
        return EmbedResponse(
            data=[EmbeddingItem(index=0, embedding=[0.0, float(len(inputs[0]))])],
            model="test-model",
            revision="1" * 40,
            dim=2,
            usage=UsageInfo(tokens=sum(self.preflight(inputs))),
        )


class _HasStartedEvent(Protocol):
    started: threading.Event


async def _wait_until_started(embedder: _HasStartedEvent) -> None:
    for _ in range(200):
        if embedder.started.is_set():
            return
        await asyncio.sleep(0.01)
    msg = "batch inference did not start in time"
    raise AssertionError(msg)


def _make_batcher(
    *,
    embedder: RecordingEmbedder,
    metrics: AppMetrics,
    max_batch_size: int,
    max_batch_tokens: int,
    batch_timeout_ms: int,
    max_batch_queue_size: int = 16,
) -> DynamicBatcher:
    return DynamicBatcher(
        embedder=embedder,
        metrics=metrics,
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        batch_timeout_ms=batch_timeout_ms,
        max_batch_queue_size=max_batch_queue_size,
    )


def _submit(
    batcher: DynamicBatcher,
    *,
    text: str,
    tokens: int,
) -> BatchSubmission:
    return batcher.submit([text], [tokens])


def test_batcher_flushes_by_max_batch_size() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = RecordingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=2,
            max_batch_tokens=100,
            batch_timeout_ms=50,
        )
        await batcher.start()
        try:
            first = _submit(batcher, text="one", tokens=2)
            second = _submit(batcher, text="two", tokens=2)
            third = _submit(batcher, text="three", tokens=2)

            await asyncio.wait_for(first.future, timeout=1)
            await asyncio.wait_for(second.future, timeout=1)
            await asyncio.wait_for(third.future, timeout=1)
        finally:
            await batcher.shutdown()

        assert embedder.calls == [["one", "two"], ["three"]]
        assert _counter_value(
            metrics,
            "batch_flush_total",
            labels={"reason": "max_batch_size"},
        ) == pytest.approx(1.0)

    asyncio.run(run())


def test_batcher_enforces_fifo_when_token_cap_blocks_head_fit() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = RecordingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=8,
            max_batch_tokens=10,
            batch_timeout_ms=100,
        )
        await batcher.start()
        try:
            first = _submit(batcher, text="first", tokens=6)
            second = _submit(batcher, text="second", tokens=5)
            third = _submit(batcher, text="third", tokens=1)

            await asyncio.wait_for(first.future, timeout=1)
            await asyncio.wait_for(second.future, timeout=1)
            await asyncio.wait_for(third.future, timeout=1)
        finally:
            await batcher.shutdown()

        assert embedder.calls == [["first"], ["second", "third"]]
        assert _counter_value(
            metrics,
            "batch_flush_total",
            labels={"reason": "max_batch_tokens"},
        ) == pytest.approx(1.0)

    asyncio.run(run())


def test_batcher_processes_singleton_when_request_exceeds_token_cap() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = RecordingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=4,
            max_batch_tokens=3,
            batch_timeout_ms=50,
        )
        await batcher.start()
        try:
            submission = _submit(batcher, text="oversized", tokens=7)
            response = await asyncio.wait_for(submission.future, timeout=1)
        finally:
            await batcher.shutdown()

        assert response.data[0].index == 0
        assert embedder.calls == [["oversized"]]
        assert _counter_value(
            metrics,
            "batch_flush_total",
            labels={"reason": "max_batch_tokens"},
        ) == pytest.approx(1.0)

    asyncio.run(run())


def test_batcher_cancellation_prunes_job_before_inference() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = RecordingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=8,
            max_batch_tokens=100,
            batch_timeout_ms=50,
        )
        await batcher.start()
        try:
            cancelled = _submit(batcher, text="cancel-me", tokens=2)
            kept = _submit(batcher, text="keep-me", tokens=2)
            batcher.cancel(cancelled.job_id)

            response = await asyncio.wait_for(kept.future, timeout=1)
            assert response.data[0].index == 0
            assert cancelled.future.cancelled()
        finally:
            await batcher.shutdown()

        assert embedder.calls == [["keep-me"]]

    asyncio.run(run())


def test_batcher_rejects_when_queue_is_full() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = EventBlockingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=1,
            max_batch_tokens=100,
            batch_timeout_ms=1000,
            max_batch_queue_size=1,
        )
        await batcher.start()
        try:
            first = _submit(batcher, text="first", tokens=1)
            await _wait_until_started(embedder)

            queued = _submit(batcher, text="queued", tokens=1)
            assert _gauge_value(metrics, "batch_queue_depth") == 1

            with pytest.raises(BatcherQueueFullError):
                _submit(batcher, text="overflow", tokens=1)

            embedder.release.set()
            await asyncio.wait_for(first.future, timeout=2)
            await asyncio.wait_for(queued.future, timeout=2)
        finally:
            embedder.release.set()
            await batcher.shutdown()

    asyncio.run(run())


def test_batcher_prunes_cancelled_queue_entry_before_reporting_overload() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = StartedSleepingEmbedder(sleep_seconds=0.25)
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=1,
            max_batch_tokens=100,
            batch_timeout_ms=1000,
            max_batch_queue_size=1,
        )
        await batcher.start()
        try:
            first = _submit(batcher, text="first", tokens=1)
            await _wait_until_started(embedder)

            cancelled = _submit(batcher, text="cancelled", tokens=1)
            batcher.cancel(cancelled.job_id)
            replacement = _submit(batcher, text="replacement", tokens=1)

            assert cancelled.future.cancelled()

            await asyncio.wait_for(first.future, timeout=5)
            replacement_response = await asyncio.wait_for(replacement.future, timeout=5)
        finally:
            await batcher.shutdown()

        assert replacement_response.data[0].index == 0
        assert embedder.calls == [["first"], ["replacement"]]

    asyncio.run(run())


def test_batcher_shutdown_completes_inflight_and_fails_unresolved_queue() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = EventBlockingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=1,
            max_batch_tokens=100,
            batch_timeout_ms=1000,
        )
        await batcher.start()
        try:
            first = _submit(batcher, text="first", tokens=1)
            await _wait_until_started(embedder)
            second = _submit(batcher, text="second", tokens=1)

            shutdown_task = asyncio.create_task(batcher.shutdown())
            await asyncio.sleep(0.05)
            embedder.release.set()
            await asyncio.wait_for(shutdown_task, timeout=5)

            first_response = await asyncio.wait_for(first.future, timeout=5)
            assert first_response.data[0].index == 0

            with pytest.raises(BatcherShuttingDownError):
                await asyncio.wait_for(second.future, timeout=5)
        finally:
            await batcher.shutdown()

    asyncio.run(run())


def test_batcher_shutdown_waits_for_non_stoppable_inflight_inference() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = NonStoppableBlockingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=1,
            max_batch_tokens=100,
            batch_timeout_ms=1000,
        )
        await batcher.start()
        first: BatchSubmission | None = None
        shutdown_task: asyncio.Task[None] | None = None
        try:
            first = _submit(batcher, text="first", tokens=1)
            await _wait_until_started(embedder)

            shutdown_task = asyncio.create_task(batcher.shutdown())
            done, _ = await asyncio.wait({shutdown_task}, timeout=1.25)
            assert shutdown_task not in done
            assert not embedder.finished.is_set()

            embedder.allow_finish.set()
            await asyncio.wait_for(shutdown_task, timeout=2)

            first_response = await asyncio.wait_for(first.future, timeout=2)
            assert first_response.data[0].index == 0
        finally:
            embedder.allow_finish.set()
            if first is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(first.future, timeout=2)
            if shutdown_task is not None and not shutdown_task.done():
                await asyncio.wait_for(shutdown_task, timeout=2)
            if batcher._worker_task is not None and not batcher._worker_task.done():
                batcher._worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await batcher._worker_task
            batcher._worker_task = None

    asyncio.run(run())


def test_batcher_shutdown_caller_timeout_does_not_cancel_inflight_worker() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = NonStoppableBlockingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=1,
            max_batch_tokens=100,
            batch_timeout_ms=1000,
        )
        await batcher.start()
        first: BatchSubmission | None = None
        try:
            first = _submit(batcher, text="first", tokens=1)
            await _wait_until_started(embedder)

            shutdown_timeout = (
                batcher._SHUTDOWN_WAIT_SECONDS + batcher._SHUTDOWN_GRACE_SECONDS + 0.25
            )
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(batcher.shutdown(), timeout=shutdown_timeout)

            worker_task = batcher._worker_task
            assert worker_task is not None
            assert not worker_task.done()
            assert not first.future.done()

            embedder.allow_finish.set()
            await asyncio.wait_for(worker_task, timeout=2)

            first_response = await asyncio.wait_for(first.future, timeout=2)
            assert first_response.data[0].index == 0
        finally:
            embedder.allow_finish.set()
            if first is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(first.future, timeout=2)
            if batcher._worker_task is not None and not batcher._worker_task.done():
                batcher._worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await batcher._worker_task
            batcher._worker_task = None

    asyncio.run(run())


def test_batcher_shutdown_rejects_queued_jobs_before_inflight_completes() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = EventBlockingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=1,
            max_batch_tokens=100,
            batch_timeout_ms=1000,
        )
        await batcher.start()
        first: BatchSubmission | None = None
        queued: BatchSubmission | None = None
        shutdown_task: asyncio.Task[None] | None = None
        try:
            first = _submit(batcher, text="first", tokens=1)
            await _wait_until_started(embedder)
            queued = _submit(batcher, text="queued", tokens=1)

            shutdown_task = asyncio.create_task(batcher.shutdown())
            await asyncio.sleep(0.05)

            done, _ = await asyncio.wait({queued.future}, timeout=0.25)
            assert queued.future in done
            with pytest.raises(BatcherShuttingDownError):
                queued.future.result()

            assert not first.future.done()

            embedder.release.set()
            await asyncio.wait_for(shutdown_task, timeout=2)
            first_response = await asyncio.wait_for(first.future, timeout=2)
            assert first_response.data[0].index == 0
        finally:
            embedder.release.set()
            if shutdown_task is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(shutdown_task, timeout=2)
            if first is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(first.future, timeout=2)
            if queued is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(queued.future, timeout=2)
            if batcher._worker_task is not None and not batcher._worker_task.done():
                batcher._worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await batcher._worker_task
            batcher._worker_task = None

    asyncio.run(run())


def test_batcher_shutdown_does_not_block_when_queue_is_full() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = EventBlockingEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=1,
            max_batch_tokens=100,
            batch_timeout_ms=1000,
            max_batch_queue_size=1,
        )
        await batcher.start()
        shutdown_task: asyncio.Task[None] | None = None
        try:
            first = _submit(batcher, text="first", tokens=1)
            await _wait_until_started(embedder)
            queued = _submit(batcher, text="queued", tokens=1)

            shutdown_task = asyncio.create_task(batcher.shutdown())
            done, _ = await asyncio.wait({shutdown_task}, timeout=1.2)
            assert shutdown_task in done

            first_response = await asyncio.wait_for(first.future, timeout=0.5)
            assert first_response.data[0].index == 0

            with pytest.raises(BatcherShuttingDownError):
                await asyncio.wait_for(queued.future, timeout=0.5)
        finally:
            embedder.release.set()
            if shutdown_task is not None:
                if not shutdown_task.done():
                    shutdown_task.cancel()
                with suppress(asyncio.CancelledError):
                    await shutdown_task
            if batcher._worker_task is not None and not batcher._worker_task.done():
                batcher._worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await batcher._worker_task
            batcher._worker_task = None

    asyncio.run(run())


def test_batcher_rejects_entire_batch_when_fan_out_payload_is_malformed() -> None:
    async def run() -> None:
        metrics = create_metrics()
        embedder = InvalidFanOutEmbedder()
        batcher = _make_batcher(
            embedder=embedder,
            metrics=metrics,
            max_batch_size=2,
            max_batch_tokens=100,
            batch_timeout_ms=50,
        )
        await batcher.start()
        try:
            first = _submit(batcher, text="first", tokens=1)
            second = _submit(batcher, text="second", tokens=1)

            with pytest.raises(BatchInferenceError):
                await asyncio.wait_for(first.future, timeout=1)
            with pytest.raises(BatchInferenceError):
                await asyncio.wait_for(second.future, timeout=1)
        finally:
            await batcher.shutdown()

        assert embedder.calls == [["first", "second"]]

    asyncio.run(run())
