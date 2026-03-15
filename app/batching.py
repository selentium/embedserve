from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from fastapi.concurrency import run_in_threadpool

from app.engine.embedder import Embedder
from app.metrics import AppMetrics
from app.schemas import EmbeddingItem, EmbedResponse, UsageInfo

FlushReason = Literal["max_batch_size", "max_batch_tokens", "timeout", "shutdown"]


class BatcherQueueFullError(Exception):
    pass


class BatcherShuttingDownError(Exception):
    pass


class BatchInferenceError(Exception):
    pass


@dataclass(frozen=True)
class BatchSubmission:
    job_id: int
    future: asyncio.Future[EmbedResponse]


@dataclass
class _BatchJob:
    job_id: int
    inputs: list[str]
    token_counts: list[int]
    enqueued_at: float
    future: asyncio.Future[EmbedResponse]
    cancelled: bool = False

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)


class _ShutdownSignal:
    pass


class DynamicBatcher:
    _QUEUE_POLL_INTERVAL_SECONDS = 0.001
    _SHUTDOWN_WAIT_SECONDS = 1.0
    _SHUTDOWN_GRACE_SECONDS = 0.2

    def __init__(
        self,
        *,
        embedder: Embedder,
        metrics: AppMetrics,
        max_batch_size: int,
        max_batch_tokens: int,
        batch_timeout_ms: int,
        max_batch_queue_size: int,
    ) -> None:
        self._embedder = embedder
        self._metrics = metrics
        self._max_batch_size = max_batch_size
        self._max_batch_tokens = max_batch_tokens
        self._batch_timeout_seconds = batch_timeout_ms / 1000.0
        self._queue: asyncio.Queue[_BatchJob | _ShutdownSignal] = asyncio.Queue(
            maxsize=max_batch_queue_size
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._embed_inflight = False
        self._accepting = False
        self._shutdown_requested = False
        self._next_job_id = 1
        self._jobs_by_id: dict[int, _BatchJob] = {}

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._accepting = True
        self._shutdown_requested = False
        self._worker_task = asyncio.create_task(self._worker(), name="embedserve-batch-worker")
        self._set_queue_depth()

    async def shutdown(self) -> None:
        if self._worker_task is None:
            return
        worker_task = self._worker_task
        self._accepting = False
        self._shutdown_requested = True
        with suppress(asyncio.QueueFull):
            self._queue.put_nowait(_ShutdownSignal())

        deadline = perf_counter() + self._SHUTDOWN_WAIT_SECONDS
        while not worker_task.done() and perf_counter() < deadline:
            await asyncio.sleep(self._QUEUE_POLL_INTERVAL_SECONDS)

        if not worker_task.done():
            self._request_embedder_stop()
            grace_deadline = perf_counter() + self._SHUTDOWN_GRACE_SECONDS
            while not worker_task.done() and perf_counter() < grace_deadline:
                await asyncio.sleep(self._QUEUE_POLL_INTERVAL_SECONDS)

        if not worker_task.done():
            if not self._embed_inflight:
                worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await worker_task
                self._fail_pending_jobs_with_shutdown()
                self._drain_with_shutdown()
            else:
                await worker_task
        else:
            worker_task.result()

        self._worker_task = None
        self._embed_inflight = False
        self._set_queue_depth()

    def submit(self, inputs: list[str], token_counts: list[int]) -> BatchSubmission:
        if not self._accepting:
            raise BatcherShuttingDownError

        loop = asyncio.get_running_loop()
        job = _BatchJob(
            job_id=self._next_job_id,
            inputs=list(inputs),
            token_counts=list(token_counts),
            enqueued_at=perf_counter(),
            future=loop.create_future(),
        )
        self._next_job_id += 1

        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            self._prune_cancelled_jobs_from_queue()
            try:
                self._queue.put_nowait(job)
            except asyncio.QueueFull as retry_exc:
                raise BatcherQueueFullError from retry_exc

        self._jobs_by_id[job.job_id] = job
        self._set_queue_depth()
        return BatchSubmission(job_id=job.job_id, future=job.future)

    def cancel(self, job_id: int) -> None:
        job = self._jobs_by_id.get(job_id)
        if job is None:
            return
        job.cancelled = True
        if not job.future.done():
            job.future.cancel()
        self._prune_cancelled_jobs_from_queue()

    async def _worker(self) -> None:
        carry: _BatchJob | None = None

        while True:
            if carry is None:
                if self._shutdown_requested:
                    break
                queued_item = await self._wait_for_next_item()
                if queued_item is None:
                    continue
                if isinstance(queued_item, _ShutdownSignal):
                    if self._shutdown_requested:
                        break
                    continue
                carry = queued_item

            job = carry
            carry = None
            if self._should_prune(job):
                self._complete_job(job)
                continue

            batch: list[_BatchJob] = [job]
            reason: FlushReason = "timeout"
            total_tokens = job.total_tokens

            if total_tokens > self._max_batch_tokens:
                reason = "max_batch_tokens"
            else:
                deadline = perf_counter() + self._batch_timeout_seconds
                while len(batch) < self._max_batch_size:
                    if self._shutdown_requested:
                        reason = "shutdown"
                        break
                    remaining = deadline - perf_counter()
                    if remaining <= 0:
                        reason = "timeout"
                        break

                    queued_item = await self._get_with_timeout(remaining)
                    if queued_item is None:
                        reason = "timeout"
                        break
                    if isinstance(queued_item, _ShutdownSignal):
                        if self._shutdown_requested:
                            reason = "shutdown"
                            break
                        continue

                    self._set_queue_depth()
                    if self._should_prune(queued_item):
                        self._complete_job(queued_item)
                        continue

                    next_total_tokens = queued_item.total_tokens
                    if total_tokens + next_total_tokens > self._max_batch_tokens:
                        carry = queued_item
                        reason = "max_batch_tokens"
                        break

                    batch.append(queued_item)
                    total_tokens += next_total_tokens
                    if len(batch) >= self._max_batch_size:
                        reason = "max_batch_size"
                        break

            await self._flush_batch(batch, reason=reason)
            if self._shutdown_requested:
                break

        if carry is not None:
            self._fail_with_shutdown(carry)

        self._drain_with_shutdown()

    async def _get_with_timeout(
        self,
        timeout_seconds: float,
    ) -> _BatchJob | _ShutdownSignal | None:
        deadline = perf_counter() + timeout_seconds
        while True:
            queued_item = self._pop_next_item_nowait()
            if queued_item is not None:
                return queued_item
            if perf_counter() >= deadline:
                return None
            await asyncio.sleep(self._QUEUE_POLL_INTERVAL_SECONDS)

    async def _wait_for_next_item(self) -> _BatchJob | _ShutdownSignal | None:
        while True:
            queued_item = self._pop_next_item_nowait()
            if queued_item is not None:
                return queued_item
            if self._shutdown_requested:
                return _ShutdownSignal()
            await asyncio.sleep(self._QUEUE_POLL_INTERVAL_SECONDS)

    def _pop_next_item_nowait(self) -> _BatchJob | _ShutdownSignal | None:
        try:
            queued_item = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        self._set_queue_depth()
        return queued_item

    def _drain_with_shutdown(self) -> None:
        while True:
            queued_item = self._pop_next_item_nowait()
            if queued_item is None:
                break
            if isinstance(queued_item, _ShutdownSignal):
                continue
            self._fail_with_shutdown(queued_item)

    def _fail_pending_jobs_with_shutdown(self) -> None:
        for job in list(self._jobs_by_id.values()):
            self._fail_with_shutdown(job)

    async def _flush_batch(self, batch: list[_BatchJob], *, reason: FlushReason) -> None:
        active_jobs = [job for job in batch if not self._should_prune(job)]
        for job in batch:
            if job not in active_jobs:
                self._complete_job(job)

        if not active_jobs:
            return

        flushed_at = perf_counter()
        batch_inputs: list[str] = []
        total_tokens = 0
        for job in active_jobs:
            self._metrics.batch_queue_wait_seconds.observe(flushed_at - job.enqueued_at)
            batch_inputs.extend(job.inputs)
            total_tokens += job.total_tokens

        self._metrics.batch_flush_total.labels(reason=reason).inc()
        self._metrics.batch_size.observe(len(active_jobs))
        self._metrics.batch_token_count.observe(total_tokens)

        self._embed_inflight = True
        try:
            merged_response = await run_in_threadpool(self._embedder.embed, batch_inputs)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._metrics.batch_inference_failures_total.inc()
            for job in active_jobs:
                self._set_job_exception(job, BatchInferenceError())
            return
        finally:
            self._embed_inflight = False

        self._fan_out(active_jobs, merged_response)

    def _fan_out(self, jobs: list[_BatchJob], merged_response: EmbedResponse) -> None:
        try:
            cursor = 0
            payload_items = merged_response.data
            responses: list[tuple[_BatchJob, EmbedResponse]] = []
            for job in jobs:
                request_items = payload_items[cursor : cursor + len(job.inputs)]
                if len(request_items) != len(job.inputs):
                    raise ValueError
                cursor += len(job.inputs)

                responses.append(
                    (
                        job,
                        EmbedResponse(
                            data=[
                                EmbeddingItem(index=index, embedding=item.embedding)
                                for index, item in enumerate(request_items)
                            ],
                            model=merged_response.model,
                            revision=merged_response.revision,
                            dim=merged_response.dim,
                            usage=UsageInfo(tokens=sum(job.token_counts)),
                        ),
                    )
                )

            if cursor != len(payload_items):
                raise ValueError

            for job, response in responses:
                self._set_job_result(job, response)
        except Exception:
            self._metrics.batch_inference_failures_total.inc()
            for job in jobs:
                self._set_job_exception(job, BatchInferenceError())

    def _fail_with_shutdown(self, job: _BatchJob) -> None:
        if self._should_prune(job):
            self._complete_job(job)
            return
        self._set_job_exception(job, BatcherShuttingDownError())

    def _set_job_result(self, job: _BatchJob, response: EmbedResponse) -> None:
        try:
            if job.future.done():
                return
            job.future.set_result(response)
        finally:
            self._complete_job(job)

    def _set_job_exception(self, job: _BatchJob, error: Exception) -> None:
        try:
            if job.future.done():
                return
            job.future.set_exception(error)
        finally:
            self._complete_job(job)

    def _complete_job(self, job: _BatchJob) -> None:
        self._jobs_by_id.pop(job.job_id, None)
        self._set_queue_depth()

    def _prune_cancelled_jobs_from_queue(self) -> None:
        queue_items = getattr(self._queue, "_queue", None)
        if queue_items is None:
            return

        pruned_jobs: list[_BatchJob] = []
        retained_items: list[_BatchJob | _ShutdownSignal] = []
        for queued_item in queue_items:
            if isinstance(queued_item, _BatchJob) and self._should_prune(queued_item):
                pruned_jobs.append(queued_item)
                continue
            retained_items.append(queued_item)

        if not pruned_jobs:
            return

        queue_items.clear()
        queue_items.extend(retained_items)
        if hasattr(self._queue, "_unfinished_tasks"):
            self._queue._unfinished_tasks = max(
                0,
                self._queue._unfinished_tasks - len(pruned_jobs),
            )
            if self._queue._unfinished_tasks == 0:
                finished = getattr(self._queue, "_finished", None)
                if finished is not None:
                    finished.set()

        for job in pruned_jobs:
            self._jobs_by_id.pop(job.job_id, None)

        self._set_queue_depth()

    def _set_queue_depth(self) -> None:
        self._metrics.batch_queue_depth.set(self._queue.qsize())

    def _request_embedder_stop(self) -> None:
        for attr in ("cancel", "shutdown", "close"):
            maybe_method = getattr(self._embedder, attr, None)
            if callable(maybe_method):
                maybe_method()
                return

        release = getattr(self._embedder, "release", None)
        maybe_set = getattr(release, "set", None)
        if callable(maybe_set):
            maybe_set()

    @staticmethod
    def _should_prune(job: _BatchJob) -> bool:
        return job.cancelled or job.future.done()
