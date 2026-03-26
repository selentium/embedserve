from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    GCCollector,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)

if TYPE_CHECKING:
    from app.runtime import RuntimeState

RequestFailureReason = Literal["overload", "timeout", "internal_error", "shutdown"]


@dataclass(frozen=True)
class AppMetrics:
    registry: CollectorRegistry
    http_requests_total: Counter
    http_request_duration_seconds: Histogram
    app_ready: Gauge
    batch_queue_depth: Gauge
    batch_queue_wait_seconds: Histogram
    batch_size: Histogram
    batch_token_count: Histogram
    batch_flush_total: Counter
    batch_overload_rejections_total: Counter
    batch_request_timeouts_total: Counter
    batch_request_cancellations_total: Counter
    batch_shutdown_rejections_total: Counter
    batch_inference_failures_total: Counter
    gpu_memory_allocated_bytes: Gauge
    gpu_memory_reserved_bytes: Gauge
    gpu_oom_total: Counter
    request_failures_total: Counter
    unhandled_exceptions_total: Counter


def create_metrics() -> AppMetrics:
    registry = CollectorRegistry(auto_describe=True)

    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    GCCollector(registry=registry)

    http_requests_total = Counter(
        "embedserve_http_requests_total",
        "Total HTTP requests handled by the service.",
        labelnames=("method", "route", "status_code"),
        registry=registry,
    )
    http_request_duration_seconds = Histogram(
        "embedserve_http_request_duration_seconds",
        "HTTP request latency in seconds.",
        labelnames=("method", "route"),
        registry=registry,
    )
    app_ready = Gauge(
        "embedserve_app_ready",
        "Application readiness state by mode.",
        labelnames=("mode",),
        registry=registry,
    )
    batch_queue_depth = Gauge(
        "embedserve_batch_queue_depth",
        "Current number of pending embedding jobs in the batch queue.",
        registry=registry,
    )
    batch_queue_wait_seconds = Histogram(
        "embedserve_batch_queue_wait_seconds",
        "Time spent waiting in the batch queue before inference execution.",
        buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        registry=registry,
    )
    batch_size = Histogram(
        "embedserve_batch_size",
        "Number of requests merged into each inference batch.",
        buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        registry=registry,
    )
    batch_token_count = Histogram(
        "embedserve_batch_token_count",
        "Total post-truncation token count per executed inference batch.",
        buckets=(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
        registry=registry,
    )
    batch_flush_total = Counter(
        "embedserve_batch_flush_total",
        "Total number of batch flushes by flush trigger reason.",
        labelnames=("reason",),
        registry=registry,
    )
    batch_overload_rejections_total = Counter(
        "embedserve_batch_overload_rejections_total",
        "Total embed requests rejected because the batch queue is full.",
        registry=registry,
    )
    batch_request_timeouts_total = Counter(
        "embedserve_batch_request_timeouts_total",
        "Total embed requests that timed out while waiting for a batch result.",
        registry=registry,
    )
    batch_request_cancellations_total = Counter(
        "embedserve_batch_request_cancellations_total",
        "Total embed requests cancelled by clients before completion.",
        registry=registry,
    )
    batch_shutdown_rejections_total = Counter(
        "embedserve_batch_shutdown_rejections_total",
        "Total embed requests rejected due to service shutdown.",
        registry=registry,
    )
    batch_inference_failures_total = Counter(
        "embedserve_batch_inference_failures_total",
        "Total batch inference execution failures.",
        registry=registry,
    )
    gpu_memory_allocated_bytes = Gauge(
        "embedserve_gpu_memory_allocated_bytes",
        "Current CUDA memory allocated by the active runtime device in bytes.",
        labelnames=("device",),
        registry=registry,
    )
    gpu_memory_reserved_bytes = Gauge(
        "embedserve_gpu_memory_reserved_bytes",
        "Current CUDA memory reserved by the active runtime device in bytes.",
        labelnames=("device",),
        registry=registry,
    )
    gpu_oom_total = Counter(
        "embedserve_gpu_oom_total",
        "Total CUDA out-of-memory incidents seen during inference execution.",
        labelnames=("device",),
        registry=registry,
    )
    request_failures_total = Counter(
        "embedserve_request_failures_total",
        "Total request failures by operational reason.",
        labelnames=("reason",),
        registry=registry,
    )
    unhandled_exceptions_total = Counter(
        "embedserve_unhandled_exceptions_total",
        "Total unhandled request exceptions caught by top-level middleware.",
        registry=registry,
    )

    for reason in ("max_batch_size", "max_batch_tokens", "timeout", "shutdown"):
        batch_flush_total.labels(reason=reason)
    for reason in ("overload", "timeout", "internal_error", "shutdown"):
        request_failures_total.labels(reason=reason)

    return AppMetrics(
        registry=registry,
        http_requests_total=http_requests_total,
        http_request_duration_seconds=http_request_duration_seconds,
        app_ready=app_ready,
        batch_queue_depth=batch_queue_depth,
        batch_queue_wait_seconds=batch_queue_wait_seconds,
        batch_size=batch_size,
        batch_token_count=batch_token_count,
        batch_flush_total=batch_flush_total,
        batch_overload_rejections_total=batch_overload_rejections_total,
        batch_request_timeouts_total=batch_request_timeouts_total,
        batch_request_cancellations_total=batch_request_cancellations_total,
        batch_shutdown_rejections_total=batch_shutdown_rejections_total,
        batch_inference_failures_total=batch_inference_failures_total,
        gpu_memory_allocated_bytes=gpu_memory_allocated_bytes,
        gpu_memory_reserved_bytes=gpu_memory_reserved_bytes,
        gpu_oom_total=gpu_oom_total,
        request_failures_total=request_failures_total,
        unhandled_exceptions_total=unhandled_exceptions_total,
    )


def touch_http_metrics(
    metrics: AppMetrics,
    *,
    method: str,
    route: str,
    status_code: int,
) -> None:
    metrics.http_requests_total.labels(method=method, route=route, status_code=str(status_code))
    metrics.http_request_duration_seconds.labels(method=method, route=route)


def observe_http_request(
    metrics: AppMetrics,
    *,
    method: str,
    route: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    metrics.http_requests_total.labels(
        method=method,
        route=route,
        status_code=str(status_code),
    ).inc()
    metrics.http_request_duration_seconds.labels(method=method, route=route).observe(
        duration_seconds,
    )


def set_ready_state(metrics: AppMetrics, *, mode: str, ready: bool) -> None:
    metrics.app_ready.labels(mode=mode).set(1 if ready else 0)


def observe_request_failure(metrics: AppMetrics, *, reason: RequestFailureReason) -> None:
    metrics.request_failures_total.labels(reason=reason).inc()


def observe_unhandled_exception(metrics: AppMetrics) -> None:
    metrics.unhandled_exceptions_total.inc()


def observe_gpu_oom(metrics: AppMetrics, *, device: str) -> None:
    metrics.gpu_oom_total.labels(device=device).inc()


def refresh_runtime_metrics(metrics: AppMetrics, runtime: RuntimeState) -> None:
    embedder = runtime.embedder
    if embedder is None:
        return

    sample_device_memory = getattr(embedder, "sample_device_memory", None)
    if not callable(sample_device_memory):
        return

    snapshot = sample_device_memory()
    if snapshot is None:
        return

    metrics.gpu_memory_allocated_bytes.labels(device=runtime.device).set(snapshot.allocated_bytes)
    metrics.gpu_memory_reserved_bytes.labels(device=runtime.device).set(snapshot.reserved_bytes)


def render_metrics(metrics: AppMetrics) -> bytes:
    return generate_latest(metrics.registry)


__all__ = [
    "AppMetrics",
    "CONTENT_TYPE_LATEST",
    "create_metrics",
    "observe_gpu_oom",
    "observe_http_request",
    "observe_request_failure",
    "observe_unhandled_exception",
    "refresh_runtime_metrics",
    "render_metrics",
    "set_ready_state",
    "touch_http_metrics",
]
