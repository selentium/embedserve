from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class AppMetrics:
    registry: CollectorRegistry
    http_requests_total: Counter
    http_request_duration_seconds: Histogram
    app_ready: Gauge


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

    return AppMetrics(
        registry=registry,
        http_requests_total=http_requests_total,
        http_request_duration_seconds=http_request_duration_seconds,
        app_ready=app_ready,
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


def mark_ready(metrics: AppMetrics, *, mode: str) -> None:
    metrics.app_ready.labels(mode=mode).set(1)


def render_metrics(metrics: AppMetrics) -> bytes:
    return generate_latest(metrics.registry)


__all__ = [
    "AppMetrics",
    "CONTENT_TYPE_LATEST",
    "create_metrics",
    "mark_ready",
    "observe_http_request",
    "render_metrics",
    "touch_http_metrics",
]
