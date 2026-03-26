from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import httpx

_ALLOWED_503_DETAILS = {
    "Batch queue is full": "overload",
    "Embedding request timed out": "timeout",
    "Service is shutting down": "shutdown",
}
_INTERNAL_ERROR_DETAIL = "Embedding inference failed"


@dataclass(frozen=True)
class LoadTestConfig:
    embed_url: str
    metrics_url: str
    ready_url: str
    duration_seconds: float
    concurrency: int
    warmup_requests: int
    inputs_per_request: int
    input_token_count: int
    timeout_seconds: float
    metrics_poll_interval_seconds: float
    max_vram_drift_bytes: int
    hardware_id: str
    model_id: str
    model_revision: str
    max_batch_size: int
    max_batch_tokens: int
    batch_timeout_ms: int
    max_batch_queue_size: int
    batch_request_timeout_ms: int


@dataclass(frozen=True)
class RequestResult:
    status_code: int
    category: str
    valid: bool


@dataclass(frozen=True)
class MetricsSnapshot:
    elapsed_seconds: float
    request_failures_total: dict[str, float]
    unhandled_exceptions_total: float
    gpu_oom_total: float
    gpu_memory_allocated_bytes: float | None
    gpu_memory_reserved_bytes: float | None
    process_resident_memory_bytes: float | None


@dataclass(frozen=True)
class NumericSeriesSummary:
    start: float
    min: float
    max: float
    end: float
    end_minus_start: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class AggregatedResults:
    total_requests: int = 0
    status_counts: dict[str, int] = field(
        default_factory=lambda: {"200": 0, "500": 0, "503": 0, "other": 0}
    )
    category_counts: dict[str, int] = field(
        default_factory=lambda: {
            "success": 0,
            "overload": 0,
            "timeout": 0,
            "shutdown": 0,
            "internal_error": 0,
            "transport_error": 0,
            "unexpected_status": 0,
        }
    )
    invalid_responses: int = 0

    def record(self, result: RequestResult) -> None:
        self.total_requests += 1
        if result.status_code == 200:
            self.status_counts["200"] += 1
        elif result.status_code == 500:
            self.status_counts["500"] += 1
        elif result.status_code == 503:
            self.status_counts["503"] += 1
        else:
            self.status_counts["other"] += 1

        self.category_counts[result.category] = self.category_counts.get(result.category, 0) + 1
        if not result.valid:
            self.invalid_responses += 1


@dataclass(frozen=True)
class LoadTestResult:
    passed: bool
    exit_code: int
    config: LoadTestConfig
    request_summary: AggregatedResults
    metrics_deltas: dict[str, float]
    gpu_allocated_summary: NumericSeriesSummary | None
    gpu_reserved_summary: NumericSeriesSummary | None
    process_rss_summary: NumericSeriesSummary | None
    gpu_metrics_available: bool
    checks: dict[str, bool]
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "exit_code": self.exit_code,
            "config": asdict(self.config),
            "request_summary": {
                "total_requests": self.request_summary.total_requests,
                "status_counts": self.request_summary.status_counts,
                "category_counts": self.request_summary.category_counts,
                "invalid_responses": self.request_summary.invalid_responses,
            },
            "metrics_deltas": self.metrics_deltas,
            "gpu_metrics_available": self.gpu_metrics_available,
            "gpu_memory_allocated_bytes": (
                None if self.gpu_allocated_summary is None else self.gpu_allocated_summary.to_dict()
            ),
            "gpu_memory_reserved_bytes": (
                None if self.gpu_reserved_summary is None else self.gpu_reserved_summary.to_dict()
            ),
            "process_resident_memory_bytes": (
                None if self.process_rss_summary is None else self.process_rss_summary.to_dict()
            ),
            "checks": self.checks,
            "failure_reasons": self.failure_reasons,
        }


class LoadTestOperationalError(Exception):
    pass


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        msg = "value must be >= 1"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        msg = "value must be > 0"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sustained load against a live EmbedServe instance",
    )
    parser.add_argument("--embed-url", default="http://127.0.0.1:8000/embed")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--ready-url", default="http://127.0.0.1:8000/readyz")
    parser.add_argument("--duration-seconds", type=_positive_float, default=3600.0)
    parser.add_argument("--concurrency", type=_positive_int, default=32)
    parser.add_argument("--warmup-requests", type=_positive_int, default=64)
    parser.add_argument("--inputs-per-request", type=_positive_int, default=2)
    parser.add_argument("--input-token-count", type=_positive_int, default=24)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--metrics-poll-interval-seconds", type=_positive_float, default=5.0)
    parser.add_argument("--max-vram-drift-bytes", type=_positive_int, default=268435456)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-batch-size", type=_positive_int, default=128)
    parser.add_argument("--max-batch-tokens", type=_positive_int, default=8192)
    parser.add_argument("--batch-timeout-ms", type=_positive_int, default=2)
    parser.add_argument("--max-batch-queue-size", type=_positive_int, default=1024)
    parser.add_argument("--batch-request-timeout-ms", type=_positive_int, default=5000)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> LoadTestConfig:
    return LoadTestConfig(
        embed_url=args.embed_url,
        metrics_url=args.metrics_url,
        ready_url=args.ready_url,
        duration_seconds=args.duration_seconds,
        concurrency=args.concurrency,
        warmup_requests=args.warmup_requests,
        inputs_per_request=args.inputs_per_request,
        input_token_count=args.input_token_count,
        timeout_seconds=args.timeout_seconds,
        metrics_poll_interval_seconds=args.metrics_poll_interval_seconds,
        max_vram_drift_bytes=args.max_vram_drift_bytes,
        hardware_id=args.hardware_id,
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        batch_timeout_ms=args.batch_timeout_ms,
        max_batch_queue_size=args.max_batch_queue_size,
        batch_request_timeout_ms=args.batch_request_timeout_ms,
    )


def _build_inputs(request_index: int, config: LoadTestConfig) -> list[str]:
    token = f"tok{request_index}"
    phrase = " ".join([token] * config.input_token_count)
    return [f"{phrase} sample_{index}" for index in range(config.inputs_per_request)]


def _validate_success_payload(payload: Any, expected_items: int) -> bool:
    if not isinstance(payload, dict):
        return False

    data = payload.get("data")
    usage = payload.get("usage")
    if not isinstance(data, list) or len(data) != expected_items:
        return False
    if not isinstance(usage, dict) or not isinstance(usage.get("tokens"), int):
        return False

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            return False
        if item.get("index") != index:
            return False
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            return False
        if any(
            not isinstance(value, int | float) or isinstance(value, bool) for value in embedding
        ):
            return False
    return True


def _extract_metric_value(metrics_text: str, metric_name: str, labels: str | None = None) -> float:
    label_pattern = "" if labels is None else rf"\{{{labels}\}}"
    pattern = rf"^{re.escape(metric_name)}{label_pattern}\s+([-+0-9.eE]+)$"
    match = re.search(pattern, metrics_text, flags=re.MULTILINE)
    if match is None:
        return 0.0
    return float(match.group(1))


def _extract_labeled_metric_values(
    metrics_text: str, metric_name: str
) -> list[tuple[dict[str, str], float]]:
    pattern = rf"^{re.escape(metric_name)}(?:\{{(?P<labels>[^}}]+)\}})?\s+(?P<value>[-+0-9.eE]+)$"
    matches = re.finditer(pattern, metrics_text, flags=re.MULTILINE)
    values: list[tuple[dict[str, str], float]] = []
    for match in matches:
        labels_text = match.group("labels")
        labels: dict[str, str] = {}
        if labels_text:
            for fragment in labels_text.split(","):
                key, value = fragment.split("=", maxsplit=1)
                labels[key] = value.strip('"')
        values.append((labels, float(match.group("value"))))
    return values


def _parse_metrics_snapshot(metrics_text: str, *, elapsed_seconds: float) -> MetricsSnapshot:
    request_failures_total = {
        reason: _extract_metric_value(
            metrics_text,
            "embedserve_request_failures_total",
            labels=rf'reason="{reason}"',
        )
        for reason in ("overload", "timeout", "internal_error", "shutdown")
    }
    gpu_oom_total = sum(
        value
        for _, value in _extract_labeled_metric_values(metrics_text, "embedserve_gpu_oom_total")
    )
    allocated_samples = _extract_labeled_metric_values(
        metrics_text,
        "embedserve_gpu_memory_allocated_bytes",
    )
    reserved_samples = _extract_labeled_metric_values(
        metrics_text,
        "embedserve_gpu_memory_reserved_bytes",
    )
    allocated = allocated_samples[0][1] if allocated_samples else None
    reserved = reserved_samples[0][1] if reserved_samples else None

    return MetricsSnapshot(
        elapsed_seconds=elapsed_seconds,
        request_failures_total=request_failures_total,
        unhandled_exceptions_total=_extract_metric_value(
            metrics_text,
            "embedserve_unhandled_exceptions_total",
        ),
        gpu_oom_total=gpu_oom_total,
        gpu_memory_allocated_bytes=allocated,
        gpu_memory_reserved_bytes=reserved,
        process_resident_memory_bytes=_extract_metric_value(
            metrics_text,
            "process_resident_memory_bytes",
        )
        or None,
    )


def _summarize_series(values: list[float]) -> NumericSeriesSummary | None:
    if not values:
        return None
    return NumericSeriesSummary(
        start=values[0],
        min=min(values),
        max=max(values),
        end=values[-1],
        end_minus_start=values[-1] - values[0],
    )


def _delta_counter(final_value: float, baseline_value: float) -> float:
    return max(final_value - baseline_value, 0.0)


def _evaluate_checks(
    request_summary: AggregatedResults,
    metrics_deltas: dict[str, float],
    *,
    max_vram_drift_bytes: int,
    gpu_allocated_summary: NumericSeriesSummary | None,
    gpu_reserved_summary: NumericSeriesSummary | None,
) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "all_responses_valid": request_summary.invalid_responses == 0,
        "no_transport_errors": request_summary.category_counts["transport_error"] == 0,
        "no_internal_errors": (
            request_summary.category_counts["internal_error"] == 0
            and metrics_deltas["internal_error"] == 0.0
        ),
        "no_unhandled_exceptions": metrics_deltas["unhandled_exceptions"] == 0.0,
        "gpu_oom_count_zero": metrics_deltas["gpu_oom"] == 0.0,
        "gpu_allocated_drift_within_threshold": (
            gpu_allocated_summary is None
            or gpu_allocated_summary.end_minus_start <= max_vram_drift_bytes
        ),
        "gpu_reserved_drift_within_threshold": (
            gpu_reserved_summary is None
            or gpu_reserved_summary.end_minus_start <= max_vram_drift_bytes
        ),
    }

    failure_reasons: list[str] = []
    if not checks["all_responses_valid"]:
        failure_reasons.append("invalid_or_unexpected_responses_detected")
    if not checks["no_transport_errors"]:
        failure_reasons.append("transport_errors_detected")
    if not checks["no_internal_errors"]:
        failure_reasons.append("internal_errors_detected")
    if not checks["no_unhandled_exceptions"]:
        failure_reasons.append("unhandled_exceptions_detected")
    if not checks["gpu_oom_count_zero"]:
        failure_reasons.append("gpu_oom_detected")
    if not checks["gpu_allocated_drift_within_threshold"]:
        failure_reasons.append("gpu_allocated_memory_drift_exceeded_threshold")
    if not checks["gpu_reserved_drift_within_threshold"]:
        failure_reasons.append("gpu_reserved_memory_drift_exceeded_threshold")
    return checks, failure_reasons


async def _ensure_ready(client: httpx.AsyncClient, config: LoadTestConfig) -> None:
    try:
        response = await client.get(config.ready_url, timeout=config.timeout_seconds)
    except httpx.HTTPError as exc:
        msg = f"ready_check_failed: {exc}"
        raise LoadTestOperationalError(msg) from exc

    if response.status_code != 200:
        msg = f"ready_check_failed: received HTTP {response.status_code}"
        raise LoadTestOperationalError(msg)


async def _fetch_metrics(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    *,
    elapsed_seconds: float,
) -> MetricsSnapshot:
    try:
        response = await client.get(config.metrics_url, timeout=config.timeout_seconds)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        msg = f"metrics_request_failed: {exc}"
        raise LoadTestOperationalError(msg) from exc

    return _parse_metrics_snapshot(response.text, elapsed_seconds=elapsed_seconds)


async def _execute_request(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    *,
    request_index: int,
) -> RequestResult:
    try:
        response = await client.post(
            config.embed_url,
            json={"inputs": _build_inputs(request_index, config)},
            timeout=config.timeout_seconds,
        )
    except httpx.HTTPError:
        return RequestResult(status_code=0, category="transport_error", valid=False)

    payload: Any
    detail: str | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        maybe_detail = payload.get("detail")
        if isinstance(maybe_detail, str):
            detail = maybe_detail

    if response.status_code == 200:
        return RequestResult(
            status_code=200,
            category="success",
            valid=_validate_success_payload(payload, expected_items=config.inputs_per_request),
        )
    if response.status_code == 500:
        return RequestResult(
            status_code=500,
            category="internal_error",
            valid=detail == _INTERNAL_ERROR_DETAIL,
        )
    if response.status_code == 503:
        category = None if detail is None else _ALLOWED_503_DETAILS.get(detail)
        if category is None:
            return RequestResult(status_code=503, category="unexpected_status", valid=False)
        return RequestResult(status_code=503, category=category, valid=True)
    return RequestResult(
        status_code=response.status_code, category="unexpected_status", valid=False
    )


def _build_client_limits(concurrency: int) -> httpx.Limits:
    # Reserve one connection for background metrics polling while workers saturate `/embed`.
    return httpx.Limits(
        max_keepalive_connections=concurrency + 1,
        max_connections=concurrency + 1,
    )


async def run_load_test(config: LoadTestConfig) -> LoadTestResult:
    limits = _build_client_limits(config.concurrency)
    start_time = perf_counter()
    request_summary = AggregatedResults()

    async with httpx.AsyncClient(limits=limits) as client:
        await _ensure_ready(client, config)

        for warmup_index in range(config.warmup_requests):
            warmup_result = await _execute_request(
                client,
                config,
                request_index=warmup_index,
            )
            if not warmup_result.valid or warmup_result.category != "success":
                msg = f"warmup_failed: category={warmup_result.category} status={warmup_result.status_code}"
                raise LoadTestOperationalError(msg)

        baseline_snapshot = await _fetch_metrics(client, config, elapsed_seconds=0.0)
        metric_samples: list[MetricsSnapshot] = [baseline_snapshot]
        metrics_error: str | None = None
        stop_polling = asyncio.Event()

        async def poll_metrics() -> None:
            nonlocal metrics_error
            while not stop_polling.is_set():
                await asyncio.sleep(config.metrics_poll_interval_seconds)
                if stop_polling.is_set():
                    break
                try:
                    metric_samples.append(
                        await _fetch_metrics(
                            client,
                            config,
                            elapsed_seconds=perf_counter() - start_time,
                        )
                    )
                except LoadTestOperationalError as exc:
                    metrics_error = str(exc)
                    stop_polling.set()
                    return

        next_request_index = config.warmup_requests
        deadline = perf_counter() + config.duration_seconds

        async def worker() -> None:
            nonlocal next_request_index
            while perf_counter() < deadline and not stop_polling.is_set():
                request_index = next_request_index
                next_request_index += 1
                result = await _execute_request(
                    client,
                    config,
                    request_index=request_index,
                )
                request_summary.record(result)

        poller_task = asyncio.create_task(poll_metrics())
        worker_tasks = [asyncio.create_task(worker()) for _ in range(config.concurrency)]
        try:
            await asyncio.gather(*worker_tasks)
        finally:
            stop_polling.set()
            await poller_task

        if metrics_error is not None:
            raise LoadTestOperationalError(metrics_error)

        metric_samples.append(
            await _fetch_metrics(
                client,
                config,
                elapsed_seconds=perf_counter() - start_time,
            )
        )

    final_snapshot = metric_samples[-1]
    metrics_deltas = {
        reason: _delta_counter(
            final_snapshot.request_failures_total[reason],
            baseline_snapshot.request_failures_total[reason],
        )
        for reason in ("overload", "timeout", "internal_error", "shutdown")
    }
    metrics_deltas["unhandled_exceptions"] = _delta_counter(
        final_snapshot.unhandled_exceptions_total,
        baseline_snapshot.unhandled_exceptions_total,
    )
    metrics_deltas["gpu_oom"] = _delta_counter(
        final_snapshot.gpu_oom_total,
        baseline_snapshot.gpu_oom_total,
    )

    allocated_values = [
        sample.gpu_memory_allocated_bytes
        for sample in metric_samples
        if sample.gpu_memory_allocated_bytes is not None
    ]
    reserved_values = [
        sample.gpu_memory_reserved_bytes
        for sample in metric_samples
        if sample.gpu_memory_reserved_bytes is not None
    ]
    process_rss_values = [
        sample.process_resident_memory_bytes
        for sample in metric_samples
        if sample.process_resident_memory_bytes is not None
    ]

    gpu_allocated_summary = _summarize_series(allocated_values)
    gpu_reserved_summary = _summarize_series(reserved_values)
    process_rss_summary = _summarize_series(process_rss_values)
    checks, failure_reasons = _evaluate_checks(
        request_summary,
        metrics_deltas,
        max_vram_drift_bytes=config.max_vram_drift_bytes,
        gpu_allocated_summary=gpu_allocated_summary,
        gpu_reserved_summary=gpu_reserved_summary,
    )

    passed = all(checks.values())
    return LoadTestResult(
        passed=passed,
        exit_code=0 if passed else 1,
        config=config,
        request_summary=request_summary,
        metrics_deltas=metrics_deltas,
        gpu_allocated_summary=gpu_allocated_summary,
        gpu_reserved_summary=gpu_reserved_summary,
        process_rss_summary=process_rss_summary,
        gpu_metrics_available=bool(allocated_values and reserved_values),
        checks=checks,
        failure_reasons=failure_reasons,
    )


def _render_human_readable(result: LoadTestResult) -> str:
    lines = [
        f"pass: {result.passed}",
        f"duration_seconds: {result.config.duration_seconds}",
        f"concurrency: {result.config.concurrency}",
        f"request_volume: {result.request_summary.total_requests}",
        f"status_counts: {result.request_summary.status_counts}",
        f"category_counts: {result.request_summary.category_counts}",
        f"invalid_responses: {result.request_summary.invalid_responses}",
        f"metrics_deltas: {result.metrics_deltas}",
        f"gpu_metrics_available: {result.gpu_metrics_available}",
    ]
    if result.gpu_allocated_summary is not None:
        lines.append(f"gpu_memory_allocated_bytes: {result.gpu_allocated_summary.to_dict()}")
    if result.gpu_reserved_summary is not None:
        lines.append(f"gpu_memory_reserved_bytes: {result.gpu_reserved_summary.to_dict()}")
    if result.process_rss_summary is not None:
        lines.append(f"process_resident_memory_bytes: {result.process_rss_summary.to_dict()}")
    if result.failure_reasons:
        lines.append(f"failure_reasons: {result.failure_reasons}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _build_config(args)

    try:
        result = asyncio.run(run_load_test(config))
    except LoadTestOperationalError as exc:
        payload = {
            "pass": False,
            "exit_code": 2,
            "failure_reasons": [str(exc)],
        }
        if args.json_output:
            print(json.dumps(payload))
        else:
            print(str(exc))
        return 2

    if args.json_output:
        print(json.dumps(result.to_dict()))
    else:
        print(_render_human_readable(result))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
