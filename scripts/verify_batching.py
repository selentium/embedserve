from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

_ALLOWED_503_DETAILS = {
    "Batch queue is full",
    "Embedding request timed out",
    "Service is shutting down",
}


@dataclass(frozen=True)
class HarnessProfile:
    embed_url: str
    metrics_url: str
    hardware_id: str
    model_id: str
    model_revision: str
    concurrency: int
    total_requests: int
    warmup_requests: int
    inputs_per_request: int
    input_token_count: int
    timeout_seconds: float
    max_batch_size: int
    max_batch_tokens: int
    batch_timeout_ms: int
    max_batch_queue_size: int
    batch_request_timeout_ms: int


@dataclass(frozen=True)
class RequestResult:
    status_code: int
    detail: str | None
    valid: bool


@dataclass(frozen=True)
class MetricsSnapshot:
    batch_size_count: float
    batch_size_sum: float
    batch_token_count_count: float
    flush_total: float

    @property
    def non_singleton_batches(self) -> float:
        return max(self.batch_size_sum - self.batch_size_count, 0.0)

    def delta_from(self, baseline: MetricsSnapshot) -> MetricsSnapshot:
        return MetricsSnapshot(
            batch_size_count=self.batch_size_count - baseline.batch_size_count,
            batch_size_sum=self.batch_size_sum - baseline.batch_size_sum,
            batch_token_count_count=(
                self.batch_token_count_count - baseline.batch_token_count_count
            ),
            flush_total=self.flush_total - baseline.flush_total,
        )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify dynamic batching behavior against a live server"
    )
    parser.add_argument("--embed-url", default="http://127.0.0.1:8000/embed")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--concurrency", type=_positive_int, default=32)
    parser.add_argument("--total-requests", type=_positive_int, default=256)
    parser.add_argument("--warmup-requests", type=_positive_int, default=32)
    parser.add_argument("--inputs-per-request", type=_positive_int, default=2)
    parser.add_argument("--input-token-count", type=_positive_int, default=24)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--max-batch-size", type=_positive_int, default=128)
    parser.add_argument("--max-batch-tokens", type=_positive_int, default=8192)
    parser.add_argument("--batch-timeout-ms", type=_positive_int, default=2)
    parser.add_argument("--max-batch-queue-size", type=_positive_int, default=1024)
    parser.add_argument("--batch-request-timeout-ms", type=_positive_int, default=5000)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def _extract_metric_value(metrics_text: str, metric_name: str, labels: str | None = None) -> float:
    label_pattern = "" if labels is None else rf"\{{{labels}\}}"
    pattern = rf"^{re.escape(metric_name)}{label_pattern}\s+([-+0-9.eE]+)$"
    match = re.search(pattern, metrics_text, flags=re.MULTILINE)
    if match is None:
        return 0.0
    return float(match.group(1))


def _extract_flush_total(metrics_text: str) -> float:
    total = 0.0
    for reason in ("max_batch_size", "max_batch_tokens", "timeout", "shutdown"):
        total += _extract_metric_value(
            metrics_text,
            "embedserve_batch_flush_total",
            labels=rf'reason="{reason}"',
        )
    return total


def _parse_metrics(metrics_text: str) -> MetricsSnapshot:
    return MetricsSnapshot(
        batch_size_count=_extract_metric_value(metrics_text, "embedserve_batch_size_count"),
        batch_size_sum=_extract_metric_value(metrics_text, "embedserve_batch_size_sum"),
        batch_token_count_count=_extract_metric_value(
            metrics_text,
            "embedserve_batch_token_count_count",
        ),
        flush_total=_extract_flush_total(metrics_text),
    )


def _build_inputs(request_index: int, profile: HarnessProfile) -> list[str]:
    token = f"tok{request_index}"
    phrase = " ".join([token] * profile.input_token_count)
    return [f"{phrase} sample_{inner_index}" for inner_index in range(profile.inputs_per_request)]


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


async def _execute_request(
    *,
    client: httpx.AsyncClient,
    profile: HarnessProfile,
    request_index: int,
) -> RequestResult:
    inputs = _build_inputs(request_index, profile)
    try:
        response = await client.post(
            profile.embed_url,
            json={"inputs": inputs},
            timeout=profile.timeout_seconds,
        )
    except httpx.HTTPError:
        return RequestResult(status_code=0, detail="transport_error", valid=False)

    detail: str | None = None
    payload: Any = None
    try:
        payload = response.json()
        if isinstance(payload, dict):
            maybe_detail = payload.get("detail")
            if isinstance(maybe_detail, str):
                detail = maybe_detail
    except ValueError:
        payload = None

    if response.status_code == 200:
        return RequestResult(
            status_code=200,
            detail=None,
            valid=_validate_success_payload(payload, expected_items=profile.inputs_per_request),
        )
    if response.status_code == 503:
        return RequestResult(
            status_code=503,
            detail=detail,
            valid=detail in _ALLOWED_503_DETAILS,
        )
    return RequestResult(status_code=response.status_code, detail=detail, valid=False)


async def _run_load(profile: HarnessProfile) -> tuple[list[RequestResult], MetricsSnapshot]:
    limits = httpx.Limits(
        max_keepalive_connections=profile.concurrency, max_connections=profile.concurrency
    )
    semaphore = asyncio.Semaphore(profile.concurrency)

    async with httpx.AsyncClient(limits=limits) as client:
        baseline_response = await client.get(profile.metrics_url, timeout=profile.timeout_seconds)
        baseline_response.raise_for_status()
        baseline_metrics = _parse_metrics(baseline_response.text)

        for warmup_index in range(profile.warmup_requests):
            await _execute_request(
                client=client,
                profile=profile,
                request_index=warmup_index,
            )

        async def wrapped_request(request_index: int) -> RequestResult:
            async with semaphore:
                return await _execute_request(
                    client=client,
                    profile=profile,
                    request_index=request_index + profile.warmup_requests,
                )

        tasks = [
            asyncio.create_task(wrapped_request(index)) for index in range(profile.total_requests)
        ]
        results = await asyncio.gather(*tasks)
        metrics_response = await client.get(profile.metrics_url, timeout=profile.timeout_seconds)
        metrics_response.raise_for_status()

    return results, _parse_metrics(metrics_response.text).delta_from(baseline_metrics)


def _build_profile(args: argparse.Namespace) -> HarnessProfile:
    return HarnessProfile(
        embed_url=args.embed_url,
        metrics_url=args.metrics_url,
        hardware_id=args.hardware_id,
        model_id=args.model_id,
        model_revision=args.model_revision,
        concurrency=args.concurrency,
        total_requests=args.total_requests,
        warmup_requests=args.warmup_requests,
        inputs_per_request=args.inputs_per_request,
        input_token_count=args.input_token_count,
        timeout_seconds=args.timeout_seconds,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        batch_timeout_ms=args.batch_timeout_ms,
        max_batch_queue_size=args.max_batch_queue_size,
        batch_request_timeout_ms=args.batch_request_timeout_ms,
    )


def main() -> int:
    args = parse_args()
    profile = _build_profile(args)

    try:
        request_results, metrics = asyncio.run(_run_load(profile))
    except httpx.HTTPError as exc:
        payload = {"pass": False, "error": f"metrics_request_failed: {exc}"}
        if args.json_output:
            print(json.dumps(payload))
        else:
            print(payload["error"])
        return 2

    status_counts: dict[str, int] = {"200": 0, "503": 0, "other": 0}
    invalid_responses = 0
    unexpected_503_details: set[str] = set()
    for result in request_results:
        if result.status_code == 200:
            status_counts["200"] += 1
        elif result.status_code == 503:
            status_counts["503"] += 1
            if result.detail not in _ALLOWED_503_DETAILS:
                unexpected_503_details.add(str(result.detail))
        else:
            status_counts["other"] += 1
        if not result.valid:
            invalid_responses += 1

    checks = {
        "all_requests_valid": invalid_responses == 0,
        "non_singleton_batch_seen": metrics.non_singleton_batches >= 1,
        "batch_size_metric_non_zero": metrics.batch_size_count > 0,
        "batch_token_metric_non_zero": metrics.batch_token_count_count > 0,
        "batch_flush_metric_non_zero": metrics.flush_total > 0,
    }
    passed = all(checks.values())

    summary: dict[str, Any] = {
        "pass": passed,
        "profile": {
            "hardware_id": profile.hardware_id,
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "concurrency": profile.concurrency,
            "total_requests": profile.total_requests,
            "warmup_requests": profile.warmup_requests,
            "inputs_per_request": profile.inputs_per_request,
            "input_token_count": profile.input_token_count,
            "batching_settings": {
                "max_batch_size": profile.max_batch_size,
                "max_batch_tokens": profile.max_batch_tokens,
                "batch_timeout_ms": profile.batch_timeout_ms,
                "max_batch_queue_size": profile.max_batch_queue_size,
                "batch_request_timeout_ms": profile.batch_request_timeout_ms,
            },
        },
        "results": {
            "status_counts": status_counts,
            "invalid_responses": invalid_responses,
            "unexpected_503_details": sorted(unexpected_503_details),
        },
        "metrics": {
            "batch_size_count": metrics.batch_size_count,
            "batch_size_sum": metrics.batch_size_sum,
            "batch_token_count_count": metrics.batch_token_count_count,
            "batch_flush_total": metrics.flush_total,
            "non_singleton_batches_estimate": metrics.non_singleton_batches,
        },
        "checks": checks,
    }

    if args.json_output:
        print(json.dumps(summary))
    else:
        print(f"pass: {summary['pass']}")
        print(f"status_counts: {status_counts}")
        print(f"invalid_responses: {invalid_responses}")
        print(f"non_singleton_batches_estimate: {metrics.non_singleton_batches}")
        print(f"batch_size_count: {metrics.batch_size_count}")
        print(f"batch_token_count_count: {metrics.batch_token_count_count}")
        print(f"batch_flush_total: {metrics.flush_total}")
        if unexpected_503_details:
            print(f"unexpected_503_details: {sorted(unexpected_503_details)}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
