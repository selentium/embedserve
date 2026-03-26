from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from transformers import AutoTokenizer

DEFAULT_EMBED_URL = "http://127.0.0.1:8000/embed"
DEFAULT_HARDWARE_ID = "local-dev"
DEFAULT_LABEL = "batched"
DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "826711e54e001c83835913827a843d8dd0a1def9"


class BenchmarkOperationalError(Exception):
    pass


@dataclass(frozen=True)
class BenchmarkProfile:
    embed_url: str
    hardware_id: str
    label: str
    concurrency: int
    total_texts: int
    warmup_requests: int
    inputs_per_request: int
    input_token_count: int
    timeout_seconds: float
    model_id: str
    model_revision: str
    device: str
    dtype: str
    max_length: int
    truncate: bool
    max_batch_size: int
    max_batch_tokens: int
    batch_timeout_ms: int
    max_batch_queue_size: int
    batch_request_timeout_ms: int
    warm_cache: bool
    warm_model: bool
    warm_container: bool

    @property
    def total_requests(self) -> int:
        return self.total_texts // self.inputs_per_request


@dataclass(frozen=True)
class RequestOutcome:
    status_code: int | None
    success: bool
    latency_ms: float | None
    timed_out: bool = False
    transport_error: bool = False
    invalid_response: bool = False
    failure_detail: str | None = None


@dataclass(frozen=True)
class BenchmarkResults:
    duration_seconds: float
    successful_requests: int
    successful_texts: int
    requests_per_second: float
    texts_per_second: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    error_count: int
    timeout_count: int
    status_counts: dict[str, int]
    transport_error_count: int
    invalid_response_count: int


@dataclass(frozen=True)
class BenchmarkResult:
    exit_code: int
    profile: BenchmarkProfile
    results: BenchmarkResults | None
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "exit_code": self.exit_code,
            "profile": {
                "embed_url": self.profile.embed_url,
                "hardware_id": self.profile.hardware_id,
                "label": self.profile.label,
                "concurrency": self.profile.concurrency,
                "total_texts": self.profile.total_texts,
                "total_requests": self.profile.total_requests,
                "warmup_requests": self.profile.warmup_requests,
                "inputs_per_request": self.profile.inputs_per_request,
                "input_token_count": self.profile.input_token_count,
                "timeout_seconds": self.profile.timeout_seconds,
                "model_id": self.profile.model_id,
                "model_revision": self.profile.model_revision,
                "device": self.profile.device,
                "dtype": self.profile.dtype,
                "tokenization": {
                    "max_length": self.profile.max_length,
                    "truncate": self.profile.truncate,
                },
                "batching": {
                    "max_batch_size": self.profile.max_batch_size,
                    "max_batch_tokens": self.profile.max_batch_tokens,
                    "batch_timeout_ms": self.profile.batch_timeout_ms,
                    "max_batch_queue_size": self.profile.max_batch_queue_size,
                    "batch_request_timeout_ms": self.profile.batch_request_timeout_ms,
                },
                "warm_state": {
                    "cache": self.profile.warm_cache,
                    "model": self.profile.warm_model,
                    "container": self.profile.warm_container,
                },
            },
            "failure_reason": self.failure_reason,
        }
        if self.results is None:
            payload["results"] = None
        else:
            payload["results"] = {
                "duration_seconds": self.results.duration_seconds,
                "successful_requests": self.results.successful_requests,
                "successful_texts": self.results.successful_texts,
                "requests_per_second": self.results.requests_per_second,
                "texts_per_second": self.results.texts_per_second,
                "latency_ms": {
                    "p50": self.results.latency_p50_ms,
                    "p95": self.results.latency_p95_ms,
                    "p99": self.results.latency_p99_ms,
                },
                "error_count": self.results.error_count,
                "timeout_count": self.results.timeout_count,
                "status_counts": self.results.status_counts,
                "transport_error_count": self.results.transport_error_count,
                "invalid_response_count": self.results.invalid_response_count,
            }
        return payload


class _ResponseLike(Protocol):
    status_code: int

    def json(self) -> Any: ...


class _HttpClientLike(Protocol):
    async def get(self, url: str, *, timeout: float) -> _ResponseLike: ...

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> _ResponseLike: ...


@dataclass(frozen=True)
class SyntheticInputFactory:
    stable_tokens: tuple[str, ...]
    input_token_count: int
    inputs_per_request: int

    def build_inputs(self, *, request_index: int) -> list[str]:
        inputs: list[str] = []
        token_count = len(self.stable_tokens)

        for item_index in range(self.inputs_per_request):
            offset = (request_index * self.inputs_per_request + item_index) % token_count
            tokens = [
                self.stable_tokens[(offset + token_index) % token_count]
                for token_index in range(self.input_token_count)
            ]
            inputs.append(" ".join(tokens))

        return inputs


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        msg = "value must be >= 1"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        msg = "value must be >= 0"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        msg = "value must be > 0"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    msg = "value must be a boolean string"
    raise argparse.ArgumentTypeError(msg)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark /embed throughput and client-side latency against a live server"
    )
    parser.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
    parser.add_argument("--hardware-id", default=DEFAULT_HARDWARE_ID)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--concurrency", type=_positive_int, default=32)
    parser.add_argument("--total-texts", type=_positive_int, default=10000)
    parser.add_argument("--warmup-requests", type=_non_negative_int, default=200)
    parser.add_argument("--inputs-per-request", type=_positive_int, default=1)
    parser.add_argument("--input-token-count", type=_positive_int, default=24)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--truncate", type=_bool_arg, default=True)
    parser.add_argument("--max-batch-size", type=_positive_int, default=128)
    parser.add_argument("--max-batch-tokens", type=_positive_int, default=8192)
    parser.add_argument("--batch-timeout-ms", type=_positive_int, default=2)
    parser.add_argument("--max-batch-queue-size", type=_positive_int, default=1024)
    parser.add_argument("--batch-request-timeout-ms", type=_positive_int, default=5000)
    parser.add_argument("--warm-cache", type=_bool_arg, default=True)
    parser.add_argument("--warm-model", type=_bool_arg, default=True)
    parser.add_argument("--warm-container", type=_bool_arg, default=True)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _build_profile(args: argparse.Namespace) -> BenchmarkProfile:
    if args.total_texts % args.inputs_per_request != 0:
        msg = "total-texts must be divisible by inputs-per-request"
        raise ValueError(msg)

    return BenchmarkProfile(
        embed_url=args.embed_url,
        hardware_id=args.hardware_id,
        label=args.label,
        concurrency=args.concurrency,
        total_texts=args.total_texts,
        warmup_requests=args.warmup_requests,
        inputs_per_request=args.inputs_per_request,
        input_token_count=args.input_token_count,
        timeout_seconds=args.timeout_seconds,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        truncate=args.truncate,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        batch_timeout_ms=args.batch_timeout_ms,
        max_batch_queue_size=args.max_batch_queue_size,
        batch_request_timeout_ms=args.batch_request_timeout_ms,
        warm_cache=args.warm_cache,
        warm_model=args.warm_model,
        warm_container=args.warm_container,
    )


@lru_cache(maxsize=8)
def _load_input_tokenizer(model_id: str, revision: str) -> Any:
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def _derive_ready_url(embed_url: str) -> str:
    parsed = urlsplit(embed_url)
    segments = [segment for segment in parsed.path.split("/") if segment]

    if segments and segments[-1] == "embed":
        segments[-1] = "readyz"
    else:
        segments.append("readyz")

    ready_path = "/" + "/".join(segments) if segments else "/readyz"
    return urlunsplit((parsed.scheme, parsed.netloc, ready_path, "", ""))


def _stable_token_pool(tokenizer: Any) -> tuple[str, ...]:
    stable_tokens: list[str] = []

    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        if token.startswith("##"):
            continue

        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip()
        if not decoded.isalpha() or not decoded.islower():
            continue

        encoded = tokenizer(decoded, add_special_tokens=False)["input_ids"]
        if encoded == [token_id]:
            stable_tokens.append(decoded)

    if len(stable_tokens) < 8:
        msg = "tokenizer does not expose enough stable single-token text candidates"
        raise BenchmarkOperationalError(msg)

    return tuple(stable_tokens)


def _build_input_factory(profile: BenchmarkProfile) -> SyntheticInputFactory:
    try:
        tokenizer = _load_input_tokenizer(profile.model_id, profile.model_revision)
        stable_tokens = _stable_token_pool(tokenizer)
    except BenchmarkOperationalError:
        raise
    except Exception as exc:
        msg = f"failed to initialize tokenizer-stable inputs: {exc}"
        raise BenchmarkOperationalError(msg) from exc

    factory = SyntheticInputFactory(
        stable_tokens=stable_tokens,
        input_token_count=profile.input_token_count,
        inputs_per_request=profile.inputs_per_request,
    )

    probe_inputs = factory.build_inputs(request_index=0)
    for probe_input in probe_inputs:
        actual_token_count = len(tokenizer(probe_input, add_special_tokens=False)["input_ids"])
        if actual_token_count != profile.input_token_count:
            msg = (
                "tokenizer-stable input generation mismatch: "
                f"expected {profile.input_token_count}, got {actual_token_count}"
            )
            raise BenchmarkOperationalError(msg)

    return factory


def _parse_ready_response(
    payload: Any,
    *,
    profile: BenchmarkProfile,
) -> None:
    if not isinstance(payload, dict):
        msg = "readyz payload is not an object"
        raise BenchmarkOperationalError(msg)

    if payload.get("status") != "ready":
        msg = "readyz payload missing ready status"
        raise BenchmarkOperationalError(msg)

    tokenization = payload.get("tokenization")
    batching = payload.get("batching")
    if not isinstance(tokenization, dict):
        msg = "readyz payload missing tokenization settings"
        raise BenchmarkOperationalError(msg)
    if not isinstance(batching, dict):
        msg = "readyz payload missing batching settings"
        raise BenchmarkOperationalError(msg)

    checks: list[tuple[str, Any, Any]] = [
        ("model", payload.get("model"), profile.model_id),
        ("revision", payload.get("revision"), profile.model_revision),
        ("device", payload.get("device"), profile.device),
        ("dtype", payload.get("dtype"), profile.dtype),
        ("tokenization.max_length", tokenization.get("max_length"), profile.max_length),
        ("tokenization.truncate", tokenization.get("truncate"), profile.truncate),
        ("batching.max_batch_size", batching.get("max_batch_size"), profile.max_batch_size),
        ("batching.max_batch_tokens", batching.get("max_batch_tokens"), profile.max_batch_tokens),
        ("batching.batch_timeout_ms", batching.get("batch_timeout_ms"), profile.batch_timeout_ms),
        (
            "batching.max_batch_queue_size",
            batching.get("max_batch_queue_size"),
            profile.max_batch_queue_size,
        ),
        (
            "batching.batch_request_timeout_ms",
            batching.get("batch_request_timeout_ms"),
            profile.batch_request_timeout_ms,
        ),
    ]

    for field_name, actual, expected in checks:
        if actual != expected:
            msg = f"readyz {field_name} mismatch: expected {expected}, got {actual}"
            raise BenchmarkOperationalError(msg)


async def _verify_server_configuration(
    *,
    client: _HttpClientLike,
    profile: BenchmarkProfile,
) -> None:
    ready_url = _derive_ready_url(profile.embed_url)

    try:
        response = await client.get(ready_url, timeout=profile.timeout_seconds)
    except httpx.TimeoutException as exc:
        msg = f"readyz request timed out at {ready_url}"
        raise BenchmarkOperationalError(msg) from exc
    except httpx.HTTPError as exc:
        msg = f"readyz request failed at {ready_url}: {exc}"
        raise BenchmarkOperationalError(msg) from exc

    if response.status_code != 200:
        detail = f"HTTP {response.status_code}"
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            reason = payload.get("reason")
            message = payload.get("detail")
            if isinstance(reason, str) and isinstance(message, str):
                detail = f"{reason}: {message}"
            elif isinstance(message, str):
                detail = message

        msg = f"server readiness check failed at {ready_url}: {detail}"
        raise BenchmarkOperationalError(msg)

    try:
        payload = response.json()
    except ValueError as exc:
        msg = "readyz response body is not valid JSON"
        raise BenchmarkOperationalError(msg) from exc

    _parse_ready_response(payload, profile=profile)


def _parse_embed_response(
    payload: Any,
    *,
    expected_items: int,
    expected_model_id: str,
    expected_model_revision: str,
) -> None:
    if not isinstance(payload, dict):
        msg = "response payload is not an object"
        raise BenchmarkOperationalError(msg)

    dim = payload.get("dim")
    data = payload.get("data")
    usage = payload.get("usage")
    model = payload.get("model")
    revision = payload.get("revision")

    if not isinstance(model, str) or not model:
        msg = "response payload missing non-empty model"
        raise BenchmarkOperationalError(msg)
    if not isinstance(revision, str) or not revision:
        msg = "response payload missing non-empty revision"
        raise BenchmarkOperationalError(msg)
    if model != expected_model_id:
        msg = f"response model mismatch: expected {expected_model_id}, got {model}"
        raise BenchmarkOperationalError(msg)
    if revision != expected_model_revision:
        msg = f"response revision mismatch: expected {expected_model_revision}, got {revision}"
        raise BenchmarkOperationalError(msg)
    if not isinstance(dim, int) or dim < 1:
        msg = "response payload missing valid integer dim"
        raise BenchmarkOperationalError(msg)
    if not isinstance(data, list) or len(data) != expected_items:
        msg = f"response payload data length mismatch: expected {expected_items}"
        raise BenchmarkOperationalError(msg)
    if not isinstance(usage, dict) or not isinstance(usage.get("tokens"), int):
        msg = "response payload missing usage.tokens"
        raise BenchmarkOperationalError(msg)

    for expected_index, item in enumerate(data):
        if not isinstance(item, dict):
            msg = f"response data item at index {expected_index} is not an object"
            raise BenchmarkOperationalError(msg)
        if item.get("index") != expected_index:
            msg = f"response index mismatch at position {expected_index}"
            raise BenchmarkOperationalError(msg)

        embedding = item.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != dim:
            msg = f"response embedding width mismatch at index {expected_index}"
            raise BenchmarkOperationalError(msg)
        if any(
            isinstance(value, bool) or not isinstance(value, int | float) for value in embedding
        ):
            msg = f"embedding at index {expected_index} contains non-numeric values"
            raise BenchmarkOperationalError(msg)


async def _issue_request(
    *,
    client: _HttpClientLike,
    profile: BenchmarkProfile,
    request_index: int,
    input_factory: SyntheticInputFactory,
) -> RequestOutcome:
    inputs = input_factory.build_inputs(request_index=request_index)
    started_at = perf_counter()

    try:
        response = await client.post(
            profile.embed_url,
            json={"inputs": inputs},
            timeout=profile.timeout_seconds,
        )
    except httpx.TimeoutException:
        return RequestOutcome(
            status_code=None,
            success=False,
            latency_ms=None,
            timed_out=True,
        )
    except httpx.HTTPError:
        return RequestOutcome(
            status_code=None,
            success=False,
            latency_ms=None,
            transport_error=True,
        )

    if response.status_code != 200:
        return RequestOutcome(status_code=response.status_code, success=False, latency_ms=None)

    try:
        payload = response.json()
        _parse_embed_response(
            payload,
            expected_items=profile.inputs_per_request,
            expected_model_id=profile.model_id,
            expected_model_revision=profile.model_revision,
        )
    except (BenchmarkOperationalError, ValueError) as exc:
        return RequestOutcome(
            status_code=200,
            success=False,
            latency_ms=None,
            invalid_response=True,
            failure_detail=str(exc),
        )

    latency_ms = (perf_counter() - started_at) * 1000.0
    return RequestOutcome(status_code=200, success=True, latency_ms=latency_ms)


def _nearest_rank_percentile(values: list[float], percentile: int) -> float:
    if not values:
        msg = "cannot calculate percentile for empty values"
        raise BenchmarkOperationalError(msg)

    sorted_values = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(sorted_values)))
    return sorted_values[rank - 1]


def _summarize_outcomes(
    *,
    profile: BenchmarkProfile,
    outcomes: list[RequestOutcome],
    duration_seconds: float,
) -> BenchmarkResults:
    successful_latencies = [outcome.latency_ms for outcome in outcomes if outcome.success]
    latencies_ms = [latency for latency in successful_latencies if latency is not None]

    status_counts = {"200": 0, "503": 0, "other": 0}
    timeout_count = 0
    transport_error_count = 0
    invalid_response_count = 0

    for outcome in outcomes:
        if outcome.status_code == 200:
            status_counts["200"] += 1
        elif outcome.status_code == 503:
            status_counts["503"] += 1
        elif outcome.status_code is not None:
            status_counts["other"] += 1

        if outcome.timed_out:
            timeout_count += 1
        if outcome.transport_error:
            transport_error_count += 1
        if outcome.invalid_response:
            invalid_response_count += 1

    successful_requests = len(latencies_ms)
    successful_texts = successful_requests * profile.inputs_per_request
    error_count = (
        status_counts["503"]
        + status_counts["other"]
        + transport_error_count
        + invalid_response_count
    )

    return BenchmarkResults(
        duration_seconds=duration_seconds,
        successful_requests=successful_requests,
        successful_texts=successful_texts,
        requests_per_second=successful_requests / duration_seconds,
        texts_per_second=successful_texts / duration_seconds,
        latency_p50_ms=_nearest_rank_percentile(latencies_ms, 50) if latencies_ms else None,
        latency_p95_ms=_nearest_rank_percentile(latencies_ms, 95) if latencies_ms else None,
        latency_p99_ms=_nearest_rank_percentile(latencies_ms, 99) if latencies_ms else None,
        error_count=error_count,
        timeout_count=timeout_count,
        status_counts=status_counts,
        transport_error_count=transport_error_count,
        invalid_response_count=invalid_response_count,
    )


def _format_outcome(outcome: RequestOutcome) -> str:
    if outcome.failure_detail is not None:
        return outcome.failure_detail
    if outcome.timed_out:
        return "client timeout"
    if outcome.transport_error:
        return "transport error"
    if outcome.status_code is not None:
        return f"HTTP {outcome.status_code}"
    return "unknown failure"


def _measured_failure_reason(outcomes: list[RequestOutcome]) -> str:
    for outcome in outcomes:
        if outcome.failure_detail is not None:
            return outcome.failure_detail
    return "benchmark completed without a successful measured request"


async def run_benchmark(profile: BenchmarkProfile) -> BenchmarkResult:
    limits = httpx.Limits(
        max_keepalive_connections=profile.concurrency,
        max_connections=profile.concurrency,
    )
    semaphore = asyncio.Semaphore(profile.concurrency)

    try:
        input_factory = _build_input_factory(profile)
        async with httpx.AsyncClient(limits=limits) as client:
            await _verify_server_configuration(client=client, profile=profile)

            for request_index in range(profile.warmup_requests):
                outcome = await _issue_request(
                    client=client,
                    profile=profile,
                    request_index=request_index,
                    input_factory=input_factory,
                )
                if not outcome.success:
                    msg = f"warmup request {request_index} failed with {_format_outcome(outcome)}"
                    raise BenchmarkOperationalError(msg)

            async def wrapped_request(request_index: int) -> RequestOutcome:
                async with semaphore:
                    return await _issue_request(
                        client=client,
                        profile=profile,
                        request_index=request_index + profile.warmup_requests,
                        input_factory=input_factory,
                    )

            started_at = perf_counter()
            tasks = [
                asyncio.create_task(wrapped_request(index))
                for index in range(profile.total_requests)
            ]
            outcomes = await asyncio.gather(*tasks)
            duration_seconds = max(perf_counter() - started_at, 1e-9)
            results = _summarize_outcomes(
                profile=profile,
                outcomes=outcomes,
                duration_seconds=duration_seconds,
            )
    except BenchmarkOperationalError as exc:
        return BenchmarkResult(
            exit_code=2,
            profile=profile,
            results=None,
            failure_reason=str(exc),
        )

    if results.successful_requests == 0:
        return BenchmarkResult(
            exit_code=2,
            profile=profile,
            results=results,
            failure_reason=_measured_failure_reason(outcomes),
        )

    return BenchmarkResult(
        exit_code=0,
        profile=profile,
        results=results,
        failure_reason=None,
    )


def _format_latency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _format_human(result: BenchmarkResult) -> str:
    lines = [
        f"label: {result.profile.label}",
        f"hardware_id: {result.profile.hardware_id}",
        f"embed_url: {result.profile.embed_url}",
        f"total_texts: {result.profile.total_texts}",
        f"total_requests: {result.profile.total_requests}",
        f"warmup_requests: {result.profile.warmup_requests}",
        f"concurrency: {result.profile.concurrency}",
        f"inputs_per_request: {result.profile.inputs_per_request}",
        f"input_token_count: {result.profile.input_token_count}",
    ]

    if result.results is not None:
        lines.extend(
            [
                f"duration_seconds: {result.results.duration_seconds:.6f}",
                f"successful_requests: {result.results.successful_requests}",
                f"successful_texts: {result.results.successful_texts}",
                f"requests_per_second: {result.results.requests_per_second:.6f}",
                f"texts_per_second: {result.results.texts_per_second:.6f}",
                f"latency_p50_ms: {_format_latency(result.results.latency_p50_ms)}",
                f"latency_p95_ms: {_format_latency(result.results.latency_p95_ms)}",
                f"latency_p99_ms: {_format_latency(result.results.latency_p99_ms)}",
                f"error_count: {result.results.error_count}",
                f"timeout_count: {result.results.timeout_count}",
                f"status_counts: {result.results.status_counts}",
                f"transport_error_count: {result.results.transport_error_count}",
                f"invalid_response_count: {result.results.invalid_response_count}",
            ]
        )

    if result.failure_reason is not None:
        lines.append(f"failure_reason: {result.failure_reason}")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        profile = _build_profile(args)
    except ValueError as exc:
        failed_result = BenchmarkResult(
            exit_code=2,
            profile=BenchmarkProfile(
                embed_url=args.embed_url,
                hardware_id=args.hardware_id,
                label=args.label,
                concurrency=args.concurrency,
                total_texts=args.total_texts,
                warmup_requests=args.warmup_requests,
                inputs_per_request=args.inputs_per_request,
                input_token_count=args.input_token_count,
                timeout_seconds=args.timeout_seconds,
                model_id=args.model_id,
                model_revision=args.model_revision,
                device=args.device,
                dtype=args.dtype,
                max_length=args.max_length,
                truncate=args.truncate,
                max_batch_size=args.max_batch_size,
                max_batch_tokens=args.max_batch_tokens,
                batch_timeout_ms=args.batch_timeout_ms,
                max_batch_queue_size=args.max_batch_queue_size,
                batch_request_timeout_ms=args.batch_request_timeout_ms,
                warm_cache=args.warm_cache,
                warm_model=args.warm_model,
                warm_container=args.warm_container,
            ),
            results=None,
            failure_reason=str(exc),
        )
        if args.json_output:
            print(json.dumps(failed_result.to_dict(), sort_keys=True))
        else:
            print(_format_human(failed_result))
        return failed_result.exit_code

    result = asyncio.run(run_benchmark(profile))

    if args.json_output:
        print(json.dumps(result.to_dict(), sort_keys=True))
    else:
        print(_format_human(result))

    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
