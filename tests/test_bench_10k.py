from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from scripts import bench_10k as bench


@dataclass
class FakeResponse:
    status_code: int
    payload: Any

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeAsyncClient:
    def __init__(self, sequence: list[Any]) -> None:
        self._sequence = list(sequence)
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append((url, json, timeout))
        if not self._sequence:
            msg = "No more fake responses configured"
            raise AssertionError(msg)

        next_item = self._sequence.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        if not isinstance(next_item, FakeResponse):
            msg = "Configured item must be FakeResponse or Exception"
            raise AssertionError(msg)
        return next_item


def _payload(vectors: list[list[float]], *, tokens: int = 8) -> dict[str, Any]:
    dim = len(vectors[0]) if vectors else 0
    return {
        "data": [
            {
                "index": index,
                "embedding": vector,
            }
            for index, vector in enumerate(vectors)
        ],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "826711e54e001c83835913827a843d8dd0a1def9",
        "dim": dim,
        "usage": {"tokens": tokens},
    }


def _payload_with_metadata(
    vectors: list[list[float]],
    *,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    revision: str = "826711e54e001c83835913827a843d8dd0a1def9",
    tokens: int = 8,
) -> dict[str, Any]:
    dim = len(vectors[0]) if vectors else 0
    return {
        "data": [
            {
                "index": index,
                "embedding": vector,
            }
            for index, vector in enumerate(vectors)
        ],
        "model": model,
        "revision": revision,
        "dim": dim,
        "usage": {"tokens": tokens},
    }


def _profile(
    *,
    total_texts: int = 4,
    warmup_requests: int = 1,
    inputs_per_request: int = 1,
) -> bench.BenchmarkProfile:
    return bench.BenchmarkProfile(
        embed_url="http://127.0.0.1:8000/embed",
        hardware_id="local-dev",
        label="batched",
        concurrency=2,
        total_texts=total_texts,
        warmup_requests=warmup_requests,
        inputs_per_request=inputs_per_request,
        input_token_count=4,
        timeout_seconds=10.0,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="826711e54e001c83835913827a843d8dd0a1def9",
        device="cpu",
        dtype="float32",
        max_length=512,
        truncate=True,
        max_batch_size=128,
        max_batch_tokens=8192,
        batch_timeout_ms=2,
        max_batch_queue_size=1024,
        batch_request_timeout_ms=5000,
        warm_cache=True,
        warm_model=True,
        warm_container=True,
    )


def test_parse_args_defaults_and_overrides() -> None:
    defaults = bench.parse_args([])

    assert defaults.embed_url == "http://127.0.0.1:8000/embed"
    assert defaults.hardware_id == "local-dev"
    assert defaults.label == "batched"
    assert defaults.concurrency == 32
    assert defaults.total_texts == 10000
    assert defaults.warmup_requests == 200
    assert defaults.inputs_per_request == 1
    assert defaults.input_token_count == 24
    assert defaults.timeout_seconds == 10.0
    assert defaults.json_output is False

    overridden = bench.parse_args(
        [
            "--embed-url",
            "http://localhost:8010/embed",
            "--hardware-id",
            "lab-gpu",
            "--label",
            "no_batching",
            "--concurrency",
            "8",
            "--total-texts",
            "2000",
            "--warmup-requests",
            "32",
            "--inputs-per-request",
            "2",
            "--input-token-count",
            "16",
            "--timeout-seconds",
            "3.5",
            "--warm-cache",
            "false",
            "--json",
        ]
    )

    assert overridden.embed_url == "http://localhost:8010/embed"
    assert overridden.hardware_id == "lab-gpu"
    assert overridden.label == "no_batching"
    assert overridden.concurrency == 8
    assert overridden.total_texts == 2000
    assert overridden.warmup_requests == 32
    assert overridden.inputs_per_request == 2
    assert overridden.input_token_count == 16
    assert overridden.timeout_seconds == 3.5
    assert overridden.warm_cache is False
    assert overridden.json_output is True


def test_parse_args_allows_zero_warmup_requests() -> None:
    parsed = bench.parse_args(["--warmup-requests", "0"])

    assert parsed.warmup_requests == 0


def test_build_profile_rejects_non_divisible_total_texts() -> None:
    args = bench.parse_args(["--total-texts", "3", "--inputs-per-request", "2"])

    with pytest.raises(ValueError, match="divisible"):
        bench._build_profile(args)


def test_summarize_outcomes_uses_successes_only_for_latency_and_throughput() -> None:
    profile = _profile(total_texts=8, warmup_requests=1, inputs_per_request=2)
    outcomes = [
        bench.RequestOutcome(status_code=200, success=True, latency_ms=10.0),
        bench.RequestOutcome(status_code=200, success=True, latency_ms=20.0),
        bench.RequestOutcome(status_code=200, success=True, latency_ms=30.0),
        bench.RequestOutcome(status_code=200, success=True, latency_ms=40.0),
        bench.RequestOutcome(status_code=503, success=False, latency_ms=None),
        bench.RequestOutcome(
            status_code=200, success=False, latency_ms=None, invalid_response=True
        ),
        bench.RequestOutcome(
            status_code=None, success=False, latency_ms=None, transport_error=True
        ),
        bench.RequestOutcome(status_code=None, success=False, latency_ms=None, timed_out=True),
    ]

    results = bench._summarize_outcomes(
        profile=profile,
        outcomes=outcomes,
        duration_seconds=2.0,
    )

    assert results.successful_requests == 4
    assert results.successful_texts == 8
    assert results.requests_per_second == 2.0
    assert results.texts_per_second == 4.0
    assert results.latency_p50_ms == 20.0
    assert results.latency_p95_ms == 40.0
    assert results.latency_p99_ms == 40.0
    assert results.error_count == 3
    assert results.timeout_count == 1
    assert results.status_counts == {"200": 5, "503": 1, "other": 0}
    assert results.transport_error_count == 1
    assert results.invalid_response_count == 1


def test_summarize_outcomes_preserves_counts_when_every_request_fails() -> None:
    profile = _profile(total_texts=4, warmup_requests=0, inputs_per_request=1)
    outcomes = [
        bench.RequestOutcome(status_code=503, success=False, latency_ms=None),
        bench.RequestOutcome(
            status_code=200,
            success=False,
            latency_ms=None,
            invalid_response=True,
            failure_detail="response model mismatch: expected model-a, got model-b",
        ),
        bench.RequestOutcome(
            status_code=None,
            success=False,
            latency_ms=None,
            transport_error=True,
        ),
        bench.RequestOutcome(status_code=None, success=False, latency_ms=None, timed_out=True),
    ]

    results = bench._summarize_outcomes(
        profile=profile,
        outcomes=outcomes,
        duration_seconds=2.0,
    )

    assert results.successful_requests == 0
    assert results.successful_texts == 0
    assert results.requests_per_second == 0.0
    assert results.texts_per_second == 0.0
    assert results.latency_p50_ms is None
    assert results.latency_p95_ms is None
    assert results.latency_p99_ms is None
    assert results.error_count == 3
    assert results.timeout_count == 1
    assert results.status_counts == {"200": 1, "503": 1, "other": 0}
    assert results.transport_error_count == 1
    assert results.invalid_response_count == 1


def test_run_benchmark_excludes_warmup_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeAsyncClient(
        [
            FakeResponse(200, _payload([[1.0, 2.0]])),
            FakeResponse(200, _payload([[1.0, 2.0]])),
            FakeResponse(200, _payload([[1.0, 2.0]])),
        ]
    )
    monkeypatch.setattr("scripts.bench_10k.httpx.AsyncClient", lambda **_: client)

    result = asyncio.run(bench.run_benchmark(_profile(total_texts=2, warmup_requests=1)))

    assert result.exit_code == 0
    assert result.results is not None
    assert result.results.successful_requests == 2
    assert result.results.successful_texts == 2
    assert len(client.calls) == 3


def test_run_benchmark_returns_operational_failure_when_warmup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeAsyncClient([FakeResponse(503, {"detail": "queue full"})])
    monkeypatch.setattr("scripts.bench_10k.httpx.AsyncClient", lambda **_: client)

    result = asyncio.run(bench.run_benchmark(_profile(total_texts=2, warmup_requests=1)))

    assert result.exit_code == 2
    assert result.results is None
    assert result.failure_reason == "warmup request 0 failed with HTTP 503"


def test_run_benchmark_preserves_results_when_measured_requests_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeAsyncClient(
        [
            FakeResponse(503, {"detail": "queue full"}),
            httpx.TimeoutException("timed out"),
        ]
    )
    monkeypatch.setattr("scripts.bench_10k.httpx.AsyncClient", lambda **_: client)

    result = asyncio.run(bench.run_benchmark(_profile(total_texts=2, warmup_requests=0)))

    assert result.exit_code == 2
    assert result.results is not None
    assert result.results.successful_requests == 0
    assert result.results.latency_p50_ms is None
    assert result.results.error_count == 1
    assert result.results.timeout_count == 1
    assert result.results.status_counts == {"200": 0, "503": 1, "other": 0}
    assert result.failure_reason == "benchmark completed without a successful measured request"


def test_run_benchmark_rejects_wrong_model_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeAsyncClient(
        [
            FakeResponse(
                200,
                _payload_with_metadata(
                    [[1.0, 2.0]],
                    revision="1111111111111111111111111111111111111111",
                ),
            )
        ]
    )
    monkeypatch.setattr("scripts.bench_10k.httpx.AsyncClient", lambda **_: client)

    result = asyncio.run(bench.run_benchmark(_profile(total_texts=1, warmup_requests=0)))

    assert result.exit_code == 2
    assert result.results is not None
    assert result.results.successful_requests == 0
    assert result.results.invalid_response_count == 1
    assert (
        result.failure_reason
        == "response revision mismatch: expected 826711e54e001c83835913827a843d8dd0a1def9, got 1111111111111111111111111111111111111111"
    )


def test_main_supports_json_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run(profile: bench.BenchmarkProfile) -> bench.BenchmarkResult:
        return bench.BenchmarkResult(
            exit_code=0,
            profile=profile,
            results=bench.BenchmarkResults(
                duration_seconds=1.25,
                successful_requests=4,
                successful_texts=4,
                requests_per_second=3.2,
                texts_per_second=3.2,
                latency_p50_ms=12.0,
                latency_p95_ms=15.0,
                latency_p99_ms=15.0,
                error_count=0,
                timeout_count=0,
                status_counts={"200": 4, "503": 0, "other": 0},
                transport_error_count=0,
                invalid_response_count=0,
            ),
            failure_reason=None,
        )

    monkeypatch.setattr("scripts.bench_10k.run_benchmark", fake_run)

    exit_code = bench.main(["--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == 0
    assert payload["profile"]["label"] == "batched"
    assert payload["results"]["successful_requests"] == 4


def test_main_returns_operational_code_for_invalid_profile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = bench.main(
        [
            "--total-texts",
            "3",
            "--inputs-per-request",
            "2",
            "--json",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == 2
    assert payload["failure_reason"] == "total-texts must be divisible by inputs-per-request"
