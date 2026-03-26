from __future__ import annotations

import json

import pytest

from scripts import load_test


def _config() -> load_test.LoadTestConfig:
    return load_test.LoadTestConfig(
        embed_url="http://127.0.0.1:8000/embed",
        metrics_url="http://127.0.0.1:8000/metrics",
        ready_url="http://127.0.0.1:8000/readyz",
        duration_seconds=60.0,
        concurrency=8,
        warmup_requests=4,
        inputs_per_request=2,
        input_token_count=24,
        timeout_seconds=10.0,
        metrics_poll_interval_seconds=5.0,
        max_vram_drift_bytes=1024,
        hardware_id="local-dev",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="1" * 40,
        max_batch_size=128,
        max_batch_tokens=8192,
        batch_timeout_ms=2,
        max_batch_queue_size=1024,
        batch_request_timeout_ms=5000,
    )


def _result(*, passed: bool = True, exit_code: int = 0) -> load_test.LoadTestResult:
    request_summary = load_test.AggregatedResults()
    request_summary.total_requests = 128
    request_summary.status_counts["200"] = 128
    request_summary.category_counts["success"] = 128
    summary = load_test.NumericSeriesSummary(
        start=100.0,
        min=100.0,
        max=120.0,
        end=110.0,
        end_minus_start=10.0,
    )
    return load_test.LoadTestResult(
        passed=passed,
        exit_code=exit_code,
        config=_config(),
        request_summary=request_summary,
        metrics_deltas={
            "overload": 0.0,
            "timeout": 0.0,
            "internal_error": 0.0,
            "shutdown": 0.0,
            "unhandled_exceptions": 0.0,
            "gpu_oom": 0.0,
        },
        gpu_allocated_summary=summary,
        gpu_reserved_summary=summary,
        process_rss_summary=summary,
        gpu_metrics_available=True,
        checks={
            "all_responses_valid": True,
            "no_transport_errors": True,
            "no_internal_errors": True,
            "no_unhandled_exceptions": True,
            "gpu_oom_count_zero": True,
            "gpu_allocated_drift_within_threshold": True,
            "gpu_reserved_drift_within_threshold": True,
        },
        failure_reasons=[],
    )


def test_parse_args_defaults_and_overrides() -> None:
    defaults = load_test.parse_args(
        [
            "--hardware-id",
            "local-dev",
            "--model-id",
            "model",
            "--model-revision",
            "1" * 40,
        ]
    )

    assert defaults.embed_url == "http://127.0.0.1:8000/embed"
    assert defaults.metrics_url == "http://127.0.0.1:8000/metrics"
    assert defaults.ready_url == "http://127.0.0.1:8000/readyz"
    assert defaults.duration_seconds == 3600.0
    assert defaults.metrics_poll_interval_seconds == 5.0
    assert defaults.max_vram_drift_bytes == 268435456

    overridden = load_test.parse_args(
        [
            "--embed-url",
            "http://localhost:9000/embed",
            "--metrics-url",
            "http://localhost:9000/metrics",
            "--ready-url",
            "http://localhost:9000/readyz",
            "--duration-seconds",
            "120",
            "--concurrency",
            "4",
            "--max-vram-drift-bytes",
            "2048",
            "--hardware-id",
            "gpu-box",
            "--model-id",
            "model",
            "--model-revision",
            "2" * 40,
            "--json",
        ]
    )

    assert overridden.embed_url == "http://localhost:9000/embed"
    assert overridden.metrics_url == "http://localhost:9000/metrics"
    assert overridden.ready_url == "http://localhost:9000/readyz"
    assert overridden.duration_seconds == 120.0
    assert overridden.concurrency == 4
    assert overridden.max_vram_drift_bytes == 2048
    assert overridden.json_output is True


def test_parse_metrics_snapshot_supports_gpu_and_cpu_modes() -> None:
    gpu_snapshot = load_test._parse_metrics_snapshot(
        """
embedserve_request_failures_total{reason="overload"} 1
embedserve_request_failures_total{reason="timeout"} 2
embedserve_request_failures_total{reason="internal_error"} 3
embedserve_request_failures_total{reason="shutdown"} 4
embedserve_unhandled_exceptions_total 5
embedserve_gpu_oom_total{device="cuda:0"} 6
embedserve_gpu_memory_allocated_bytes{device="cuda:0"} 1024
embedserve_gpu_memory_reserved_bytes{device="cuda:0"} 2048
process_resident_memory_bytes 4096
        """.strip(),
        elapsed_seconds=1.5,
    )

    assert gpu_snapshot.request_failures_total["overload"] == 1.0
    assert gpu_snapshot.unhandled_exceptions_total == 5.0
    assert gpu_snapshot.gpu_oom_total == 6.0
    assert gpu_snapshot.gpu_memory_allocated_bytes == 1024.0
    assert gpu_snapshot.gpu_memory_reserved_bytes == 2048.0
    assert gpu_snapshot.process_resident_memory_bytes == 4096.0

    cpu_snapshot = load_test._parse_metrics_snapshot(
        'embedserve_request_failures_total{reason="overload"} 0\n'
        "embedserve_unhandled_exceptions_total 0\n",
        elapsed_seconds=2.0,
    )

    assert cpu_snapshot.gpu_oom_total == 0.0
    assert cpu_snapshot.gpu_memory_allocated_bytes is None
    assert cpu_snapshot.gpu_memory_reserved_bytes is None
    assert cpu_snapshot.process_resident_memory_bytes is None


def test_build_client_limits_reserves_metrics_headroom() -> None:
    limits = load_test._build_client_limits(8)

    assert limits.max_connections == 9
    assert limits.max_keepalive_connections == 9


def test_evaluate_checks_fails_on_gpu_drift_and_internal_errors() -> None:
    request_summary = load_test.AggregatedResults()
    request_summary.total_requests = 10
    request_summary.category_counts["internal_error"] = 1

    checks, failure_reasons = load_test._evaluate_checks(
        request_summary,
        {
            "overload": 0.0,
            "timeout": 0.0,
            "internal_error": 1.0,
            "shutdown": 0.0,
            "unhandled_exceptions": 0.0,
            "gpu_oom": 0.0,
        },
        max_vram_drift_bytes=32,
        gpu_allocated_summary=load_test.NumericSeriesSummary(
            start=0.0,
            min=0.0,
            max=128.0,
            end=64.0,
            end_minus_start=64.0,
        ),
        gpu_reserved_summary=None,
    )

    assert checks["no_internal_errors"] is False
    assert checks["gpu_allocated_drift_within_threshold"] is False
    assert "internal_errors_detected" in failure_reasons
    assert "gpu_allocated_memory_drift_exceeded_threshold" in failure_reasons


def test_main_supports_json_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_run(config: load_test.LoadTestConfig) -> load_test.LoadTestResult:
        assert config.hardware_id == "local-dev"
        return _result()

    monkeypatch.setattr("scripts.load_test.run_load_test", fake_run)

    exit_code = load_test.main(
        [
            "--hardware-id",
            "local-dev",
            "--model-id",
            "model",
            "--model-revision",
            "1" * 40,
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is True
    assert payload["request_summary"]["total_requests"] == 128


def test_main_returns_operational_error_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run(config: load_test.LoadTestConfig) -> load_test.LoadTestResult:
        raise load_test.LoadTestOperationalError("metrics_request_failed: timeout")

    monkeypatch.setattr("scripts.load_test.run_load_test", fake_run)

    exit_code = load_test.main(
        [
            "--hardware-id",
            "local-dev",
            "--model-id",
            "model",
            "--model-revision",
            "1" * 40,
            "--json",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is False
    assert payload["exit_code"] == 2
    assert payload["failure_reasons"] == ["metrics_request_failed: timeout"]
