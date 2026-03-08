from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts import verify_determinism as verifier


@dataclass
class FakeResponse:
    status_code: int
    payload: Any

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeClient:
    def __init__(self, sequence: list[Any]) -> None:
        self._sequence = list(sequence)
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
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


def _payload(vectors: list[list[float]]) -> dict[str, Any]:
    dim = len(vectors[0]) if vectors else 0
    return {
        "dim": dim,
        "data": [
            {
                "index": index,
                "embedding": vector,
            }
            for index, vector in enumerate(vectors)
        ],
    }


def _config() -> verifier.VerificationConfig:
    return verifier.VerificationConfig(
        url="http://127.0.0.1:8000/embed",
        iterations=2,
        max_abs_diff_threshold=2e-3,
        min_cosine_similarity_threshold=0.9995,
        timeout_seconds=10.0,
        inputs=["alpha", "beta"],
        input_source="builtin",
    )


def test_run_verification_passes_when_repeated_outputs_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(
        [
            FakeResponse(200, _payload([[1.0, 2.0], [3.0, 4.0]])),
            FakeResponse(200, _payload([[1.0, 2.0], [3.0, 4.0]])),
        ]
    )
    monkeypatch.setattr("scripts.verify_determinism.httpx.Client", lambda: client)

    result = verifier.run_verification(_config())

    assert result.exit_code == 0
    assert result.passed is True
    assert result.metrics is not None
    assert result.metrics.max_abs_diff == 0.0
    assert result.metrics.min_cosine_similarity == 1.0


def test_run_verification_fails_when_threshold_is_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(
        [
            FakeResponse(200, _payload([[1.0, 0.0], [0.0, 1.0]])),
            FakeResponse(200, _payload([[1.01, 0.0], [0.0, 1.0]])),
        ]
    )
    monkeypatch.setattr("scripts.verify_determinism.httpx.Client", lambda: client)

    result = verifier.run_verification(
        verifier.VerificationConfig(
            url="http://127.0.0.1:8000/embed",
            iterations=2,
            max_abs_diff_threshold=1e-3,
            min_cosine_similarity_threshold=0.5,
            timeout_seconds=10.0,
            inputs=["alpha", "beta"],
            input_source="builtin",
        )
    )

    assert result.exit_code == 1
    assert result.passed is False
    assert result.failure_reason is not None
    assert "max_abs_diff" in result.failure_reason


def test_run_verification_fails_when_cosine_threshold_is_not_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(
        [
            FakeResponse(200, _payload([[1.0, 0.0], [0.0, 1.0]])),
            FakeResponse(200, _payload([[-1.0, 0.0], [0.0, 1.0]])),
        ]
    )
    monkeypatch.setattr("scripts.verify_determinism.httpx.Client", lambda: client)

    result = verifier.run_verification(
        verifier.VerificationConfig(
            url="http://127.0.0.1:8000/embed",
            iterations=2,
            max_abs_diff_threshold=10.0,
            min_cosine_similarity_threshold=0.9,
            timeout_seconds=10.0,
            inputs=["alpha", "beta"],
            input_source="builtin",
        )
    )

    assert result.exit_code == 1
    assert result.passed is False
    assert result.failure_reason is not None
    assert "min_cosine_similarity" in result.failure_reason


def test_run_verification_returns_operational_failure_on_non_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient([FakeResponse(500, {"detail": "oops"})])
    monkeypatch.setattr("scripts.verify_determinism.httpx.Client", lambda: client)

    result = verifier.run_verification(_config())

    assert result.exit_code == 2
    assert result.passed is False
    assert result.failure_reason == "received HTTP 500"


def test_run_verification_returns_operational_failure_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient([httpx.TimeoutException("timeout")])
    monkeypatch.setattr("scripts.verify_determinism.httpx.Client", lambda: client)

    result = verifier.run_verification(_config())

    assert result.exit_code == 2
    assert result.passed is False
    assert result.failure_reason is not None
    assert "timed out" in result.failure_reason


def test_run_verification_returns_operational_failure_on_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(
        [
            FakeResponse(200, _payload([[1.0, 2.0], [3.0, 4.0]])),
            FakeResponse(200, _payload([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])),
        ]
    )
    monkeypatch.setattr("scripts.verify_determinism.httpx.Client", lambda: client)

    result = verifier.run_verification(_config())

    assert result.exit_code == 2
    assert result.passed is False
    assert result.failure_reason is not None
    assert "dim changed" in result.failure_reason


def test_run_verification_returns_operational_failure_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient([FakeResponse(200, {"dim": 2, "data": "bad"})])
    monkeypatch.setattr("scripts.verify_determinism.httpx.Client", lambda: client)

    result = verifier.run_verification(_config())

    assert result.exit_code == 2
    assert result.passed is False
    assert result.failure_reason == "response payload missing list data"


def test_parse_args_defaults_and_overrides() -> None:
    defaults = verifier.parse_args([])

    assert defaults.url == "http://127.0.0.1:8000/embed"
    assert defaults.iterations == 25
    assert defaults.max_abs_diff == 2e-3
    assert defaults.min_cosine_similarity == 0.9995
    assert defaults.timeout_seconds == 10.0
    assert defaults.inputs_file is None
    assert defaults.json_output is False

    overridden = verifier.parse_args(
        [
            "--url",
            "http://localhost:8010/embed",
            "--iterations",
            "12",
            "--max-abs-diff",
            "0.1",
            "--min-cosine-similarity",
            "0.7",
            "--timeout-seconds",
            "2.5",
            "--json",
        ]
    )

    assert overridden.url == "http://localhost:8010/embed"
    assert overridden.iterations == 12
    assert overridden.max_abs_diff == 0.1
    assert overridden.min_cosine_similarity == 0.7
    assert overridden.timeout_seconds == 2.5
    assert overridden.json_output is True


def test_main_supports_inputs_file_and_json_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs_path = tmp_path / "inputs.txt"
    inputs_path.write_text("alpha\nbeta\n", encoding="utf-8")

    def fake_run(config: verifier.VerificationConfig) -> verifier.VerificationResult:
        assert config.inputs == ["alpha", "beta"]
        assert config.input_source == str(inputs_path)
        return verifier.VerificationResult(
            passed=True,
            exit_code=0,
            config=config,
            metrics=verifier.VerificationMetrics(max_abs_diff=0.0, min_cosine_similarity=1.0),
            failure_reason=None,
        )

    monkeypatch.setattr("scripts.verify_determinism.run_verification", fake_run)

    exit_code = verifier.main(["--inputs-file", str(inputs_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is True
    assert payload["config"]["input_count"] == 2
    assert payload["config"]["input_source"] == str(inputs_path)


def test_main_returns_operational_code_for_invalid_inputs_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text('{"inputs": ["alpha"]}', encoding="utf-8")

    exit_code = verifier.main(["--inputs-file", str(inputs_path), "--json"])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is False
    assert payload["exit_code"] == 2
    assert payload["failure_reason"] is not None
