from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_URL = "http://127.0.0.1:8000/embed"
DEFAULT_INPUTS = (
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "1234567890",
    "Symbols: !@#$%^&*()[]{}<>/?",
    "mixed CASE Input with punctuation, numbers 42 and tabs\\t",
    "short",
    "Longer sentence to create a wider tokenization footprint for drift checks.",
    "line one\\nline two",
    "    leading and trailing spaces preserved    ",
    "final sample input",
)


@dataclass(frozen=True)
class VerificationConfig:
    url: str
    iterations: int
    max_abs_diff_threshold: float
    min_cosine_similarity_threshold: float
    timeout_seconds: float
    inputs: list[str]
    input_source: str


@dataclass(frozen=True)
class VerificationMetrics:
    max_abs_diff: float
    min_cosine_similarity: float


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    exit_code: int
    config: VerificationConfig
    metrics: VerificationMetrics | None
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pass": self.passed,
            "exit_code": self.exit_code,
            "config": {
                "url": self.config.url,
                "iterations": self.config.iterations,
                "max_abs_diff_threshold": self.config.max_abs_diff_threshold,
                "min_cosine_similarity_threshold": self.config.min_cosine_similarity_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "input_source": self.config.input_source,
                "input_count": len(self.config.inputs),
            },
            "failure_reason": self.failure_reason,
        }
        if self.metrics is not None:
            payload["metrics"] = {
                "max_abs_diff": self.metrics.max_abs_diff,
                "min_cosine_similarity": self.metrics.min_cosine_similarity,
            }
        else:
            payload["metrics"] = None
        return payload


class VerificationOperationalError(Exception):
    pass


def _iterations_type(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        msg = "iterations must be >= 2"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _timeout_type(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        msg = "timeout-seconds must be > 0"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify embedding numerical determinism against a live server"
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--inputs-file", default=None)
    parser.add_argument("--iterations", type=_iterations_type, default=25)
    parser.add_argument("--max-abs-diff", type=float, default=2e-3)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.9995)
    parser.add_argument("--timeout-seconds", type=_timeout_type, default=10.0)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _load_inputs(inputs_file: str | None) -> tuple[list[str], str]:
    if inputs_file is None:
        return list(DEFAULT_INPUTS), "builtin"

    path = Path(inputs_file)
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()

    if stripped.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            msg = "inputs-file JSON payload must be an array of strings"
            raise ValueError(msg)
        inputs = [item for item in parsed if item.strip()]
    else:
        inputs = [line for line in raw.splitlines() if line.strip()]

    if not inputs:
        msg = "inputs-file did not provide any non-empty inputs"
        raise ValueError(msg)

    return inputs, str(path)


def _parse_embed_response(payload: Any, *, expected_items: int) -> tuple[list[list[float]], int]:
    if not isinstance(payload, dict):
        msg = "response payload is not an object"
        raise VerificationOperationalError(msg)

    dim = payload.get("dim")
    data = payload.get("data")

    if not isinstance(dim, int) or dim < 0:
        msg = "response payload missing valid integer dim"
        raise VerificationOperationalError(msg)

    if not isinstance(data, list):
        msg = "response payload missing list data"
        raise VerificationOperationalError(msg)

    if len(data) != expected_items:
        msg = f"response data length changed: expected {expected_items}, got {len(data)}"
        raise VerificationOperationalError(msg)

    vectors: list[list[float]] = []
    for expected_index, item in enumerate(data):
        if not isinstance(item, dict):
            msg = f"response data item at index {expected_index} is not an object"
            raise VerificationOperationalError(msg)

        index = item.get("index")
        embedding = item.get("embedding")
        if index != expected_index:
            msg = f"response index mismatch at position {expected_index}: got {index}"
            raise VerificationOperationalError(msg)

        if not isinstance(embedding, list):
            msg = f"response embedding at index {expected_index} is not a list"
            raise VerificationOperationalError(msg)

        vector: list[float] = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = f"embedding at index {expected_index} contains non-numeric values"
                raise VerificationOperationalError(msg)
            vector.append(float(value))

        if len(vector) != dim:
            msg = (
                f"embedding width mismatch at index {expected_index}: "
                f"expected {dim}, got {len(vector)}"
            )
            raise VerificationOperationalError(msg)

        vectors.append(vector)

    return vectors, dim


def _fetch_embeddings(
    client: httpx.Client,
    *,
    url: str,
    timeout_seconds: float,
    inputs: list[str],
) -> tuple[list[list[float]], int]:
    try:
        response = client.post(
            url,
            json={"inputs": inputs},
            timeout=timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        msg = f"request timed out after {timeout_seconds} seconds"
        raise VerificationOperationalError(msg) from exc
    except httpx.HTTPError as exc:
        msg = f"request failed: {exc}"
        raise VerificationOperationalError(msg) from exc

    if response.status_code != 200:
        msg = f"received HTTP {response.status_code}"
        raise VerificationOperationalError(msg)

    try:
        payload = response.json()
    except ValueError as exc:
        msg = "response body is not valid JSON"
        raise VerificationOperationalError(msg) from exc

    return _parse_embed_response(payload, expected_items=len(inputs))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        msg = "cosine similarity vectors must have equal size"
        raise VerificationOperationalError(msg)

    dot = 0.0
    left_norm_sq = 0.0
    right_norm_sq = 0.0

    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_norm_sq += left_value * left_value
        right_norm_sq += right_value * right_value

    left_norm = math.sqrt(left_norm_sq)
    right_norm = math.sqrt(right_norm_sq)

    if left_norm == 0 and right_norm == 0:
        return 1.0
    if left_norm == 0 or right_norm == 0:
        return 0.0

    similarity = dot / (left_norm * right_norm)
    if math.isclose(similarity, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return 1.0
    if math.isclose(similarity, -1.0, rel_tol=0.0, abs_tol=1e-12):
        return -1.0
    return max(-1.0, min(1.0, similarity))


def run_verification(config: VerificationConfig) -> VerificationResult:
    max_abs_diff = 0.0
    min_cosine_similarity = 1.0

    try:
        with httpx.Client() as client:
            baseline_vectors, baseline_dim = _fetch_embeddings(
                client,
                url=config.url,
                timeout_seconds=config.timeout_seconds,
                inputs=config.inputs,
            )

            for _ in range(config.iterations - 1):
                current_vectors, current_dim = _fetch_embeddings(
                    client,
                    url=config.url,
                    timeout_seconds=config.timeout_seconds,
                    inputs=config.inputs,
                )

                if current_dim != baseline_dim:
                    msg = (
                        f"embedding dim changed across runs: baseline={baseline_dim}, "
                        f"current={current_dim}"
                    )
                    raise VerificationOperationalError(msg)

                if len(current_vectors) != len(baseline_vectors):
                    msg = (
                        "response vector count changed across runs: "
                        f"baseline={len(baseline_vectors)}, current={len(current_vectors)}"
                    )
                    raise VerificationOperationalError(msg)

                for vector_index, (baseline, current) in enumerate(
                    zip(baseline_vectors, current_vectors, strict=True)
                ):
                    if len(current) != len(baseline):
                        msg = (
                            f"embedding width changed at index {vector_index}: "
                            f"baseline={len(baseline)}, current={len(current)}"
                        )
                        raise VerificationOperationalError(msg)

                    for baseline_value, current_value in zip(baseline, current, strict=True):
                        diff = abs(current_value - baseline_value)
                        if diff > max_abs_diff:
                            max_abs_diff = diff

                    similarity = _cosine_similarity(baseline, current)
                    if similarity < min_cosine_similarity:
                        min_cosine_similarity = similarity

    except VerificationOperationalError as exc:
        return VerificationResult(
            passed=False,
            exit_code=2,
            config=config,
            metrics=None,
            failure_reason=str(exc),
        )

    metrics = VerificationMetrics(
        max_abs_diff=max_abs_diff,
        min_cosine_similarity=min_cosine_similarity,
    )

    if max_abs_diff > config.max_abs_diff_threshold:
        return VerificationResult(
            passed=False,
            exit_code=1,
            config=config,
            metrics=metrics,
            failure_reason=(
                f"max_abs_diff {max_abs_diff:.8f} exceeded threshold "
                f"{config.max_abs_diff_threshold:.8f}"
            ),
        )

    if min_cosine_similarity < config.min_cosine_similarity_threshold:
        return VerificationResult(
            passed=False,
            exit_code=1,
            config=config,
            metrics=metrics,
            failure_reason=(
                f"min_cosine_similarity {min_cosine_similarity:.8f} below threshold "
                f"{config.min_cosine_similarity_threshold:.8f}"
            ),
        )

    return VerificationResult(
        passed=True,
        exit_code=0,
        config=config,
        metrics=metrics,
        failure_reason=None,
    )


def _format_human(result: VerificationResult) -> str:
    lines = [
        f"verification: {'PASS' if result.passed else 'FAIL'}",
        f"url: {result.config.url}",
        f"iterations: {result.config.iterations}",
        f"input_count: {len(result.config.inputs)}",
        f"input_source: {result.config.input_source}",
        (
            "thresholds: "
            f"max_abs_diff<={result.config.max_abs_diff_threshold}, "
            f"min_cosine_similarity>={result.config.min_cosine_similarity_threshold}"
        ),
    ]

    if result.metrics is not None:
        lines.append(f"measured max_abs_diff: {result.metrics.max_abs_diff:.8f}")
        lines.append(f"measured min_cosine_similarity: {result.metrics.min_cosine_similarity:.8f}")

    if result.failure_reason is not None:
        lines.append(f"failure_reason: {result.failure_reason}")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        inputs, input_source = _load_inputs(args.inputs_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failed_result = VerificationResult(
            passed=False,
            exit_code=2,
            config=VerificationConfig(
                url=args.url,
                iterations=args.iterations,
                max_abs_diff_threshold=args.max_abs_diff,
                min_cosine_similarity_threshold=args.min_cosine_similarity,
                timeout_seconds=args.timeout_seconds,
                inputs=[],
                input_source=args.inputs_file or "builtin",
            ),
            metrics=None,
            failure_reason=str(exc),
        )
        if args.json_output:
            print(json.dumps(failed_result.to_dict(), sort_keys=True))
        else:
            print(_format_human(failed_result))
        return failed_result.exit_code

    result = run_verification(
        VerificationConfig(
            url=args.url,
            iterations=args.iterations,
            max_abs_diff_threshold=args.max_abs_diff,
            min_cosine_similarity_threshold=args.min_cosine_similarity,
            timeout_seconds=args.timeout_seconds,
            inputs=inputs,
            input_source=input_source,
        )
    )

    if args.json_output:
        print(json.dumps(result.to_dict(), sort_keys=True))
    else:
        print(_format_human(result))

    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
