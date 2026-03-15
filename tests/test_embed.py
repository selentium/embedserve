from __future__ import annotations

import asyncio
from collections.abc import Iterator
from math import sqrt
from typing import TypedDict

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.engine.embedder import _resolve_effective_max_length
from app.main import create_app
from app.runtime import RuntimeInitializer, RuntimeState
from app.settings import Settings
from tests.fakes import (
    FakeModel,
    FakeTokenizer,
    make_fake_embedder,
    make_ready_runtime,
    make_unready_runtime,
)


class RuntimeCapture(TypedDict, total=False):
    tokenizer: FakeTokenizer
    model: FakeModel


def _ready_initializer_factory(
    *,
    truncate: bool = True,
    normalize_embeddings: bool = True,
    output_dtype: str = "float32",
    effective_max_length: int = 8,
    token_map: dict[str, list[int]] | None = None,
    include_last_hidden_state: bool = True,
    capture: RuntimeCapture | None = None,
) -> RuntimeInitializer:
    def initializer(settings: Settings) -> RuntimeState:
        embedder, tokenizer, model = make_fake_embedder(
            model_id=settings.MODEL_ID,
            revision=settings.MODEL_REVISION,
            truncate=truncate,
            normalize_embeddings=normalize_embeddings,
            output_dtype=output_dtype,
            effective_max_length=effective_max_length,
            token_map=token_map,
            include_last_hidden_state=include_last_hidden_state,
        )
        if capture is not None:
            capture["tokenizer"] = tokenizer
            capture["model"] = model
        return make_ready_runtime(
            model_id=settings.MODEL_ID,
            revision=settings.MODEL_REVISION,
            embedder=embedder,
        )

    return initializer


def _unready_initializer(settings: Settings) -> RuntimeState:
    return make_unready_runtime(
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
        reason="initialization_failed",
        detail="model bootstrap failed",
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(runtime_initializer=_ready_initializer_factory())
    with TestClient(app) as test_client:
        yield test_client


def test_embed_single_input_returns_model_response(client: TestClient) -> None:
    response = client.post("/embed", json={"inputs": ["first text"]})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]

    payload = response.json()
    assert payload["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert payload["revision"] == "826711e54e001c83835913827a843d8dd0a1def9"
    assert payload["dim"] == 3
    assert payload["usage"] == {"tokens": 4}
    assert len(payload["data"]) == 1
    assert payload["data"][0]["index"] == 0
    assert len(payload["data"][0]["embedding"]) == payload["dim"]


def test_embed_multi_input_preserves_order_and_is_deterministic(client: TestClient) -> None:
    request_payload = {"inputs": ["alpha beta", "gamma", "delta epsilon zeta"]}

    first_response = client.post("/embed", json=request_payload)
    second_response = client.post("/embed", json=request_payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_data = first_response.json()["data"]
    second_data = second_response.json()["data"]

    assert [item["index"] for item in first_data] == [0, 1, 2]
    assert first_data == second_data
    assert first_data[0]["embedding"] != first_data[1]["embedding"]


def test_embed_accepts_surrounding_whitespace_without_trimming() -> None:
    app = create_app(
        runtime_initializer=_ready_initializer_factory(
            token_map={
                "  padded text  ": [9, 8, 7],
                "padded text": [1, 2],
            }
        )
    )

    with TestClient(app) as client:
        spaced_response = client.post("/embed", json={"inputs": ["  padded text  "]})
        trimmed_response = client.post("/embed", json={"inputs": ["padded text"]})

    assert spaced_response.status_code == 200
    assert trimmed_response.status_code == 200
    assert (
        spaced_response.json()["data"][0]["embedding"]
        != trimmed_response.json()["data"][0]["embedding"]
    )


def test_embed_supports_in_process_asgi_transport() -> None:
    async def run_request() -> tuple[int, str]:
        app = create_app(runtime_initializer=_ready_initializer_factory())
        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client,
        ):
            response = await asyncio.wait_for(
                client.post("/embed", json={"inputs": ["first text"]}),
                timeout=1,
            )

        return response.status_code, response.headers["X-Request-ID"]

    status_code, request_id = asyncio.run(run_request())

    assert status_code == 200
    assert request_id


@pytest.mark.parametrize(
    ("payload", "expected_loc"),
    [
        ({}, ["body", "inputs"]),
        ({"inputs": "not-a-list"}, ["body", "inputs"]),
        ({"inputs": []}, ["body", "inputs"]),
        ({"inputs": [""]}, ["body", "inputs"]),
        ({"inputs": ["   "]}, ["body", "inputs"]),
        ({"inputs": ["valid"], "extra": "field"}, ["body", "extra"]),
        ({"inputs": ["valid", "   "]}, ["body", "inputs"]),
    ],
)
def test_embed_validation_failures_return_fastapi_422_when_runtime_is_unready(
    payload: dict[str, object],
    expected_loc: list[str],
) -> None:
    app = create_app(runtime_initializer=_unready_initializer)

    with TestClient(app) as client:
        response = client.post("/embed", json=payload)

    assert response.status_code == 422
    assert response.headers["X-Request-ID"]
    body = response.json()
    assert "detail" in body
    assert body["detail"][0]["loc"] == expected_loc


def test_embed_respects_max_inputs_per_request_before_readiness_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_INPUTS_PER_REQUEST", "2")
    app = create_app(runtime_initializer=_unready_initializer)

    with TestClient(app) as client:
        success_response = client.post("/embed", json={"inputs": ["one", "two"]})
        failure_response = client.post("/embed", json={"inputs": ["one", "two", "three"]})

    assert success_response.status_code == 503
    assert failure_response.status_code == 422
    assert failure_response.headers["X-Request-ID"]
    assert failure_response.json()["detail"][0]["loc"] == ["body", "inputs"]


def test_embed_returns_503_when_runtime_is_unready() -> None:
    app = create_app(runtime_initializer=_unready_initializer)

    with TestClient(app) as client:
        response = client.post("/embed", json={"inputs": ["first text"]})

    assert response.status_code == 503
    assert response.headers["X-Request-ID"]
    assert response.json() == {"detail": "Model is not ready: initialization_failed"}


def test_embed_truncates_when_enabled_and_counts_post_truncation_tokens() -> None:
    app = create_app(
        runtime_initializer=_ready_initializer_factory(
            truncate=True,
            effective_max_length=4,
            token_map={"long input": [1, 2, 3, 4, 5, 6]},
        )
    )

    with TestClient(app) as client:
        response = client.post("/embed", json={"inputs": ["long input"]})

    assert response.status_code == 200
    assert response.json()["usage"] == {"tokens": 4}


def test_embed_rejects_overlength_inputs_when_truncation_is_disabled() -> None:
    capture: RuntimeCapture = {}
    app = create_app(
        runtime_initializer=_ready_initializer_factory(
            truncate=False,
            effective_max_length=4,
            token_map={"too long": [1, 2, 3, 4, 5]},
            capture=capture,
        )
    )

    with TestClient(app) as client:
        response = client.post("/embed", json={"inputs": ["too long"]})

    assert "tokenizer" in capture
    assert "model" in capture
    tokenizer = capture["tokenizer"]
    model = capture["model"]

    assert response.status_code == 422
    assert response.headers["X-Request-ID"]
    assert response.json()["detail"][0]["loc"] == ["body", "inputs", 0]
    assert "Input token length" in response.json()["detail"][0]["msg"]
    assert tokenizer.padded_batch_calls == 0
    assert model.forward_calls == 0


def test_embed_normalizes_vectors_by_default() -> None:
    app = create_app(runtime_initializer=_ready_initializer_factory(normalize_embeddings=True))

    with TestClient(app) as client:
        response = client.post("/embed", json={"inputs": ["alpha beta"]})

    embedding = response.json()["data"][0]["embedding"]
    norm = sqrt(sum(value * value for value in embedding))
    assert norm == pytest.approx(1.0, rel=1e-6)


def test_embed_can_disable_normalization() -> None:
    app = create_app(runtime_initializer=_ready_initializer_factory(normalize_embeddings=False))

    with TestClient(app) as client:
        response = client.post("/embed", json={"inputs": ["alpha beta"]})

    embedding = response.json()["data"][0]["embedding"]
    norm = sqrt(sum(value * value for value in embedding))
    assert norm > 1.0


def test_embed_output_dtype_affects_returned_precision() -> None:
    float32_app = create_app(runtime_initializer=_ready_initializer_factory(output_dtype="float32"))
    float16_app = create_app(runtime_initializer=_ready_initializer_factory(output_dtype="float16"))

    with TestClient(float32_app) as float32_client:
        float32_response = float32_client.post("/embed", json={"inputs": ["alpha beta gamma"]})
    with TestClient(float16_app) as float16_client:
        float16_response = float16_client.post("/embed", json={"inputs": ["alpha beta gamma"]})

    float32_embedding = float32_response.json()["data"][0]["embedding"]
    float16_embedding = float16_response.json()["data"][0]["embedding"]

    assert float32_embedding != float16_embedding


def test_effective_max_length_respects_tokenizer_cap() -> None:
    assert (
        _resolve_effective_max_length(
            configured_max_length=512,
            tokenizer_max_length=256,
        )
        == 256
    )
    assert (
        _resolve_effective_max_length(
            configured_max_length=512,
            tokenizer_max_length=10**12,
        )
        == 512
    )


def test_embed_openapi_documents_503_response() -> None:
    app = create_app(runtime_initializer=_ready_initializer_factory())

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    embed_operation = response.json()["paths"]["/embed"]["post"]
    assert "503" in embed_operation["responses"]
    response_503 = embed_operation["responses"]["503"]
    examples = response_503["content"]["application/json"]["examples"]
    assert set(examples) == {"unready", "queue_full", "request_timeout", "shutdown"}
