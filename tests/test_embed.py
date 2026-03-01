from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_embed_single_input_returns_stub_response(client: TestClient) -> None:
    response = client.post("/embed", json={"inputs": ["first text"]})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]

    payload = response.json()
    assert payload["model"] == "stub-model"
    assert payload["revision"] == "milestone1-stub"
    assert payload["dim"] == 8
    assert payload["usage"] == {"tokens": 0}
    assert len(payload["data"]) == 1
    assert payload["data"][0]["index"] == 0
    assert len(payload["data"][0]["embedding"]) == payload["dim"]


def test_embed_multi_input_preserves_order_and_is_deterministic(client: TestClient) -> None:
    request_payload = {"inputs": ["alpha", "beta", "gamma"]}

    first_response = client.post("/embed", json=request_payload)
    second_response = client.post("/embed", json=request_payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_data = first_response.json()["data"]
    second_data = second_response.json()["data"]

    assert [item["index"] for item in first_data] == [0, 1, 2]
    assert first_data == second_data
    assert first_data[0]["embedding"] != first_data[1]["embedding"]


def test_embed_accepts_surrounding_whitespace_without_trimming(client: TestClient) -> None:
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
        app = create_app()
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
def test_embed_validation_failures_return_fastapi_422(
    client: TestClient,
    payload: dict[str, object],
    expected_loc: list[str],
) -> None:
    response = client.post("/embed", json=payload)

    assert response.status_code == 422
    assert response.headers["X-Request-ID"]
    body = response.json()
    assert "detail" in body
    assert body["detail"][0]["loc"] == expected_loc


def test_embed_respects_max_inputs_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_INPUTS_PER_REQUEST", "2")

    with TestClient(create_app()) as client:
        success_response = client.post("/embed", json={"inputs": ["one", "two"]})
        failure_response = client.post("/embed", json={"inputs": ["one", "two", "three"]})

    assert success_response.status_code == 200
    assert failure_response.status_code == 422
    assert failure_response.headers["X-Request-ID"]
    assert failure_response.json()["detail"][0]["loc"] == ["body", "inputs"]
