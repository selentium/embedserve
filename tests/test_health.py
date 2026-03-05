from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.engine.embedder import (
    RuntimeInitializationError,
    _resolve_device,
    _validate_dtype_support,
)
from app.main import create_app
from app.runtime import RuntimeState, initialize_runtime
from app.settings import Settings
from tests.fakes import (
    FakeCuda,
    FakeTorch,
    make_fake_embedder,
    make_ready_runtime,
    make_unready_runtime,
)


def _ready_initializer(settings: Settings) -> RuntimeState:
    embedder, _, _ = make_fake_embedder(
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
    )
    return make_ready_runtime(
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
        embedder=embedder,
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(runtime_initializer=_ready_initializer)) as test_client:
        yield test_client


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_readyz_returns_model_runtime_details(client: TestClient) -> None:
    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "mode": "model",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "826711e54e001c83835913827a843d8dd0a1def9",
        "device": "cpu",
        "dtype": "float32",
    }
    assert response.headers["X-Request-ID"]


def test_readyz_returns_503_when_runtime_initialization_fails() -> None:
    def failing_initializer(settings: Settings) -> RuntimeState:
        raise RuntimeInitializationError(
            "device_unavailable",
            "CUDA device cuda:0 is not available",
        )

    with TestClient(create_app(runtime_initializer=failing_initializer)) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.headers["X-Request-ID"]
    assert response.json() == {
        "status": "not_ready",
        "mode": "model",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "826711e54e001c83835913827a843d8dd0a1def9",
        "device": "cpu",
        "dtype": "float32",
        "reason": "device_unavailable",
        "detail": "CUDA device cuda:0 is not available",
    }


def test_readyz_openapi_documents_503_response() -> None:
    with TestClient(create_app(runtime_initializer=_ready_initializer)) as client:
        response = client.get("/openapi.json")

    ready_operation = response.json()["paths"]["/readyz"]["get"]
    assert "503" in ready_operation["responses"]


def test_initialize_runtime_wraps_warmup_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()

    class FailingWarmupEmbedder:
        device = "cpu"
        dtype = "float32"

        def embed(self, inputs: list[str]) -> None:
            raise RuntimeInitializationError("model_load_failed", "forward pass exploded")

    monkeypatch.setattr(
        "app.runtime.build_transformer_embedder",
        lambda configured_settings: FailingWarmupEmbedder(),
    )

    with pytest.raises(RuntimeInitializationError) as exc_info:
        initialize_runtime(settings)

    assert exc_info.value.reason == "warmup_failed"
    assert exc_info.value.detail == "forward pass exploded"


def test_initialize_runtime_preserves_tokenizer_incompatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()

    class IncompatibleEmbedder:
        device = "cpu"
        dtype = "float32"

        def embed(self, inputs: list[str]) -> None:
            raise RuntimeInitializationError(
                "tokenizer_incompatible",
                "Model output did not include last_hidden_state",
            )

    monkeypatch.setattr(
        "app.runtime.build_transformer_embedder",
        lambda configured_settings: IncompatibleEmbedder(),
    )

    with pytest.raises(RuntimeInitializationError) as exc_info:
        initialize_runtime(settings)

    assert exc_info.value.reason == "tokenizer_incompatible"
    assert exc_info.value.detail == "Model output did not include last_hidden_state"


def test_unready_runtime_builder_shape() -> None:
    runtime = make_unready_runtime(
        model_id="model-id",
        revision="1" * 40,
        device="cuda:0",
        dtype="float16",
        reason="dtype_unsupported",
        detail="DTYPE float16 is not supported on cpu",
    )

    assert runtime.ready is False
    assert runtime.mode == "model"
    assert runtime.reason == "dtype_unsupported"
    assert runtime.embedder is None


def test_resolve_device_rejects_unavailable_cuda() -> None:
    with pytest.raises(RuntimeInitializationError) as exc_info:
        _resolve_device(FakeTorch(cuda=FakeCuda(available=False)), "cuda:0")

    assert exc_info.value.reason == "device_unavailable"


def test_validate_dtype_support_rejects_cpu_float16() -> None:
    with pytest.raises(RuntimeInitializationError) as exc_info:
        _validate_dtype_support(FakeTorch(), device="cpu", dtype="float16")

    assert exc_info.value.reason == "dtype_unsupported"
