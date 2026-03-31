from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.settings import Settings


def test_settings_accept_new_public_runtime_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure all public runtime settings, including batching and output controls, parse from environment variables."""
    monkeypatch.setenv("MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("MODEL_REVISION", "1" * 40)
    monkeypatch.setenv("DEVICE", "cuda:0")
    monkeypatch.setenv("DTYPE", "bfloat16")
    monkeypatch.setenv("MAX_LENGTH", "256")
    monkeypatch.setenv("TRUNCATE", "false")
    monkeypatch.setenv("NORMALIZE_EMBEDDINGS", "false")
    monkeypatch.setenv("OUTPUT_DTYPE", "float16")
    monkeypatch.setenv("MAX_INPUTS_PER_REQUEST", "8")
    monkeypatch.setenv("MAX_BATCH_SIZE", "16")
    monkeypatch.setenv("MAX_BATCH_TOKENS", "1024")
    monkeypatch.setenv("BATCH_TIMEOUT_MS", "5")
    monkeypatch.setenv("MAX_BATCH_QUEUE_SIZE", "128")
    monkeypatch.setenv("BATCH_REQUEST_TIMEOUT_MS", "2500")

    settings = Settings()

    assert settings.DEVICE == "cuda:0"
    assert settings.DTYPE == "bfloat16"
    assert settings.MAX_LENGTH == 256
    assert settings.TRUNCATE is False
    assert settings.NORMALIZE_EMBEDDINGS is False
    assert settings.OUTPUT_DTYPE == "float16"
    assert settings.MAX_INPUTS_PER_REQUEST == 8
    assert settings.MAX_BATCH_SIZE == 16
    assert settings.MAX_BATCH_TOKENS == 1024
    assert settings.BATCH_TIMEOUT_MS == 5
    assert settings.MAX_BATCH_QUEUE_SIZE == 128
    assert settings.BATCH_REQUEST_TIMEOUT_MS == 2500


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("DEVICE", "tpu"),
        ("DTYPE", "float64"),
        ("OUTPUT_DTYPE", "bfloat16"),
        ("MODEL_REVISION", "main"),
        ("MODEL_REVISION", "1234abcd"),
        ("MAX_LENGTH", "0"),
        ("MAX_BATCH_SIZE", "0"),
        ("MAX_BATCH_TOKENS", "0"),
        ("BATCH_TIMEOUT_MS", "0"),
        ("MAX_BATCH_QUEUE_SIZE", "0"),
        ("BATCH_REQUEST_TIMEOUT_MS", "0"),
    ],
)
def test_settings_reject_invalid_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    """Reject invalid enum values, malformed revisions, and zero-valued batching limits at settings validation time."""
    monkeypatch.setenv(field, value)

    with pytest.raises(ValidationError):
        Settings()
