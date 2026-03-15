from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any, Protocol

from fastapi.exceptions import RequestValidationError

from app.readiness import READY_REASON_VALUES, ReadyReason
from app.schemas import EmbeddingItem, EmbedResponse, UsageInfo
from app.settings import Settings

_UNBOUNDED_TOKENIZER_LIMIT = 1_000_000
_NORMALIZE_EPSILON = 1e-12


class Embedder(Protocol):
    device: str
    dtype: str

    def preflight(self, inputs: list[str]) -> list[int]: ...

    def embed(self, inputs: list[str]) -> EmbedResponse: ...


class RuntimeInitializationError(Exception):
    def __init__(self, reason: ReadyReason, detail: str) -> None:
        if reason not in READY_REASON_VALUES:
            msg = f"Unsupported readiness reason: {reason}"
            raise ValueError(msg)

        self.reason = reason
        self.detail = _sanitize_detail(detail)
        super().__init__(self.detail)


class TransformerEmbedder:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        device: str,
        dtype: str,
        effective_max_length: int,
        truncate: bool,
        normalize_embeddings: bool,
        output_dtype: str,
        output_torch_dtype: Any,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self.device = device
        self.dtype = dtype
        self._effective_max_length = effective_max_length
        self._truncate = truncate
        self._normalize_embeddings = normalize_embeddings
        self._output_dtype = output_dtype
        self._output_torch_dtype = output_torch_dtype
        self._lock = threading.Lock()

    def preflight(self, inputs: list[str]) -> list[int]:
        with self._lock:
            lengths = self._measure_lengths(inputs)
            if not self._truncate:
                for index, length in enumerate(lengths):
                    if length > self._effective_max_length:
                        raise _input_too_long_error(
                            index=index,
                            actual=length,
                            limit=self._effective_max_length,
                            value=inputs[index],
                        )
                return lengths

            return [min(length, self._effective_max_length) for length in lengths]

    def embed(self, inputs: list[str]) -> EmbedResponse:
        with self._lock:
            if not self._truncate:
                self._validate_lengths(inputs)

            encoded_inputs = self._tokenize_batch(inputs)
            attention_mask = encoded_inputs.get("attention_mask")
            if attention_mask is None:
                raise RuntimeInitializationError(
                    "tokenizer_incompatible",
                    "Tokenizer output did not include attention_mask",
                )

            token_counts = [int(length) for length in attention_mask.sum(dim=1).tolist()]
            model_inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded_inputs.items()
            }

            with self._torch.inference_mode():
                outputs = self._model(**model_inputs)

            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                raise RuntimeInitializationError(
                    "tokenizer_incompatible",
                    "Model output did not include last_hidden_state",
                )

            pooled = self._mean_pool(
                last_hidden_state=last_hidden_state,
                attention_mask=model_inputs["attention_mask"],
            )

            if self._normalize_embeddings:
                pooled = self._torch.nn.functional.normalize(
                    pooled,
                    p=2,
                    dim=1,
                    eps=_NORMALIZE_EPSILON,
                )

            output_tensor = pooled.to(device="cpu", dtype=self._output_torch_dtype)
            embeddings = [
                [float(value) for value in embedding] for embedding in output_tensor.tolist()
            ]
            dim = len(embeddings[0]) if embeddings else 0

            return EmbedResponse(
                data=[
                    EmbeddingItem(index=index, embedding=embedding)
                    for index, embedding in enumerate(embeddings)
                ],
                model=self._model_id,
                revision=self._revision,
                dim=dim,
                usage=UsageInfo(tokens=sum(token_counts)),
            )

    def _tokenize_batch(self, inputs: list[str]) -> Mapping[str, Any]:
        try:
            encoded = self._tokenizer(
                inputs,
                padding=True,
                truncation=self._truncate,
                max_length=self._effective_max_length,
                return_tensors="pt",
            )
        except Exception as exc:
            raise RuntimeInitializationError("tokenizer_incompatible", str(exc)) from exc

        if not isinstance(encoded, Mapping):
            raise RuntimeInitializationError(
                "tokenizer_incompatible",
                "Tokenizer returned an invalid batch payload",
            )
        return encoded

    def _validate_lengths(self, inputs: list[str]) -> None:
        lengths = self._measure_lengths(inputs)
        for index, length in enumerate(lengths):
            if length > self._effective_max_length:
                raise _input_too_long_error(
                    index=index,
                    actual=length,
                    limit=self._effective_max_length,
                    value=inputs[index],
                )

    def _measure_lengths(self, inputs: list[str]) -> list[int]:
        try:
            encoded = self._tokenizer(
                inputs,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_length=True,
            )
        except TypeError:
            return [self._measure_single_length(text) for text in inputs]
        except Exception as exc:
            raise RuntimeInitializationError("tokenizer_incompatible", str(exc)) from exc

        if isinstance(encoded, Mapping):
            lengths = encoded.get("length")
            if lengths is not None:
                return [int(length) for length in lengths]

        return [self._measure_single_length(text) for text in inputs]

    def _measure_single_length(self, text: str) -> int:
        try:
            encoded = self._tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
        except Exception as exc:
            raise RuntimeInitializationError("tokenizer_incompatible", str(exc)) from exc

        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise RuntimeInitializationError(
                "tokenizer_incompatible",
                "Tokenizer output did not include input_ids",
            )

        input_ids = encoded["input_ids"]
        return len(input_ids)

    def _mean_pool(self, *, last_hidden_state: Any, attention_mask: Any) -> Any:
        expanded_mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        masked_hidden_state = last_hidden_state * expanded_mask
        summed_hidden_state = masked_hidden_state.sum(dim=1)
        token_totals = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return summed_hidden_state / token_totals


def load_transformer_components(
    *,
    model_id: str,
    revision: str,
    torch_dtype: Any,
) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_fast=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RuntimeInitializationError("tokenizer_incompatible", str(exc)) from exc

    try:
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        raise RuntimeInitializationError("model_load_failed", str(exc)) from exc

    return tokenizer, model


def build_transformer_embedder(
    settings: Settings,
    *,
    loader: Any = load_transformer_components,
) -> TransformerEmbedder:
    import torch

    device = _resolve_device(torch, settings.DEVICE)
    model_dtype = _resolve_model_dtype(torch, settings.DTYPE)
    output_dtype = _resolve_output_dtype(torch, settings.OUTPUT_DTYPE)
    _validate_dtype_support(torch, device=device, dtype=settings.DTYPE)

    tokenizer, model = loader(
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
        torch_dtype=model_dtype,
    )

    try:
        model = model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeInitializationError("model_load_failed", str(exc)) from exc

    effective_max_length = _resolve_effective_max_length(
        configured_max_length=settings.MAX_LENGTH,
        tokenizer_max_length=getattr(tokenizer, "model_max_length", None),
    )

    return TransformerEmbedder(
        model_id=settings.MODEL_ID,
        revision=settings.MODEL_REVISION,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
        device=device,
        dtype=settings.DTYPE,
        effective_max_length=effective_max_length,
        truncate=settings.TRUNCATE,
        normalize_embeddings=settings.NORMALIZE_EMBEDDINGS,
        output_dtype=settings.OUTPUT_DTYPE,
        output_torch_dtype=output_dtype,
    )


def _resolve_device(torch_module: Any, configured_device: str) -> str:
    if configured_device == "cpu":
        return "cpu"

    if not torch_module.cuda.is_available():
        raise RuntimeInitializationError(
            "device_unavailable",
            f"CUDA device {configured_device} is not available",
        )

    if configured_device == "cuda":
        return "cuda"

    device_index = int(configured_device.split(":", maxsplit=1)[1])
    if device_index >= int(torch_module.cuda.device_count()):
        raise RuntimeInitializationError(
            "device_unavailable",
            f"CUDA device {configured_device} is not available",
        )

    return configured_device


def _resolve_model_dtype(torch_module: Any, configured_dtype: str) -> Any:
    try:
        return getattr(torch_module, configured_dtype)
    except AttributeError as exc:
        raise RuntimeInitializationError(
            "dtype_unsupported",
            f"DTYPE {configured_dtype} is not supported",
        ) from exc


def _resolve_output_dtype(torch_module: Any, configured_dtype: str) -> Any:
    try:
        return getattr(torch_module, configured_dtype)
    except AttributeError as exc:
        raise RuntimeInitializationError(
            "dtype_unsupported",
            f"OUTPUT_DTYPE {configured_dtype} is not supported",
        ) from exc


def _validate_dtype_support(torch_module: Any, *, device: str, dtype: str) -> None:
    if device == "cpu" and dtype == "float16":
        raise RuntimeInitializationError(
            "dtype_unsupported",
            "DTYPE float16 is not supported on cpu",
        )

    if device.startswith("cuda") and dtype == "bfloat16":
        is_supported = getattr(torch_module.cuda, "is_bf16_supported", None)
        if callable(is_supported) and not is_supported():
            raise RuntimeInitializationError(
                "dtype_unsupported",
                f"DTYPE {dtype} is not supported on {device}",
            )


def _resolve_effective_max_length(
    *,
    configured_max_length: int,
    tokenizer_max_length: Any,
) -> int:
    if not isinstance(tokenizer_max_length, int):
        return configured_max_length

    if tokenizer_max_length <= 0 or tokenizer_max_length >= _UNBOUNDED_TOKENIZER_LIMIT:
        return configured_max_length

    return min(configured_max_length, tokenizer_max_length)


def _sanitize_detail(detail: str) -> str:
    sanitized = " ".join(detail.strip().split())
    if not sanitized:
        return "Initialization failed"
    return sanitized[:240]


def _input_too_long_error(
    *,
    index: int,
    actual: int,
    limit: int,
    value: str,
) -> RequestValidationError:
    return RequestValidationError(
        [
            {
                "type": "too_long",
                "loc": ("body", "inputs", index),
                "msg": f"Input token length {actual} exceeds limit {limit}",
                "input": value,
                "ctx": {"actual_length": actual, "max_length": limit},
            }
        ]
    )
