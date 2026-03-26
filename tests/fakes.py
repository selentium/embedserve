from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from math import sqrt
from types import SimpleNamespace
from typing import Any

from app.engine.embedder import TransformerEmbedder
from app.readiness import ReadyReason
from app.runtime import RuntimeState


def _is_sequence(value: Any) -> bool:
    return isinstance(value, list)


def _clone(value: Any) -> Any:
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _apply(value: Any, func: Any) -> Any:
    if isinstance(value, list):
        return [_apply(item, func) for item in value]
    return func(value)


def _binary_op(left: Any, right: Any, operator: Any) -> Any:
    if not _is_sequence(left) and not _is_sequence(right):
        return operator(left, right)

    if _is_sequence(left) and _is_sequence(right):
        if len(left) == len(right):
            return [
                _binary_op(left_item, right_item, operator)
                for left_item, right_item in zip(left, right, strict=True)
            ]
        if len(left) == 1:
            return [_binary_op(left[0], right_item, operator) for right_item in right]
        if len(right) == 1:
            return [_binary_op(left_item, right[0], operator) for left_item in left]
        msg = "Unsupported broadcast shape"
        raise ValueError(msg)

    if _is_sequence(left):
        return [_binary_op(item, right, operator) for item in left]

    return [_binary_op(left, item, operator) for item in right]


def _sum_rows(rows: list[Any]) -> Any:
    total = _clone(rows[0])
    for row in rows[1:]:
        total = _binary_op(total, row, lambda left, right: left + right)
    return total


class FakeTensor:
    def __init__(self, data: Any, *, dtype: str = "float32", device: str = "cpu") -> None:
        self.data = _clone(data)
        self.dtype = dtype
        self.device = device

    def to(self, device: str | None = None, dtype: str | None = None) -> FakeTensor:
        converted = _clone(self.data)
        next_dtype = self.dtype if dtype is None else dtype
        if dtype is not None:
            precision = {
                "float32": 6,
                "float16": 3,
                "bfloat16": 2,
            }.get(dtype, 6)
            converted = _apply(converted, lambda value: round(float(value), precision))

        return FakeTensor(
            converted,
            dtype=next_dtype,
            device=self.device if device is None else device,
        )

    def sum(self, dim: int) -> FakeTensor:
        if dim != 1:
            msg = f"Unsupported sum dim: {dim}"
            raise ValueError(msg)

        if not isinstance(self.data, list) or not self.data:
            return FakeTensor([], dtype=self.dtype, device=self.device)

        first_item = self.data[0]
        if not isinstance(first_item, list):
            msg = "Unsupported tensor rank"
            raise ValueError(msg)

        if first_item and isinstance(first_item[0], list):
            return FakeTensor(
                [_sum_rows(batch_rows) for batch_rows in self.data],
                dtype=self.dtype,
                device=self.device,
            )

        return FakeTensor(
            [sum(float(value) for value in row) for row in self.data],
            dtype=self.dtype,
            device=self.device,
        )

    def unsqueeze(self, dim: int) -> FakeTensor:
        if dim != -1:
            msg = f"Unsupported unsqueeze dim: {dim}"
            raise ValueError(msg)
        return FakeTensor(
            _apply(self.data, lambda value: [value]),
            dtype=self.dtype,
            device=self.device,
        )

    def clamp(self, *, min: float) -> FakeTensor:
        return FakeTensor(
            _apply(self.data, lambda value: value if value >= min else min),
            dtype=self.dtype,
            device=self.device,
        )

    def tolist(self) -> Any:
        return _clone(self.data)

    def __mul__(self, other: Any) -> FakeTensor:
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(
            _binary_op(self.data, other_data, lambda left, right: left * right),
            dtype=self.dtype,
            device=self.device,
        )

    def __truediv__(self, other: Any) -> FakeTensor:
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(
            _binary_op(self.data, other_data, lambda left, right: left / right),
            dtype=self.dtype,
            device=self.device,
        )


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = False,
        device_count: int = 0,
        bf16_supported: bool = False,
        memory_allocated_bytes: int = 0,
        memory_reserved_bytes: int = 0,
    ) -> None:
        self._available = available
        self._device_count = device_count
        self._bf16_supported = bf16_supported
        self._memory_allocated_bytes = memory_allocated_bytes
        self._memory_reserved_bytes = memory_reserved_bytes
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._device_count

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported

    def memory_allocated(self, device: int) -> int:
        if device >= self._device_count:
            msg = f"Unsupported device index: {device}"
            raise ValueError(msg)
        return self._memory_allocated_bytes

    def memory_reserved(self, device: int) -> int:
        if device >= self._device_count:
            msg = f"Unsupported device index: {device}"
            raise ValueError(msg)
        return self._memory_reserved_bytes

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self, *, cuda: FakeCuda | None = None) -> None:
        class FakeOutOfMemoryError(RuntimeError):
            pass

        self.cuda = cuda or FakeCuda()
        self.nn = SimpleNamespace(functional=SimpleNamespace(normalize=self._normalize))
        self.OutOfMemoryError = FakeOutOfMemoryError

    def inference_mode(self) -> Any:
        return nullcontext()

    def _normalize(self, tensor: FakeTensor, *, p: int, dim: int, eps: float) -> FakeTensor:
        if p != 2 or dim != 1:
            msg = "Unsupported normalization configuration"
            raise ValueError(msg)

        normalized_rows: list[list[float]] = []
        for row in tensor.tolist():
            norm = sqrt(sum(float(value) ** 2 for value in row))
            scale = norm if norm > eps else 1.0
            normalized_rows.append([float(value) / scale for value in row])
        return FakeTensor(normalized_rows, dtype=tensor.dtype, device=tensor.device)


@dataclass
class FakeTokenizer:
    token_map: dict[str, list[int]]
    model_max_length: int = 512
    padded_batch_calls: int = 0

    def __call__(
        self,
        texts: str | list[str],
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: bool = False,
        return_length: bool = False,
        return_tensors: str | None = None,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(texts, str):
            is_single = True
            batch: list[str] = [texts]
        else:
            is_single = False
            batch = list(texts)
        token_ids = [self._tokenize(text, add_special_tokens=add_special_tokens) for text in batch]

        if truncation and max_length is not None:
            token_ids = [tokens[:max_length] for tokens in token_ids]

        if return_length:
            return {"length": [len(tokens) for tokens in token_ids]}

        if padding:
            self.padded_batch_calls += 1
            width = max(len(tokens) for tokens in token_ids)
            padded_ids = [tokens + [0] * (width - len(tokens)) for tokens in token_ids]
            masks = [[1] * len(tokens) + [0] * (width - len(tokens)) for tokens in token_ids]
            if return_tensors == "pt":
                return {
                    "input_ids": FakeTensor(padded_ids, dtype="int64"),
                    "attention_mask": FakeTensor(masks, dtype="int64"),
                }
            return {"input_ids": padded_ids, "attention_mask": masks}

        if is_single:
            return {
                "input_ids": token_ids[0],
                "attention_mask": [1] * len(token_ids[0]),
            }

        return {
            "input_ids": token_ids,
            "attention_mask": [[1] * len(tokens) for tokens in token_ids],
        }

    def _tokenize(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if text in self.token_map:
            base_ids = list(self.token_map[text])
        else:
            base_ids = list(range(1, len(text.split()) + 1))

        if not add_special_tokens:
            return base_ids

        return [101, *base_ids, 102]


class FakeModel:
    def __init__(self, *, include_last_hidden_state: bool = True) -> None:
        self.include_last_hidden_state = include_last_hidden_state
        self.device = "cpu"
        self.forward_calls = 0
        self.eval_called = False

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **model_inputs: Any) -> Any:
        self.forward_calls += 1
        input_ids = model_inputs["input_ids"].tolist()
        hidden_state = [
            [[float(token_id), float(token_id) + 0.5, float(token_id) - 0.25] for token_id in row]
            for row in input_ids
        ]

        if not self.include_last_hidden_state:
            return SimpleNamespace()

        return SimpleNamespace(
            last_hidden_state=FakeTensor(
                hidden_state,
                dtype="float32",
                device=self.device,
            )
        )


def make_fake_embedder(
    *,
    model_id: str,
    revision: str,
    truncate: bool = True,
    normalize_embeddings: bool = True,
    output_dtype: str = "float32",
    effective_max_length: int = 8,
    token_map: dict[str, list[int]] | None = None,
    tokenizer_max_length: int = 512,
    include_last_hidden_state: bool = True,
) -> tuple[TransformerEmbedder, FakeTokenizer, FakeModel]:
    torch_module = FakeTorch()
    tokenizer = FakeTokenizer(token_map=token_map or {}, model_max_length=tokenizer_max_length)
    model = FakeModel(include_last_hidden_state=include_last_hidden_state)
    model.eval()

    embedder = TransformerEmbedder(
        model_id=model_id,
        revision=revision,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch_module,
        device="cpu",
        dtype="float32",
        effective_max_length=effective_max_length,
        truncate=truncate,
        normalize_embeddings=normalize_embeddings,
        output_dtype=output_dtype,
        output_torch_dtype=getattr(torch_module, output_dtype),
    )
    return embedder, tokenizer, model


def make_ready_runtime(
    *,
    model_id: str,
    revision: str,
    embedder: TransformerEmbedder,
) -> RuntimeState:
    return RuntimeState(
        ready=True,
        mode="model",
        model_id=model_id,
        revision=revision,
        device=embedder.device,
        dtype=embedder.dtype,
        reason=None,
        detail=None,
        embedder=embedder,
    )


def make_unready_runtime(
    *,
    model_id: str,
    revision: str,
    device: str = "cpu",
    dtype: str = "float32",
    reason: ReadyReason = "initialization_failed",
    detail: str = "Initialization failed",
) -> RuntimeState:
    return RuntimeState(
        ready=False,
        mode="model",
        model_id=model_id,
        revision=revision,
        device=device,
        dtype=dtype,
        reason=reason,
        detail=detail,
        embedder=None,
    )
