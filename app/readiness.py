from __future__ import annotations

from typing import Literal, get_args

ReadyReason = Literal[
    "initialization_failed",
    "device_unavailable",
    "dtype_unsupported",
    "model_load_failed",
    "tokenizer_incompatible",
    "warmup_failed",
]

READY_REASON_VALUES = frozenset(get_args(ReadyReason))
