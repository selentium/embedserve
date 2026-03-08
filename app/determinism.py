from __future__ import annotations

import random
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class DeterminismPolicyState:
    mode: Literal["numerical_stability"]
    seed: int
    deterministic_algorithms: Literal["enabled", "unsupported"]


def apply_determinism_policy(
    *, seed: int = 0, torch_module: Any | None = None
) -> DeterminismPolicyState:
    random.seed(seed)

    if torch_module is None:
        import torch as imported_torch

        torch_module = imported_torch

    _try_torch_seed(torch_module, seed=seed)
    _try_cuda_seed(torch_module, seed=seed)
    _try_set_cudnn_flags(torch_module)
    deterministic_algorithms = _try_enable_deterministic_algorithms(torch_module)

    return DeterminismPolicyState(
        mode="numerical_stability",
        seed=seed,
        deterministic_algorithms=deterministic_algorithms,
    )


def _try_torch_seed(torch_module: Any, *, seed: int) -> None:
    manual_seed = getattr(torch_module, "manual_seed", None)
    if callable(manual_seed):
        try:
            manual_seed(seed)
        except Exception:
            return


def _try_cuda_seed(torch_module: Any, *, seed: int) -> None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return

    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return

    try:
        cuda_available = bool(is_available())
    except Exception:
        return

    if not cuda_available:
        return

    manual_seed_all = getattr(cuda, "manual_seed_all", None)
    if callable(manual_seed_all):
        try:
            manual_seed_all(seed)
        except Exception:
            return


def _try_set_cudnn_flags(torch_module: Any) -> None:
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return

    cudnn = getattr(backends, "cudnn", None)
    if cudnn is None:
        return

    with suppress(Exception):
        cudnn.deterministic = True

    with suppress(Exception):
        cudnn.benchmark = False


def _try_enable_deterministic_algorithms(
    torch_module: Any,
) -> Literal["enabled", "unsupported"]:
    use_deterministic_algorithms = getattr(torch_module, "use_deterministic_algorithms", None)
    if not callable(use_deterministic_algorithms):
        return "unsupported"

    try:
        use_deterministic_algorithms(True, warn_only=True)
        return "enabled"
    except TypeError:
        try:
            use_deterministic_algorithms(True)
            return "enabled"
        except Exception:
            return "unsupported"
    except Exception:
        return "unsupported"
