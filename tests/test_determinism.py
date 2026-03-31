from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.determinism import apply_determinism_policy


class FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self._available = available
        self.manual_seed_all_calls: list[int] = []

    def is_available(self) -> bool:
        return self._available

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_all_calls.append(seed)


class FakeTorch:
    def __init__(self, *, cuda_available: bool = False, algorithm_mode: str = "warn_only") -> None:
        self.manual_seed_calls: list[int] = []
        self.cuda = FakeCuda(available=cuda_available)
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))
        self.algorithm_calls: list[tuple[Any, ...]] = []
        self._algorithm_mode = algorithm_mode

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def use_deterministic_algorithms(self, enabled: bool, warn_only: bool = False) -> None:
        self.algorithm_calls.append((enabled, warn_only))
        if self._algorithm_mode == "fail":
            msg = "not supported"
            raise RuntimeError(msg)


class FakeTorchNoWarnOnly:
    def __init__(self) -> None:
        self.manual_seed_calls: list[int] = []
        self.cuda = FakeCuda(available=False)

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def use_deterministic_algorithms(self, enabled: bool) -> None:
        if not enabled:
            msg = "Expected enabled=True"
            raise AssertionError(msg)


def test_apply_determinism_policy_sets_seed_and_torch_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed Python and Torch state while enabling the deterministic CuDNN and algorithm flags when available."""
    random_calls: list[int] = []
    monkeypatch.setattr("app.determinism.random.seed", lambda seed: random_calls.append(seed))

    torch_module = FakeTorch(cuda_available=True)
    state = apply_determinism_policy(seed=7, torch_module=torch_module)

    assert random_calls == [7]
    assert torch_module.manual_seed_calls == [7]
    assert torch_module.cuda.manual_seed_all_calls == [7]
    assert torch_module.backends.cudnn.deterministic is True
    assert torch_module.backends.cudnn.benchmark is False
    assert torch_module.algorithm_calls == [(True, True)]

    assert state.mode == "numerical_stability"
    assert state.seed == 7
    assert state.deterministic_algorithms == "enabled"


def test_apply_determinism_policy_gracefully_handles_missing_optional_apis() -> None:
    """Treat missing Torch determinism hooks as unsupported instead of crashing the policy application."""
    state = apply_determinism_policy(torch_module=SimpleNamespace())

    assert state.mode == "numerical_stability"
    assert state.seed == 0
    assert state.deterministic_algorithms == "unsupported"


def test_apply_determinism_policy_supports_older_torch_signature() -> None:
    """Support older Torch versions whose `use_deterministic_algorithms` API lacks the `warn_only` parameter."""
    torch_module = FakeTorchNoWarnOnly()

    state = apply_determinism_policy(torch_module=torch_module)

    assert torch_module.manual_seed_calls == [0]
    assert state.deterministic_algorithms == "enabled"


def test_apply_determinism_policy_treats_algorithm_errors_as_unsupported() -> None:
    """Downgrade runtime errors from deterministic algorithm enablement to an `unsupported` capability state."""
    torch_module = FakeTorch(algorithm_mode="fail")

    state = apply_determinism_policy(torch_module=torch_module)

    assert state.deterministic_algorithms == "unsupported"
