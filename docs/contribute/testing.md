# Testing and Verification

EmbedServe uses both ordinary automated tests and explicit live-server harnesses.

## Fast Local Checks

```bash
make lint
make typecheck
make test
make check
```

What they do:

- `make lint`: Ruff lint checks
- `make typecheck`: mypy strict type checks
- `make test`: pytest with coverage configured in `pyproject.toml`
- `make check`: lint + typecheck + test

## Test Coverage Areas

The automated suite covers:

- health and readiness behavior
- schema and validation rules
- single-request embedding semantics
- dynamic batching behavior
- determinism policy helpers
- metrics exposure and failure accounting
- benchmark and load-test harness argument parsing and result handling
- Docker artifact and dependency-locking expectations
- worktree helper behavior

## Live-Server Verification Commands

Determinism:

```bash
make verify-determinism
```

Batching acceptance:

```bash
make verify-batching
```

Benchmark:

```bash
make bench-10k BENCH_10K_ARGS="--json"
```

Sustained load:

```bash
make load-test LOAD_TEST_HARDWARE_ID="local-dev"
```

## When To Use Which

- Use `make test` for normal development changes.
- Use `make verify-determinism` when changing runtime, model, or determinism-related behavior.
- Use `make verify-batching` when changing queueing, batching limits, or ordering behavior.
- Use `make bench-10k` for performance comparisons across commits or settings.
- Use `make load-test` for stability checks under sustained concurrency.

## Acceptance Discipline

Every public behavior change should come with:

- automated test coverage when practical
- README or docs updates when the public contract changes
- report updates when benchmark or stability claims are refreshed
