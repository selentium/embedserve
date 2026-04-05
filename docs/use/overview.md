# Overview

EmbedServe is a model-serving API focused on one job: turning text inputs into embeddings through a stable HTTP contract.

## Intended Use

- Internal services that need a simple embedding endpoint
- Local or single-host deployments that want reproducible model configuration
- Performance-sensitive workloads that benefit from always-on dynamic batching

## Current Behavior

- `POST /embed` runs tokenizer-backed transformer inference and returns one embedding per input.
- `GET /healthz` reports process liveness only.
- `GET /readyz` reports model-runtime readiness after successful initialization and warmup.
- `GET /metrics` exposes Prometheus metrics for HTTP requests, batching, failures, readiness, and CUDA memory telemetry.
- The service can start in an unready state instead of crashing when runtime-dependent initialization fails.

## Public Guarantees

- Request ordering is preserved in successful `/embed` responses.
- Validation is all-or-nothing for a request body.
- Dynamic batching is always enabled once the runtime is ready.
- The default model revision is pinned to an exact Hugging Face commit hash.
- Environment variables define the public runtime and batching surface.

## Important Boundaries

- Determinism is best-effort numerical stability, not bitwise identity.
- Embedding values are not portable across different hardware, drivers, or low-level runtime versions.
- The checked-in Docker flow assumes a local Linux host with NVIDIA GPU support.

## Next Reading

- Start with the [Quickstart](quickstart.md) if you want to run the service immediately.
- Use the [API Reference](api.md) and [Configuration Reference](configuration.md) for integration work.
- Use [Deployment](deployment.md) and [Operations and Monitoring](operations.md) for production-style operation on a single host.
