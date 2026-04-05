# Architecture

## High-Level Flow

The service is built around a FastAPI app with four public endpoints:

- `/healthz`
- `/readyz`
- `/metrics`
- `/embed`

At startup the app:

1. loads validated settings
2. configures logging and metrics
3. applies the determinism policy
4. initializes the transformer runtime
5. performs a warmup embed
6. starts the dynamic batcher only if the runtime is ready

If runtime-dependent initialization fails, the process stays up in an unready state instead of crashing.

## Core Components

- `app/main.py`: ASGI app, middleware, lifespan, route handlers, and `/embed` orchestration
- `app/runtime.py`: runtime initialization and ready vs unready runtime state
- `app/engine/embedder.py`: tokenizer/model integration, preflight length checks, embedding generation, and CUDA memory sampling
- `app/batching.py`: async FIFO dynamic batcher, queueing, flush logic, cancellation, shutdown behavior, and fan-out
- `app/metrics.py`: Prometheus registry, metric definitions, and runtime metric refresh
- `app/settings.py`: public environment-backed configuration and validation rules

## Request Lifecycle for `POST /embed`

1. FastAPI validates the request schema.
2. The handler enforces `MAX_INPUTS_PER_REQUEST`.
3. If the runtime is unready, the handler returns `503`.
4. The embedder performs preflight token counting and length validation.
5. A submission ticket preserves FIFO admission order before batcher enqueue.
6. The batcher merges queued jobs subject to size, token, and timeout limits.
7. Batched inference runs in a dedicated single-worker threadpool.
8. Results are split back into per-request responses.

## Batching Model

- Batching is always on when the runtime is ready.
- Assembly is strict FIFO; there is no token-fit reordering.
- Flushes happen because of `max_batch_size`, `max_batch_tokens`, `timeout`, or `shutdown`.
- Queue saturation maps to `503 {"detail":"Batch queue is full"}`.
- Per-request timeout maps to `503 {"detail":"Embedding request timed out"}`.
- Shutdown drain maps to `503 {"detail":"Service is shutting down"}` for unresolved queued work.

## Readiness Model

- `/healthz` is process-only.
- `/readyz` reflects model-runtime state and exposes a closed `reason` enum when unready.
- The batcher only exists for a ready runtime.

## Metrics Ownership

The documented metric names and label values form a compatibility surface. If a change alters metric names, labels, or semantics, update the public docs and tests in the same change.
