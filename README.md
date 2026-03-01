# EmbedServe

Milestone 1 exposes the HTTP API contract for an embedding service and returns deterministic stub embeddings.
Real model-backed transformer inference, GPU execution, batching, and model-level determinism work start in Milestone 2.

## Milestone 1 scope

- FastAPI service with `GET /healthz`, `GET /readyz`, `GET /metrics`, and `POST /embed`
- Structured JSON access logging with per-request `X-Request-ID`
- Prometheus metrics with custom app metrics plus default process and Python collectors
- Deterministic stub embeddings used to lock request and response schemas before model integration

## Current behavior

`POST /embed` does not run a tokenizer or a transformer in Milestone 1.
It hashes each input string into a fixed-size stub vector so clients can integrate against a stable contract while the real inference engine is still pending.

Readiness also reflects stub mode only.
`GET /readyz` returning `{"status":"ready","mode":"stub"}` means the HTTP surface is booted, not that a model or GPU is ready.

## Determinism note

Milestone 1 determinism is limited to the stub implementation.
It guarantees repeatable stub vectors for the same input within the same code version.
Model-backed repeatability and its tolerance rules are planned for later milestones.

## Configuration

All current configuration is environment-driven.

```env
MODEL_ID=stub-model
MODEL_REVISION=milestone1-stub
LOG_LEVEL=INFO
MAX_INPUTS_PER_REQUEST=64
```

## API

### `GET /healthz`

Returns:

```json
{
  "status": "ok"
}
```

### `GET /readyz`

Returns:

```json
{
  "status": "ready",
  "mode": "stub"
}
```

### `POST /embed`

Request body:

```json
{
  "inputs": ["first text", "second text"]
}
```

Rules:

- `inputs` is required and must be a list of strings
- Empty lists are rejected
- Empty or whitespace-only strings are rejected
- Extra top-level fields are rejected
- Validation failures return FastAPI-style `422` responses

Success response:

```json
{
  "data": [
    {
      "index": 0,
      "embedding": [0.123456, -0.456789, 0.789012, 0.111111, -0.222222, 0.333333, -0.444444, 0.555555]
    }
  ],
  "model": "stub-model",
  "revision": "milestone1-stub",
  "dim": 8,
  "usage": {
    "tokens": 0
  }
}
```

Every HTTP response includes an `X-Request-ID` header.

### `GET /metrics`

Exposes Prometheus text format with:

- `embedserve_http_requests_total`
- `embedserve_http_request_duration_seconds`
- `embedserve_app_ready`
- default process and Python runtime metrics

## Planned later milestones

The long-term direction remains a model-backed embedding server with pinned revisions, GPU inference, and batching.
Those capabilities are not shipped in Milestone 1 and should be treated as planned work.
