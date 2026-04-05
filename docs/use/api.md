# API Reference

## Common Behavior

- All endpoints include an `X-Request-ID` response header.
- Validation errors use FastAPI-style `422` responses.
- Operational failures on `/embed` use FastAPI-style error payloads with a `detail` string.

## `GET /healthz`

Purpose:

- Process liveness only

Response:

```json
{
  "status": "ok"
}
```

## `GET /readyz`

Ready response:

```json
{
  "status": "ready",
  "mode": "model",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "revision": "826711e54e001c83835913827a843d8dd0a1def9",
  "device": "cpu",
  "dtype": "float32",
  "tokenization": {
    "max_length": 512,
    "truncate": true
  },
  "batching": {
    "max_batch_size": 128,
    "max_batch_tokens": 8192,
    "batch_timeout_ms": 2,
    "max_batch_queue_size": 1024,
    "batch_request_timeout_ms": 5000
  }
}
```

Unready response:

```json
{
  "status": "not_ready",
  "mode": "model",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "revision": "826711e54e001c83835913827a843d8dd0a1def9",
  "device": "cuda:0",
  "dtype": "float16",
  "reason": "device_unavailable",
  "detail": "CUDA device cuda:0 is not available",
  "tokenization": {
    "max_length": 512,
    "truncate": true
  },
  "batching": {
    "max_batch_size": 128,
    "max_batch_tokens": 8192,
    "batch_timeout_ms": 2,
    "max_batch_queue_size": 1024,
    "batch_request_timeout_ms": 5000
  }
}
```

Possible `reason` values:

- `initialization_failed`
- `device_unavailable`
- `dtype_unsupported`
- `model_load_failed`
- `tokenizer_incompatible`
- `warmup_failed`

## `POST /embed`

Request body:

```json
{
  "inputs": ["first text", "second text"]
}
```

Request rules:

- `inputs` is required and must be a list of strings.
- Empty lists are rejected.
- Empty or whitespace-only strings are rejected.
- Extra top-level fields are rejected.
- The list length must not exceed `MAX_INPUTS_PER_REQUEST`.
- Validation failures reject the whole request.

Success response:

```json
{
  "data": [
    {
      "index": 0,
      "embedding": [0.1, 0.2, 0.3]
    }
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "revision": "826711e54e001c83835913827a843d8dd0a1def9",
  "dim": 384,
  "usage": {
    "tokens": 8
  }
}
```

Response semantics:

- `data` preserves request order through the `index` field.
- `dim` is the embedding width returned by the configured model.
- `usage.tokens` is the total number of non-padding tokens actually processed after truncation and special-token insertion.
- Embedding values are JSON numbers regardless of `OUTPUT_DTYPE`.

Operational `503` responses:

Unready runtime:

```json
{
  "detail": "Model is not ready: initialization_failed"
}
```

Queue full:

```json
{
  "detail": "Batch queue is full"
}
```

Request timeout:

```json
{
  "detail": "Embedding request timed out"
}
```

Shutdown drain:

```json
{
  "detail": "Service is shutting down"
}
```

Internal inference failure:

```json
{
  "detail": "Embedding inference failed"
}
```

## `GET /metrics`

This endpoint exposes Prometheus text format.

Documented custom metric names:

- `embedserve_http_requests_total`
- `embedserve_http_request_duration_seconds`
- `embedserve_app_ready`
- `embedserve_batch_queue_depth`
- `embedserve_batch_queue_wait_seconds`
- `embedserve_batch_size`
- `embedserve_batch_token_count`
- `embedserve_batch_flush_total`
- `embedserve_batch_overload_rejections_total`
- `embedserve_batch_request_timeouts_total`
- `embedserve_batch_request_cancellations_total`
- `embedserve_batch_shutdown_rejections_total`
- `embedserve_batch_inference_failures_total`
- `embedserve_gpu_memory_allocated_bytes`
- `embedserve_gpu_memory_reserved_bytes`
- `embedserve_gpu_oom_total`
- `embedserve_request_failures_total`
- `embedserve_unhandled_exceptions_total`

Notes:

- `embedserve_app_ready{mode="model"}` is `1` when the runtime is ready and `0` otherwise.
- GPU memory series are emitted only for CUDA runtimes.
- Default process and Python runtime metrics are also included.

For operational interpretation, see [Operations and Monitoring](operations.md).
