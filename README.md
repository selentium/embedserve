# EmbedServe

EmbedServe is a FastAPI embedding service that runs single-request Hugging Face encoder inference with a pinned model revision.

Milestone 2 replaces the stub embedder with a real tokenizer + transformer pipeline, keeps the request and success response schema stable, and makes readiness reflect actual model usability instead of simple process startup.

## Current behavior

- `POST /embed` runs tokenizer-backed transformer inference.
- `GET /healthz` is liveness only and always returns `200` while the process is up.
- `GET /readyz` returns `200` only after model initialization and warmup succeed.
- Inference is serialized per process. Batching is not implemented yet.
- If the configured model revision is not already cached, the first successful startup may download artifacts from Hugging Face.

## Determinism

The service uses standard inference hygiene only: `model.eval()` and `torch.inference_mode()`.
Milestone 2 does not guarantee strict reproducibility across hardware, drivers, or library versions.

## Configuration

All configuration is environment-driven.

```env
MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
MODEL_REVISION=826711e54e001c83835913827a843d8dd0a1def9
LOG_LEVEL=INFO
MAX_INPUTS_PER_REQUEST=64
DEVICE=cpu
DTYPE=float32
MAX_LENGTH=512
TRUNCATE=true
NORMALIZE_EMBEDDINGS=true
OUTPUT_DTYPE=float32
```

### Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL_ID` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformer style encoder model only. |
| `MODEL_REVISION` | `826711e54e001c83835913827a843d8dd0a1def9` | Must be an exact 40-character hexadecimal commit hash. |
| `LOG_LEVEL` | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`, `NOTSET`. |
| `MAX_INPUTS_PER_REQUEST` | `64` | Must be at least `1`. |
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `cuda:N`. |
| `DTYPE` | `float32` | `float32`, `float16`, `bfloat16`. |
| `MAX_LENGTH` | `512` | Upper bound for tokenizer length. Must be at least `1`. |
| `TRUNCATE` | `true` | If `true`, overlength inputs are truncated. If `false`, they fail with `422`. |
| `NORMALIZE_EMBEDDINGS` | `true` | If `true`, L2-normalize pooled embeddings. |
| `OUTPUT_DTYPE` | `float32` | Final CPU-side output precision before JSON serialization. `float32` or `float16`. |

### Length handling

- Effective max length is `min(MAX_LENGTH, tokenizer.model_max_length)` when the tokenizer exposes a sane finite cap.
- If `TRUNCATE=true`, each input is truncated independently before inference.
- If `TRUNCATE=false`, any input that exceeds the effective max length fails the whole request with FastAPI-style `422`.
- `usage.tokens` is the number of non-padding tokens actually fed into the model after truncation and special-token insertion.

## API

### `GET /healthz`

Returns:

```json
{
  "status": "ok"
}
```

### `GET /readyz`

Ready response:

```json
{
  "status": "ready",
  "mode": "model",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "revision": "826711e54e001c83835913827a843d8dd0a1def9",
  "device": "cpu",
  "dtype": "float32"
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
  "detail": "CUDA device cuda:0 is not available"
}
```

Possible `reason` values:

- `initialization_failed`
- `device_unavailable`
- `dtype_unsupported`
- `model_load_failed`
- `tokenizer_incompatible`
- `warmup_failed`

### `POST /embed`

Request body:

```json
{
  "inputs": ["first text", "second text"]
}
```

Rules:

- `inputs` is required and must be a list of strings.
- Empty lists are rejected.
- Empty or whitespace-only strings are rejected.
- Extra top-level fields are rejected.
- Validation failures return FastAPI-style `422`.
- If the runtime is unready, valid requests return `503`.

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

Unready response:

```json
{
  "detail": "Model is not ready: initialization_failed"
}
```

### `GET /metrics`

Exposes Prometheus text format with:

- `embedserve_http_requests_total`
- `embedserve_http_request_duration_seconds`
- `embedserve_app_ready`
- default process and Python runtime metrics

`embedserve_app_ready{mode="model"}` is `1` when the model runtime is ready and `0` otherwise.

## Install

Install the runtime dependencies from `requirements.txt`, then run the app with your preferred ASGI server.
