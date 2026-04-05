# Configuration Reference

All runtime configuration is environment-driven.

## Runtime Variables

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL_ID` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformer style encoder model only. |
| `MODEL_REVISION` | `826711e54e001c83835913827a843d8dd0a1def9` | Must be an exact 40-character hexadecimal commit hash. |
| `LOG_LEVEL` | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`, `NOTSET`. |
| `MAX_INPUTS_PER_REQUEST` | `64` | Must be at least `1`. |
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `cuda:N`. Docker Compose defaults this to `cuda` when unset in `.env`. |
| `DTYPE` | `float32` | `float32`, `float16`, `bfloat16`. |
| `MAX_LENGTH` | `512` | Upper bound for tokenizer length. Must be at least `1`. |
| `TRUNCATE` | `true` | If `true`, overlength inputs are truncated. If `false`, they fail with `422`. |
| `NORMALIZE_EMBEDDINGS` | `true` | If `true`, L2-normalize pooled embeddings. |
| `OUTPUT_DTYPE` | `float32` | Final CPU-side output precision before JSON serialization. `float32` or `float16`. |
| `MAX_BATCH_SIZE` | `128` | Maximum number of requests merged into one batch flush. Must be at least `1`. |
| `MAX_BATCH_TOKENS` | `8192` | Maximum summed post-truncation token count merged into one batch flush. Must be at least `1`. |
| `BATCH_TIMEOUT_MS` | `2` | Maximum wait to assemble a shared batch before flush. Must be at least `1`. |
| `MAX_BATCH_QUEUE_SIZE` | `1024` | Maximum queued requests before overload rejection. Must be at least `1`. |
| `BATCH_REQUEST_TIMEOUT_MS` | `5000` | Per-request wait timeout while awaiting a batch result. Must be at least `1`. |
| `EMBEDSERVE_PORT` | `8000` | Used by the checked-in Docker flow for port publishing. |

## Length Handling

- Effective max length is `min(MAX_LENGTH, tokenizer.model_max_length)` when the tokenizer exposes a sane finite cap.
- If `TRUNCATE=true`, each input is truncated independently before inference.
- If `TRUNCATE=false`, any input that exceeds the effective max length fails the whole request with `422`.
- `usage.tokens` reports the number of non-padding tokens actually fed into the model after truncation and special-token insertion.

## Validation Boundaries

Startup fails immediately for invalid static configuration such as:

- unsupported `LOG_LEVEL`
- unsupported `DEVICE`
- unsupported `DTYPE`
- unsupported `OUTPUT_DTYPE`
- malformed `MODEL_REVISION`
- numeric values below minimum constraints

The service can stay running but unready for runtime-dependent failures such as:

- unavailable CUDA device
- unsupported dtype/device combination
- model download or load failure
- tokenizer incompatibility
- warmup failure

## Local `.env`

The tracked `.env.example` captures the common local variables. The Docker Compose file has its own defaults, so leaving `DEVICE` unset means:

- local `make start` uses the application default `cpu`
- Docker Compose falls back to `cuda`

## Related Docs

- See [Deployment](deployment.md) for Docker-specific defaults and startup notes.
- See [API Reference](api.md) for how configuration choices affect request behavior.
