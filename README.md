# EmbedServe (Milestone 5)

EmbedServe is a FastAPI embedding service that runs dynamically batched Hugging Face encoder inference with a pinned model revision.

Milestone 5 keeps the Milestone 4 API and batching behavior, adds a Dockerized GPU deployment workflow, pins the Python dependency set, and documents the host prerequisites required to reproduce the runtime stack on a clean machine.

## Current behavior

- `POST /embed` runs tokenizer-backed transformer inference.
- `GET /healthz` is liveness only and always returns `200` while the process is up.
- `GET /readyz` returns `200` only after model initialization and warmup succeed.
- Inference is always dynamically batched in strict FIFO order while preserving per-request response semantics.
- If the configured model revision is not already cached, the first successful startup may download artifacts from Hugging Face.

## Determinism

EmbedServe applies a best-effort numerical-stability policy at startup:

- `random.seed(0)`
- `torch.manual_seed(0)`
- `torch.cuda.manual_seed_all(0)` when CUDA is available
- `torch.backends.cudnn.deterministic=True`
- `torch.backends.cudnn.benchmark=False`
- `torch.use_deterministic_algorithms(True, warn_only=True)` when supported

Policy boundary:

- The service targets bounded numerical drift under a fixed environment.
- It does not guarantee bitwise-identical outputs.
- Results are not portable across hardware, drivers, PyTorch/transformers versions, or other low-level runtime changes.

For meaningful verification runs, keep all of the following fixed:

- same host and accelerator
- same `MODEL_ID` and `MODEL_REVISION`
- same runtime settings (`DEVICE`, `DTYPE`, `MAX_LENGTH`, `TRUNCATE`, `NORMALIZE_EMBEDDINGS`, `OUTPUT_DTYPE`)
- warmed runtime before measurement

Run live verification against a started server:

```bash
make verify-determinism
```

Intentional failure example (impossible cosine threshold > 1.0) to confirm non-zero exit:

```bash
make verify-determinism VERIFY_DETERMINISM_ARGS="--min-cosine-similarity 1.0001"
```

## Docker Quickstart

The Docker workflow targets a local Linux host with an NVIDIA GPU. The checked-in compose file defaults `DEVICE=cuda`, persists the Hugging Face cache in a named volume, and keeps the container unhealthy until `/readyz` succeeds after model initialization and warmup.

### Host prerequisites

- Docker Engine with Docker Compose v2
- NVIDIA GPU with a driver compatible with CUDA 12.6
- NVIDIA Container Toolkit configured for Docker
- outbound network access to Hugging Face on first startup when the cache is empty

Quick host-side sanity check before building:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 nvidia-smi
```

Primary Docker workflow:

```bash
make docker-build
make docker-up
make docker-health
make docker-test-request
```

Inspect logs or stop the stack:

```bash
make docker-logs
make docker-down
```

The Make targets wrap the checked-in compose file at `docker/docker-compose.yml`. If you need the equivalent raw commands:

```bash
docker compose -f docker/docker-compose.yml build embedserve
docker compose -f docker/docker-compose.yml up --detach embedserve
docker compose -f docker/docker-compose.yml down
```

The smoke helpers avoid host Python dependencies:

- `make docker-health` waits on the container health status and confirms `/readyz` from inside the container.
- `make docker-test-request` executes the sample `POST /embed` call from inside the running container.

### Startup notes

- `GET /healthz` is process liveness only and can return `200` before the model is ready.
- `GET /readyz` is the Docker healthcheck target and only returns `200` after model load and the startup warmup embed succeed.
- The first container startup may spend significant time downloading the pinned model revision into `/var/cache/huggingface`.
- Subsequent restarts reuse the named `hf-cache` Docker volume and should not re-download the model unless the volume is removed.

### Docker configuration defaults

Compose reads runtime settings from environment variables with defaults baked into `docker/docker-compose.yml`. The tracked `.env.example` leaves `DEVICE` unset so Docker falls back to its own `cuda` default, while the local `make start` flow still uses the application's `cpu` default unless you override it in `.env`.

- `EMBEDSERVE_PORT=8000`
- `DEVICE=cuda` when unset in `.env`
- `DTYPE=float32`
- `MODEL_ID=sentence-transformers/all-MiniLM-L6-v2`
- `MODEL_REVISION=826711e54e001c83835913827a843d8dd0a1def9`

Known non-portable pieces:

- NVIDIA driver and container toolkit compatibility are host-specific.
- Embedding values are not portable across different GPUs, drivers, or low-level runtime versions.
- First boot requires network access to Hugging Face if the model cache volume is empty.

## Configuration

All configuration is environment-driven.

```env
MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
MODEL_REVISION=826711e54e001c83835913827a843d8dd0a1def9
LOG_LEVEL=INFO
MAX_INPUTS_PER_REQUEST=64
DTYPE=float32
MAX_LENGTH=512
TRUNCATE=true
NORMALIZE_EMBEDDINGS=true
OUTPUT_DTYPE=float32
MAX_BATCH_SIZE=128
MAX_BATCH_TOKENS=8192
BATCH_TIMEOUT_MS=2
MAX_BATCH_QUEUE_SIZE=1024
BATCH_REQUEST_TIMEOUT_MS=5000
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
| `MAX_BATCH_SIZE` | `128` | Maximum number of requests merged into one batch flush. Must be at least `1`. |
| `MAX_BATCH_TOKENS` | `8192` | Maximum summed post-truncation token count merged into one batch flush. Must be at least `1`. |
| `BATCH_TIMEOUT_MS` | `2` | Max wait to assemble a shared batch before flush. Must be at least `1`. |
| `MAX_BATCH_QUEUE_SIZE` | `1024` | Maximum queued requests before overload rejection. Must be at least `1`. |
| `BATCH_REQUEST_TIMEOUT_MS` | `5000` | Per-request wait timeout while awaiting a batch result. Must be at least `1`. |

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
- If the batch queue is saturated, valid requests return `503 {"detail":"Batch queue is full"}`.
- If a queued request exceeds `BATCH_REQUEST_TIMEOUT_MS`, it returns `503 {"detail":"Embedding request timed out"}`.
- During shutdown drain, unresolved requests return `503 {"detail":"Service is shutting down"}`.
- Internal preflight/inference failures return `500 {"detail":"Embedding inference failed"}`.

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

Other `503` operational responses:

```json
{"detail":"Batch queue is full"}
```

```json
{"detail":"Embedding request timed out"}
```

```json
{"detail":"Service is shutting down"}
```

### `GET /metrics`

Exposes Prometheus text format with:

- `embedserve_http_requests_total`
- `embedserve_http_request_duration_seconds`
- `embedserve_app_ready`
- `embedserve_batch_queue_depth`
- `embedserve_batch_queue_wait_seconds`
- `embedserve_batch_size`
- `embedserve_batch_token_count`
- `embedserve_batch_flush_total{reason="max_batch_size|max_batch_tokens|timeout|shutdown"}`
- `embedserve_batch_overload_rejections_total`
- `embedserve_batch_request_timeouts_total`
- `embedserve_batch_request_cancellations_total`
- `embedserve_batch_shutdown_rejections_total`
- `embedserve_batch_inference_failures_total`
- default process and Python runtime metrics

`embedserve_app_ready{mode="model"}` is `1` when the model runtime is ready and `0` otherwise.

Batch histogram buckets are fixed in code for compatibility:

- `embedserve_batch_queue_wait_seconds`: `0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5`
- `embedserve_batch_size`: `1, 2, 4, 8, 16, 32, 64, 128, 256`
- `embedserve_batch_token_count`: `32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384`

## Batching Verification Harness

Run the batching acceptance harness against a live server:

```bash
make verify-batching
```

The harness profile captures:

- hardware identifier (`--hardware-id`)
- `MODEL_ID` (`--model-id`)
- `MODEL_REVISION` (`--model-revision`)
- concurrency and request count
- input shape (`--inputs-per-request`, `--input-token-count`)
- warmup count
- batching settings (`MAX_BATCH_SIZE`, `MAX_BATCH_TOKENS`, `BATCH_TIMEOUT_MS`, `MAX_BATCH_QUEUE_SIZE`, `BATCH_REQUEST_TIMEOUT_MS`)

## Install

For a local developer setup, run:

```bash
make bootstrap-dev
```

That creates `venv/`, installs the pinned runtime and dev dependencies, selects a Linux Torch overlay, and installs the repo's pre-commit hooks. On Linux, `make bootstrap-dev` defaults `TORCH_VARIANT=auto`, which prefers `requirements.cuda-linux.txt` when `nvidia-smi -L` detects an NVIDIA GPU and otherwise falls back to `requirements.cpu-linux.txt`. Override that selection with `make bootstrap-dev TORCH_VARIANT=cpu` or `make bootstrap-dev TORCH_VARIANT=cuda`. The repo now standardizes on Python 3.10 for both local and Docker workflows.

## Dependency Management

The repo now uses `.in` files as the human-edited dependency inputs and committed `.txt` lockfiles as the reproducible install artifacts.

Edit these files directly:

- `requirements.in` for direct runtime dependencies
- `dev-requirements.in` for direct dev tooling
- `requirements.cpu-linux.in` for the local Linux CPU Torch overlay input
- `requirements.cuda-linux.in` for the Linux CUDA Torch overlay input

Generated files:

- `requirements.txt` is the portable runtime base lockfile used by the local and Docker overlays
- `dev-requirements.txt` is the pinned dev-tools lockfile
- `requirements.cpu-linux.txt` is the local Linux CPU Torch overlay lockfile used by `make bootstrap-dev`
- `requirements.cuda-linux.txt` is the Linux CUDA Torch overlay lockfile used by `make bootstrap-dev` and `docker/Dockerfile`

The generation flow is:

1. `pip-compile` resolves a full lockfile from each `.in` file.
2. `scripts/filter_portable_requirements.py` removes CUDA-only packages from the runtime lockfile and rewrites the base Torch pin so `requirements.txt` stays portable.
3. `scripts/filter_torch_overlay_requirements.py` keeps only the Linux CPU Torch pin, then writes `requirements.cpu-linux.txt` with `-r requirements.txt` so local Linux development can install Torch without reintroducing CUDA packages into the portable base.
4. `scripts/filter_cuda_overlay_requirements.py` keeps only the CUDA runtime packages plus `torch` and `triton`, then writes `requirements.cuda-linux.txt` with `-r requirements.txt` so Linux GPU development and the Docker image both install the portable base plus the CUDA overlay.

After changing any `.in` file, regenerate the lockfiles without changing already-pinned versions:

```bash
make deps-compile
```

Use this only when you intentionally want to refresh pinned versions:

```bash
make deps-upgrade
```

Do not edit `requirements.txt`, `requirements.cpu-linux.txt`, `requirements.cuda-linux.txt`, or `dev-requirements.txt` by hand unless you are debugging the generation workflow itself.

## Worktrees

Create a new sibling worktree on a new branch named after `WORKTREE`:

```bash
make worktree-create WORKTREE=agent-a
```

This defaults to:

- base ref `HEAD`
- path `../embedserve-agent-a`
- copying `.env` from the current repo when it exists
- bootstrapping the new worktree with `make bootstrap-dev`

Create from a different base or path:

```bash
make worktree-create WORKTREE=bugfix BASE=main
make worktree-create WORKTREE=perf WORKTREE_PATH=../scratch/embedserve-perf
```

Skip setup or `.env` copy when you want a lighter-weight checkout:

```bash
make worktree-create WORKTREE=docs SETUP=0 COPY_ENV=0
```

Remove a clean worktree but keep its branch:

```bash
make worktree-remove WORKTREE=agent-a
```

Force-remove a dirty worktree and delete its branch too:

```bash
make worktree-remove WORKTREE=agent-a FORCE=1 DELETE_BRANCH=1
```
