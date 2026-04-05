# Quickstart

The fastest supported path is the checked-in Docker workflow.

## Prerequisites

- Docker Engine with Docker Compose v2
- NVIDIA GPU with a driver compatible with CUDA 12.6
- NVIDIA Container Toolkit configured for Docker
- Outbound network access to Hugging Face on first startup when the cache is empty

Verify the host can expose the GPU to containers:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 nvidia-smi
```

## Start the Service

```bash
make docker-build
make docker-up
make docker-health
```

Useful follow-up commands:

```bash
make docker-test-request
make docker-logs
```

## Make Your First Request

```bash
curl http://127.0.0.1:8000/embed \
  -H 'content-type: application/json' \
  -d '{"inputs":["hello world","quickstart example"]}'
```

Example response shape:

```json
{
  "data": [
    {
      "index": 0,
      "embedding": [0.1, 0.2, 0.3]
    },
    {
      "index": 1,
      "embedding": [0.1, 0.2, 0.3]
    }
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "revision": "826711e54e001c83835913827a843d8dd0a1def9",
  "dim": 384,
  "usage": {
    "tokens": 16
  }
}
```

The numeric values and dimension depend on the configured model and output precision.

## Health and Readiness

Liveness:

```bash
curl http://127.0.0.1:8000/healthz
```

Readiness:

```bash
curl http://127.0.0.1:8000/readyz
```

Interpretation:

- `/healthz` returning `200` means the process is up.
- `/readyz` returning `200` means model load and warmup succeeded.
- `/readyz` returning `503` means the service is running but not ready to serve embeddings.

## Stop the Service

```bash
make docker-down
```

For configuration overrides and raw Compose commands, see [Deployment](deployment.md).
