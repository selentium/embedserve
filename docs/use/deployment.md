# Deployment

The checked-in deployment path targets a local Linux host with an NVIDIA GPU.

## Host Prerequisites

- Docker Engine with Docker Compose v2
- NVIDIA GPU with a driver compatible with CUDA 12.6
- NVIDIA Container Toolkit configured for Docker
- Outbound network access to Hugging Face on first startup when the cache is empty

Sanity check:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 nvidia-smi
```

## Supported Docker Flow

Build and start:

```bash
make docker-build
make docker-up
make docker-health
make docker-test-request
```

Inspect or stop:

```bash
make docker-logs
make docker-ps
make docker-down
```

Raw Compose equivalents:

```bash
docker compose -f docker/docker-compose.yml build embedserve
docker compose -f docker/docker-compose.yml up --detach embedserve
docker compose -f docker/docker-compose.yml down
```

## Compose Defaults

The checked-in Compose file publishes port `8000`, requests all GPUs, and persists the Hugging Face cache in the `hf-cache` named volume.

Important defaults when variables are unset:

- `DEVICE=cuda`
- `DTYPE=float32`
- `MODEL_ID=sentence-transformers/all-MiniLM-L6-v2`
- `MODEL_REVISION=826711e54e001c83835913827a843d8dd0a1def9`

The container sets `HF_HOME=/var/cache/huggingface` and reuses it through the named volume.

## Startup Notes

- `/healthz` can return `200` before the model is ready.
- `/readyz` is the Docker healthcheck target.
- The first successful startup may spend significant time downloading the pinned model revision.
- Later restarts reuse the `hf-cache` volume and should not re-download the model unless the volume is removed.

## Port and Environment Overrides

Typical override patterns:

```bash
EMBEDSERVE_PORT=9000 make docker-up
DEVICE=cuda:0 DTYPE=float16 make docker-up
MODEL_ID=sentence-transformers/all-MiniLM-L6-v2 MODEL_REVISION=826711e54e001c83835913827a843d8dd0a1def9 make docker-up
```

## Non-Portable Pieces

- NVIDIA driver compatibility is host-specific.
- Embedding values are not portable across different GPUs, drivers, or low-level runtime versions.
- The image is built from a pinned CUDA runtime base and pinned Python lockfiles, but host GPU runtime compatibility still matters.

For runtime interpretation after deployment, see [Operations and Monitoring](operations.md).
