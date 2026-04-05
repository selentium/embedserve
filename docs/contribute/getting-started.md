# Local Setup

## Prerequisites

- Python 3.10
- `python3 -m venv`
- optional NVIDIA GPU for CUDA-backed local development

## Bootstrap

Create the local virtual environment, install pinned dependencies, select the Linux Torch overlay, and install pre-commit hooks:

```bash
make bootstrap-dev
```

On Linux, `make bootstrap-dev` defaults `TORCH_VARIANT=auto`:

- if `nvidia-smi -L` detects an NVIDIA GPU, it prefers `requirements.cuda-linux.txt`
- otherwise it falls back to `requirements.cpu-linux.txt`

Override that behavior when needed:

```bash
make bootstrap-dev TORCH_VARIANT=cpu
make bootstrap-dev TORCH_VARIANT=cuda
```

## Local Configuration

Use `.env.example` as the starting point for local configuration. Typical local notes:

- local `make start` loads `.env` if present
- the application default for `DEVICE` is `cpu`
- Docker Compose uses its own `cuda` fallback when `DEVICE` is unset

## Run the App

```bash
make start
```

This launches:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Basic Dev Checks

```bash
make lint
make typecheck
make test
```

For live-server verification commands, see [Testing and Verification](testing.md).
