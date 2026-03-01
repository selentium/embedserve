# EmbedServe

__Deterministic, GPU-accelerated embedding inference server with dynamic batching__

A production-style self-hosted embedding service for transformer models (e.g. Qwen3-Embedding-8B), designed for:

- high-throughput GPU inference
- numerically stable outputs
- dynamic batching under concurrent load
- reproducible deployments via pinned model revisions
- Docker-ready infrastructure


## Features

GPU transformer inference (fp16 / bf16)
- Model revision pinning (exact HuggingFace commit hash)
- Numerically stable embeddings (repeatable within float tolerance)
- Dynamic batching with latency cap
- FastAPI HTTP service
- Prometheus metrics endpoint
- Dockerized deployment
- Determinism verification script
- 10k-text benchmark script
- 1-hour load stability test (VRAM monitoring)

## Configuration

All configuration via environment variables.

### Model

```
MODEL_ID=Qwen/Qwen3-Embedding-8B
MODEL_REVISION=<exact_commit_hash>
DTYPE=bf16
DEVICE=cuda:0
```

### Batching

```
MAX_BATCH_SIZE=64
MAX_BATCH_TOKENS=16384
BATCH_TIMEOUT_MS=10
```

### Tokenization

```
MAX_LENGTH=512
TRUNCATE=true
```

### Output

```
NORMALIZE_EMBEDDINGS=true
OUTPUT_DTYPE=float32
```

## Dynamic batching design


The batching engine:

 - collects incoming requests

 - groups them until:

    - batch size reached

    - token budget reached

    - timeout exceeded

 - executes single GPU forward pass

 - splits outputs back to original callers

This dramatically improves throughput under load.


## API  

 - POST /embed
 - GET /healthz
 - GET /readyz
 - GET /metrics

## Repo layout

```
embedserve/
  app/
    __init__.py
    main.py
    settings.py
    schemas.py
    logging.py
    metrics.py
    deps.py
    engine/
      __init__.py
      model.py
      embedder.py
      batcher.py
      utils.py
  scripts/
    verify_determinism.py
    bench_10k.py
    load_test.py
  tests/
    test_api.py
    test_batcher.py
  requirements.txt
  README.md
  .gitignore
```