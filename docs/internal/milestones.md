# Milestone 1 — Repo bootstrap + runnable API skeleton

*Goal:* service starts and responds, even before GPU/model work is perfect.

### Deliverables

 - Repo layout in place (app/, docker/, scripts/, tests/)

 - FastAPI app with:

    - GET /healthz, GET /readyz, GET /metrics

    - POST /embed (stub or minimal embedding path)

 - Logging + settings via env vars

 - Basic pytest smoke tests (/healthz, /embed)


### Acceptance checks

 - uvicorn app.main:app runs locally

 - /metrics returns Prometheus text format





# Milestone 2 — Model loading + revision pinning + single-request embeddings

*Goal:* real embeddings from a pinned HF revision.


### Deliverables

 - HF model + tokenizer loading with:

    - MODEL_ID, MODEL_REVISION (exact hash supported)

    - DTYPE (bf16/fp16) + DEVICE (cuda:0)

 - Embedding pipeline:

    - tokenization

    - pooling (mean pooling baseline)

    - optional normalization (L2)

    - output dtype control (float32)

 - /readyz checks model loaded + device available



### Acceptance checks

 - POST /embed returns vectors with correct dimension

 - Response includes model, revision, dim, usage.tokens






# Milestone 3 — Determinism controls + verification script

*Goal:* “deterministic” claim is backed by a script + documented policy.

### Deliverables

 - Determinism flags + seeding (best-effort)

 - scripts/verify_determinism.py:

    - repeats same inputs N times

    - reports max_abs_diff + min_cosine

    - prints pass/fail thresholds (configurable)

 - README section: “Numerical stability vs bitwise determinism” + what you guarantee

### Acceptance checks

Script runs against server and produces stable results within tolerance (on your setup)



# Milestone 4 — Dynamic batching v1 (queue + worker + latency cap)

*Goal:* concurrent requests get merged into GPU batches.

### Deliverables

 - Async batching engine:

    - request queue

    - batch assembly loop

    - constraints: MAX_BATCH_SIZE, MAX_BATCH_TOKENS, BATCH_TIMEOUT_MS

    - splits outputs back to callers

 - Metrics hooks for batching:

    - queue depth, queue wait

    - batch size/tokens

    - flush reasons

### Acceptance checks

 - Under concurrency (e.g. 32 clients), throughput improves vs no batching

 - /metrics shows non-zero batching counters/histograms


# Milestone 5 — Docker GPU deployment (reproducible run)

*Goal:* “one command” deployment.

### Deliverables

 - docker/Dockerfile (CUDA base image)

 - docker/docker-compose.yml GPU-enabled

 - README quickstart:

    - docker build + docker run --gpus all

    - compose run instructions

 - Healthcheck notes

### Acceptance checks

Fresh machine (or clean environment) can run the service in Docker and hit /embed



# Milestone 6 — Benchmark suite (10k test + latency/throughput report)

*Goal:* measurable performance numbers with a repeatable script.

### Deliverables

 - scripts/bench_10k.py:

    - sends N texts

    - reports throughput + p50/p95 latency (client-side)

 - BENCHMARK.md (or README section):

    - hardware specs used

    - settings used (batch sizes, dtype, max tokens)

    - results table

### Acceptance checks

 - bench_10k.py completes and prints a consistent report

 - You can reproduce results with same config



# Milestone 7 — 1-hour stability test + memory monitoring

*Goal:* prove no VRAM creep / stability under sustained load.

### Deliverables

 - scripts/load_test.py:

    - sustained concurrency for duration

    - periodic metrics sampling (from /metrics and/or torch.cuda)

 - Add GPU memory metrics:

    - allocated/reserved bytes gauges

    - OOM counter

 - A “Stability Results” section with your run summary

### Acceptance checks

 - 1-hour run completes without crashes

 - VRAM doesn’t monotonically grow beyond small tolerance

 - No unhandled exceptions under load