# Milestones

This roadmap is derived from the product contract in `README.md`.
Each milestone should leave the repository in a usable state with clear pass/fail checks.

## Milestone 1 - Repo bootstrap and API contract

*Goal:* the service starts locally, exposes the public endpoints, and has a stable API contract before GPU work begins.

### Deliverables

- Repository layout in place for `app/`, `scripts/`, `tests/`, and `docker/`.
- FastAPI app with `GET /healthz`, `GET /readyz`, `GET /metrics`, and `POST /embed`.
- Initial request and response schemas for `/embed`, including validation errors.
- Environment-driven settings and structured logging.
- Basic Prometheus instrumentation for request count, request latency, and process health.
- Smoke tests for `/healthz`, `/readyz`, `/metrics`, and `/embed`.

### Required API decisions

- Whether `/embed` accepts a single string, a list of strings, or both.
- Stable response shape for success and validation failure.
- Whether response metadata includes `model`, `revision`, `dim`, and `usage.tokens`.
- Initial readiness policy before model load exists.

### Acceptance checks

- `uvicorn app.main:app` starts locally with documented environment variables only.
- `GET /healthz` returns `200`.
- `GET /readyz` returns a documented status that matches current initialization state.
- `GET /metrics` returns valid Prometheus text exposition.
- `POST /embed` returns either a stub response matching the final schema or a documented `501` placeholder.
- Pytest smoke suite passes locally.

## Milestone 2 - Model loading, tokenization, and single-request inference

*Goal:* real embeddings are returned from a pinned Hugging Face model revision on a specific device.

### Deliverables

- Hugging Face model and tokenizer loading with `MODEL_ID` and `MODEL_REVISION`.
- Device and dtype configuration through `DEVICE` and `DTYPE`.
- Tokenization settings through `MAX_LENGTH` and `TRUNCATE`.
- Embedding pipeline covering tokenization, forward pass, pooling, optional L2 normalization, and output dtype conversion.
- `/readyz` reflects model load state and device availability.
- Unit coverage for tokenization limits, truncation behavior, and response metadata.

### Required API decisions

- Default pooling algorithm and whether it is configurable.
- Maximum supported input count per request.
- Behavior when inputs exceed `MAX_LENGTH`.
- Behavior when model load fails at startup.

### Acceptance checks

- `POST /embed` returns vectors with the expected embedding dimension for the configured model.
- Response includes `model`, `revision`, `dim`, and `usage.tokens`.
- `MODEL_REVISION` accepts an exact commit hash and is surfaced in the response.
- `MAX_LENGTH` and `TRUNCATE` behavior is covered by automated tests.
- `/readyz` returns non-ready status when model initialization fails or target device is unavailable.

## Milestone 3 - Determinism policy and verification

*Goal:* the determinism claim is narrowed to something measurable and enforced by tooling.

### Deliverables

- Best-effort determinism settings documented in code and README.
- Explicit policy that distinguishes numerical stability from bitwise determinism.
- `scripts/verify_determinism.py` sends the same inputs repeatedly, reports `max_abs_diff` and `min_cosine_similarity`, accepts configurable thresholds and iteration count, and exits non-zero on threshold failure.
- README section that states exactly what is guaranteed and under which environment assumptions.

### Acceptance checks

- Verification script runs against a live server.
- Default thresholds are documented and checked automatically.
- Script exits `0` on a known-good setup and non-zero when thresholds are intentionally violated.
- README documents the exact limits of the determinism claim.

## Milestone 4 - Dynamic batching v1

*Goal:* concurrent requests are merged into larger GPU batches without breaking per-request semantics.

### Deliverables

- Async batching engine with request queue, background worker, batch assembly, and response fan-out to original callers.
- Batching constraints driven by `MAX_BATCH_SIZE`, `MAX_BATCH_TOKENS`, and `BATCH_TIMEOUT_MS`.
- Request-level timeout and cancellation handling.
- Defined overload policy with queue size limit and explicit rejection behavior.
- Metrics for queue depth, queue wait time, batch size, batch token count, flush reason, and overload rejections.
- Tests for request ordering, timeouts, cancellation, and split-output correctness.

### Required API decisions

- HTTP status and response body for overload rejection.
- Whether callers can provide per-request timeout hints.
- Whether batching is optional or always on once enabled.

### Acceptance checks

- Under concurrent load, batching produces higher throughput than the no-batching path using the same model and hardware.
- `/metrics` shows non-zero batching counters and histograms during the test.
- Requests are either served correctly or rejected according to the documented overload policy.
- No caller receives another request's output.
- Automated tests cover timeout, cancellation, and queue saturation cases.

## Milestone 5 - Dockerized reproducible deployment

*Goal:* the service can be run on a clean machine with a reproducible software stack.

### Deliverables

- `docker/Dockerfile` based on a pinned CUDA runtime image.
- `docker/docker-compose.yml` or equivalent local GPU run configuration.
- Pinned Python dependency set suitable for reproducible rebuilds.
- README quickstart for local Docker build and GPU run.
- Healthcheck configuration and startup notes for model warmup.
- Documented expectations for NVIDIA runtime, driver, and container toolkit.

### Acceptance checks

- A clean environment can build the image and start the service with documented steps only.
- Containerized service answers `POST /embed` successfully on GPU.
- Docker artifacts pin enough of the stack to reproduce the runtime environment across rebuilds.
- README documents required host prerequisites and known non-portable pieces.

## Milestone 6 - Benchmark suite and performance report

*Goal:* performance can be measured and compared using a repeatable script and a fixed reporting format.

### Deliverables

- `scripts/bench_10k.py` sends a configurable number of texts, reports throughput, reports p50, p95, and p99 client-side latency, and records error count and timeout count.
- Benchmark report in `BENCHMARK.md` or README includes hardware used, model and revision, dtype, batching settings, tokenization settings, concurrency level, and a results table.

### Acceptance checks

- Benchmark script completes successfully against a live server and prints a machine-readable summary.
- The report format is stable enough to compare runs across commits.
- Running the benchmark twice with the same config produces results within a documented variance band.
- Report clearly states whether numbers were collected with warm cache, warm model, and warm container.

## Milestone 7 - One-hour stability and memory monitoring

*Goal:* sustained load does not cause crashes, runaway memory growth, or silent failure.

### Deliverables

- `scripts/load_test.py` for sustained configurable concurrency and duration.
- GPU memory metrics for allocated bytes, reserved bytes, and OOM count.
- Request failure metrics for timeout, overload rejection, and internal error.
- Stability report section includes test duration, concurrency level, request volume, error summary, VRAM trend summary, and known caveats.

### Acceptance checks

- A 1-hour run completes without process crash.
- No unhandled exceptions are emitted under sustained load.
- OOM count remains `0` for the documented test configuration.
- VRAM usage does not show unbounded monotonic growth; acceptable drift threshold is documented in the report.
- Final report includes enough detail to reproduce the run conditions.

## Cross-cutting rules

These apply to every milestone after the relevant capability exists.

- New public behavior must be covered by automated tests.
- README must stay aligned with the actual product contract.
- Metrics names and labels should be treated as a compatibility surface once documented.
- Every operational claim in README should map to a measurable check in this roadmap.
