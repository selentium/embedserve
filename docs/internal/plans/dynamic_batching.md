## Milestone 4 Plan: Dynamic Batching v1

### Summary

Implement an always-on async batching path for `POST /embed` that merges concurrent requests into shared forward passes while preserving per-request semantics and existing success schema.

Locked decisions:

1. Overload rejection is HTTP `503` with FastAPI-style `{"detail": "Batch queue is full"}`.
2. Overload rejection does not include a `Retry-After` header in v1.
3. Request timeout is HTTP `503` with FastAPI-style `{"detail": "Embedding request timed out"}`.
4. Shutdown rejection is HTTP `503` with FastAPI-style `{"detail": "Service is shutting down"}`.
5. Internal batch inference failure is HTTP `500` with FastAPI-style `{"detail": "Embedding inference failed"}` for affected requests.
6. No client-provided per-request timeout hint (body or headers) in v1.
7. Batching is always enabled when runtime is ready.
8. Queue capacity and request timeout are public env-configurable settings.
9. Preflight runs at submit time (before enqueue) and only validated jobs enter the queue.
10. Batch assembly uses strict FIFO request order; no token-fit reordering in v1.
11. `MAX_INPUTS_PER_REQUEST` is enforced before readiness check, preflight, and enqueue.
12. Any non-validation preflight failure uses the same internal failure contract:
    HTTP `500` with FastAPI-style `{"detail": "Embedding inference failed"}`.

### Public Contract and Interface Changes

- Keep `/embed` success payload unchanged.
- Keep existing `422` request validation behavior for schema and blank-input rules.
- Keep existing `MAX_INPUTS_PER_REQUEST` enforcement order: reject over-limit payloads with
  `422` before readiness check, preflight, and enqueue.
- Keep existing unready behavior `503 {"detail": "Model is not ready: <reason>"}`.
- Overload `503` response remains body-only in v1; no `Retry-After` header is returned.
- Add public batch env vars:
  `MAX_BATCH_SIZE` (`>=1`, default `128`),
  `MAX_BATCH_TOKENS` (`>=1`, default `8192`),
  `BATCH_TIMEOUT_MS` (`>=1`, default `2`),
  `MAX_BATCH_QUEUE_SIZE` (`>=1`, default `1024`),
  `BATCH_REQUEST_TIMEOUT_MS` (`>=1`, default `5000`).
- Extend embedder interface with a preflight API for safe batching:
  `preflight(inputs: list[str]) -> list[int]`.
  Returned list contains per-input token counts aligned with `usage.tokens` semantics (post-truncation and special-token insertion).
- `preflight` may raise `RequestValidationError` (for `TRUNCATE=false` length violations), and this must fail only that request.
- `/embed` OpenAPI keeps one `503` response schema and documents named examples for `unready`, `queue_full`, `request_timeout`, and `shutdown`.

### Implementation Changes

- Batcher core: add `DynamicBatcher` with bounded `asyncio.Queue`, one worker task, per-job futures, and response fan-out.
- Submit path preflight: endpoint performs `preflight(inputs)` before enqueue (thread-offloaded), stores per-request token counts on the job, and enqueues only validated jobs.
- Submit ordering: endpoint enforces `MAX_INPUTS_PER_REQUEST` first, then checks runtime readiness, then runs preflight, then enqueues.
- Batch assembly: worker builds a batch in strict FIFO order until `MAX_BATCH_SIZE`, `MAX_BATCH_TOKENS`, or `BATCH_TIMEOUT_MS` is reached; flush reason enum is fixed to `max_batch_size`, `max_batch_tokens`, `timeout`, `shutdown`.
- FIFO rule: worker never skips the queue head to improve token packing; if the current partial batch cannot admit the FIFO head under caps, flush current batch and start the next batch with that head.
- Oversized single request: if one request exceeds `MAX_BATCH_TOKENS`, process it as a singleton batch (no queue-level rejection).
- Concurrency safety: preflight and inference use the same embedder synchronization boundary (single lock or equivalent) so tokenizer/model state is never accessed concurrently in unsafe ways.
- Non-blocking inference: worker executes blocking embedder inference through thread offload (`run_in_threadpool` or `asyncio.to_thread`) so model/tokenizer execution never blocks the event loop.
- Split correctness: merged outputs are sliced back to original callers with correct boundaries; each caller gets indices reset to `0..n-1` and per-request `usage.tokens` from that caller's preflight counts.
- Future lifecycle contract: set result/exception only on pending futures; if a future is already done/cancelled (caller timeout/cancel), skip write and clean bookkeeping without raising `InvalidStateError`.
- Failure isolation:
  preflight `422` for one request does not poison other queued requests;
  preflight non-validation internal failure returns `500 {"detail": "Embedding inference failed"}` for that request only;
  internal inference exception for one flushed batch fails only requests in that batch with the `500` contract above.
- Timeout/cancellation: endpoint awaits job future with `BATCH_REQUEST_TIMEOUT_MS`; timeout returns timeout `503` and increments timeout metric; cancelled callers are marked cancelled and dropped from fan-out safely; worker prunes cancelled/timed-out jobs before token accounting and before inference execution.
- Runtime wiring: create/start batcher during lifespan only when runtime is ready; store batcher in app state; route `/embed` through `await batcher.submit(payload.inputs)`.
- Shutdown behavior: on app shutdown, stop new intake, flush worker with `shutdown` reason, return completed results when available, fail unresolved queued requests with shutdown `503` (not timeout), and stop worker cleanly.
- Metrics (compatibility surface):
  `embedserve_batch_queue_depth` (gauge),
  `embedserve_batch_queue_wait_seconds` (histogram),
  `embedserve_batch_size` (histogram),
  `embedserve_batch_token_count` (histogram),
  `embedserve_batch_flush_total{reason=...}` (counter),
  `embedserve_batch_overload_rejections_total` (counter),
  `embedserve_batch_request_timeouts_total` (counter),
  `embedserve_batch_request_cancellations_total` (counter),
  `embedserve_batch_shutdown_rejections_total` (counter),
  `embedserve_batch_inference_failures_total` (counter).
  Label sets are fixed and low-cardinality; histogram buckets are fixed in code and documented in README.

### Test Plan

1. Unit tests for `DynamicBatcher`: flush-by-size, flush-by-token-cap, flush-by-timeout, singleton-oversize handling, shutdown flush, and queue-depth accounting.
2. Unit tests for FIFO behavior: queue head is never skipped for token-fit packing.
3. API tests for submit-time preflight: invalid overlength request returns `422` without entering queue, while concurrent valid requests proceed normally.
4. API tests for validation ordering: `MAX_INPUTS_PER_REQUEST` over-limit payload returns `422` before readiness and preflight paths.
5. Split/fan-out tests: no cross-request leakage, per-request index reset, and per-request `usage.tokens` correctness.
6. API tests: queue saturation returns overload `503`, request timeout returns timeout `503`, shutdown returns shutdown `503`, mixed valid/invalid concurrent requests yield isolated `200` and `422`.
7. Cancellation tests: cancelled request does not create HTTP `500` noise, does not corrupt queue state, and pruned jobs are not sent to inference.
8. Failure tests: injected preflight internal exception maps to `500 {"detail": "Embedding inference failed"}` for that request only; injected batch inference exception yields the same `500` for affected requests, does not hang futures, and increments inference-failure metric.
9. Metrics tests: `/metrics` exposes all new series and expected flush reason label values.
10. OpenAPI tests: `/embed` documents one `503` response schema with named operational examples.

### Docs and Acceptance Harness

- Update README Milestone-3 wording (serialized inference) to batching behavior.
- Document all new env vars and exact overload/timeout/shutdown semantics.
- Document batching metric names, labels, and histogram compatibility expectations.
- Add `scripts/verify_batching.py` and document a fixed harness profile: hardware identifier, `MODEL_ID`, `MODEL_REVISION`, concurrency, input shape, warmup count, and batching settings.
- Harness acceptance assertions: at least one non-singleton batch, batching metrics non-zero, and every request either succeeds with correct semantics or is rejected by documented overload/timeout/shutdown policy.

### Assumptions and Defaults

- Overload, request-timeout, and shutdown remain `503` and are differentiated by detail text and dedicated metrics.
- Overload `503` does not include `Retry-After` in v1.
- Internal inference failures remain `500` for affected requests.
- Preflight non-validation internal failures use the same `500 {"detail": "Embedding inference failed"}` contract.
- No client timeout hint in v1.
- Batching has no feature flag in Milestone 4.
- Preflight runs at submit time; invalid requests do not enter the batch queue.
- `MAX_INPUTS_PER_REQUEST` is enforced before readiness check and preflight.
- FIFO batching is strict; no reordering for token-fit packing in v1.
- Default latency/throughput balance remains `BATCH_TIMEOUT_MS=2`, `MAX_BATCH_SIZE=128`, `MAX_BATCH_TOKENS=8192`, `MAX_BATCH_QUEUE_SIZE=1024`, `BATCH_REQUEST_TIMEOUT_MS=5000`.
