 ## Milestone 4 Plan: Dynamic Batching v1

  ### Summary

  Implement an always-on async batching path for /embed that merges concurrent requests into shared model forward passes while preserving current request/response semantics.
  Locked product decisions from this planning pass:

  1. Overload rejection is 503 with FastAPI-style {"detail": "..."}.
  2. No client-provided per-request timeout hint in request body or headers.
  3. Batching is always enabled when runtime is ready.
  4. Queue capacity and request timeout are public env-configurable settings.

  ### Public Contract and Config Changes

  1. Keep /embed request/response schema unchanged for success payloads.
  2. Keep existing 422 validation behavior for payload shape and invalid input text.
  3. Add two new 503 operational failure details on /embed (same error envelope as today):

  - Queue overload: {"detail": "Batch queue is full"}
  - Request timeout: {"detail": "Embedding request timed out"}

  4. Add/validate new env vars:

  - MAX_BATCH_SIZE (default 128, >=1) meaning max total input strings in one model batch.
  - MAX_BATCH_TOKENS (default 8192, >=1) meaning max summed tokens targeted per model batch.
  - BATCH_TIMEOUT_MS (default 2, >=0) max wait to accumulate a batch after first queued item.
  - MAX_BATCH_QUEUE_SIZE (default 1024, >=1) max queued requests before overload rejection.
  - BATCH_REQUEST_TIMEOUT_MS (default 5000, >=1) max per-request wait for queue+batch processing.

  5. Preserve existing readiness contract; if batching engine startup fails, runtime is marked unready with initialization_failed.

  ### Implementation Changes

  1. Batching engine

  - Add an async DynamicBatcher component with:
      - bounded asyncio.Queue of request jobs,
      - one background worker task,
      - fan-out of one merged model response back to per-request futures.
  - Worker assembly policy:
      - build batch from queued jobs until hitting MAX_BATCH_SIZE, MAX_BATCH_TOKENS, or BATCH_TIMEOUT_MS.
      - flush reasons are explicit enum labels: max_batch_size, max_batch_tokens, timeout, shutdown.
      - if a single request exceeds MAX_BATCH_TOKENS, run it as a singleton batch (no rejection) to preserve semantics.
  - Cancellation policy:
      - canceled caller marks job canceled; worker skips canceled queued jobs.
      - if cancellation happens after dispatch, result is dropped for that caller only.
  - Timeout policy:
      - endpoint waits on job future with BATCH_REQUEST_TIMEOUT_MS; timeout returns 503 and marks job timed out.

  2. Embedder preflight API for batching safety

  - Extend embedder interface with a request preflight method that:
      - validates tokenizer-length rules per request (TRUNCATE=false still raises RequestValidationError),
      - returns token count estimate aligned with usage.tokens semantics (post-truncation effective length).
  - Use preflight in batch worker per job before inclusion so one invalid request yields only that request’s 422 and does not poison other requests in the merged batch.

  3. Runtime and endpoint wiring

  - Initialize and start DynamicBatcher in app lifespan only when runtime is ready.
  - Store batcher in app state and add dependency accessor.
  - Replace /embed direct run_in_threadpool(embedder.embed, ...) with await batcher.submit(...).
  - Keep existing validation order: request schema and MAX_INPUTS_PER_REQUEST checks happen before queue submission.

  4. Metrics

  - Add batching metrics to Prometheus registry:
      - embedserve_batch_queue_depth (gauge)
      - embedserve_batch_queue_wait_seconds (histogram)
      - embedserve_batch_size (histogram)
      - embedserve_batch_token_count (histogram)
      - embedserve_batch_flush_total{reason=...} (counter)
      - embedserve_batch_overload_rejections_total (counter)
      - embedserve_batch_request_timeouts_total (counter)
      - embedserve_batch_request_cancellations_total (counter)
  5. Docs and acceptance harness
  - Update README:
      - replace “inference is serialized” with batching behavior,
      - document new env vars and overload/timeout semantics,
      - document batching metrics names/labels.
  - Add a live verification script (for milestone acceptance) that drives concurrent /embed traffic and asserts:
      - at least one non-singleton batch happened (batch_size_sum > batch_size_count),
      - batching metrics are non-zero,
      - all responses are valid or rejected per overload policy.

  ### Test Plan

  1. Unit tests for DynamicBatcher:

  - merges concurrent jobs and preserves per-request output ordering/index reset,
  - flushes by size, token cap, and timeout reasons correctly,
  - handles oversized single request without rejection,
  - handles worker shutdown draining pending jobs safely.

  2. API tests for /embed:

  - queue saturation returns 503 with overload detail,
  - request timeout returns 503 timeout detail,
  - mixed valid + invalid concurrent requests return isolated results (200 and 422 as appropriate),
  - no cross-request output leakage under concurrency.

  3. Cancellation tests:

  - canceled in-flight request does not produce a 500 and does not corrupt queue depth/fan-out.

  4. Metrics tests:

  - /metrics contains new series with expected labels,
  - overload/timeout/cancellation counters increment in corresponding scenarios.

  5. OpenAPI test:

  - /embed responses include documented 503 operational behavior.

  ### Assumptions and Defaults

  1. Overload and timeout both use HTTP 503 with FastAPI default error body; differentiation is via detail string and metrics.
  2. No per-request timeout hint is accepted from clients in v1.
  3. Batching is always on for ready runtime; no feature flag in Milestone 4.
  4. Defaults are balanced for latency vs throughput: small micro-batch wait (2ms) with moderate batch caps (128 inputs / 8192 tokens).
