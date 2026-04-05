# Operations and Monitoring

## Liveness vs Readiness

- `GET /healthz` answers whether the process is up.
- `GET /readyz` answers whether model initialization and warmup succeeded.
- `GET /metrics` is always useful for diagnosis, including when `/readyz` is failing.

Treat `/readyz`, not `/healthz`, as the signal for whether the service should receive embedding traffic.

## Metrics Surface

Documented compatibility-sensitive metric names and labels:

- `embedserve_http_requests_total{method,route,status_code}`
- `embedserve_http_request_duration_seconds{method,route}`
- `embedserve_app_ready{mode}`
- `embedserve_batch_queue_depth`
- `embedserve_batch_queue_wait_seconds`
- `embedserve_batch_size`
- `embedserve_batch_token_count`
- `embedserve_batch_flush_total{reason}`
- `embedserve_batch_overload_rejections_total`
- `embedserve_batch_request_timeouts_total`
- `embedserve_batch_request_cancellations_total`
- `embedserve_batch_shutdown_rejections_total`
- `embedserve_batch_inference_failures_total`
- `embedserve_gpu_memory_allocated_bytes{device}`
- `embedserve_gpu_memory_reserved_bytes{device}`
- `embedserve_gpu_oom_total{device}`
- `embedserve_request_failures_total{reason}`
- `embedserve_unhandled_exceptions_total`

Fixed label values already used by the service:

- `embedserve_batch_flush_total{reason="max_batch_size|max_batch_tokens|timeout|shutdown"}`
- `embedserve_request_failures_total{reason="overload|timeout|internal_error|shutdown"}`
- `embedserve_app_ready{mode="model"}`

## What To Watch

Healthy baseline:

- `/readyz` returns `200`
- `embedserve_app_ready{mode="model"}` is `1`
- `embedserve_request_failures_total` is not climbing unexpectedly
- `embedserve_unhandled_exceptions_total` stays flat

Batching stress indicators:

- `embedserve_batch_queue_depth` staying high
- `embedserve_batch_overload_rejections_total` increasing
- `embedserve_batch_request_timeouts_total` increasing
- `embedserve_batch_size` and `embedserve_batch_token_count` changing relative to expected load shape

GPU-specific indicators:

- `embedserve_gpu_oom_total` increasing indicates CUDA OOM incidents
- `embedserve_gpu_memory_allocated_bytes` is the primary stability gate used by the load harness
- `embedserve_gpu_memory_reserved_bytes` is diagnostic and can remain elevated because PyTorch caches memory

## Logs and Request Tracing

- Every response includes `X-Request-ID` for request correlation.
- Access logs record method, path, status code, and duration.
- Unhandled exceptions increment the matching failure metrics and are logged with the generated request id.

## Operational Failure Meanings

On `POST /embed`, the main operational `detail` values are:

- `Model is not ready: <reason>`: runtime not ready
- `Batch queue is full`: overload rejection
- `Embedding request timed out`: request waited too long for batch completion
- `Service is shutting down`: request rejected during shutdown drain
- `Embedding inference failed`: internal preflight or inference failure

## Related Docs

- [API Reference](api.md) defines the exact response payloads.
- [Troubleshooting](troubleshooting.md) covers symptom-driven diagnosis.
- [Performance and Stability](performance.md) explains the verification and load harnesses.
