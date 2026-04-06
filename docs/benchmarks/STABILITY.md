# Stability Report

## One-Hour Stability Run

### Environment

- Date: 2026-04-02
- Hardware: `local-dev` (`NVIDIA GeForce RTX 3050 Laptop GPU`)
- GPU driver / CUDA runtime: NVIDIA driver `560.35.03`, CUDA runtime `12.6`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Revision: `826711e54e001c83835913827a843d8dd0a1def9`
- Device: `cuda`
- DType: `float32`
- Batching settings: `max_batch_size=128`, `max_batch_tokens=8192`, `batch_timeout_ms=2`, `max_batch_queue_size=1024`, `batch_request_timeout_ms=5000`
- Tokenization settings: `inputs_per_request=2`, `input_token_count=24`, `max_length=512`, `truncate=true`

### Command

```bash
make load-test \
  LOAD_TEST_EMBED_URL="http://127.0.0.1:8000/embed" \
  LOAD_TEST_METRICS_URL="http://127.0.0.1:8000/metrics" \
  LOAD_TEST_READY_URL="http://127.0.0.1:8000/readyz" \
  LOAD_TEST_DURATION_SECONDS="3600" \
  LOAD_TEST_CONCURRENCY="32" \
  LOAD_TEST_WARMUP_REQUESTS="64" \
  LOAD_TEST_INPUTS_PER_REQUEST="2" \
  LOAD_TEST_INPUT_TOKEN_COUNT="24" \
  LOAD_TEST_TIMEOUT_SECONDS="10" \
  LOAD_TEST_METRICS_POLL_INTERVAL_SECONDS="5" \
  LOAD_TEST_MAX_VRAM_DRIFT_BYTES="268435456" \
  LOAD_TEST_HARDWARE_ID="local-dev" \
  LOAD_TEST_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2" \
  LOAD_TEST_MODEL_REVISION="826711e54e001c83835913827a843d8dd0a1def9" \
  LOAD_TEST_MAX_BATCH_SIZE="128" \
  LOAD_TEST_MAX_BATCH_TOKENS="8192" \
  LOAD_TEST_BATCH_TIMEOUT_MS="2" \
  LOAD_TEST_MAX_BATCH_QUEUE_SIZE="1024" \
  LOAD_TEST_BATCH_REQUEST_TIMEOUT_MS="5000"
```

### Summary

- Pass: `True`
- Duration: `3600.0` seconds
- Concurrency: `32`
- Request volume: `432650`
- Status counts: `{'200': 432650, '500': 0, '503': 0, 'other': 0}`
- Failure counts by category: `{'success': 432650, 'overload': 0, 'timeout': 0, 'shutdown': 0, 'internal_error': 0, 'transport_error': 0, 'unexpected_status': 0}`
- Invalid responses: `0`
- Metric deltas: `{'overload': 0.0, 'timeout': 0.0, 'internal_error': 0.0, 'shutdown': 0.0, 'unhandled_exceptions': 0.0, 'gpu_oom': 0}`

### Memory Trend

- `embedserve_gpu_memory_allocated_bytes`: `109997568 / 109997568 / 120308224 / 109997568 / 0`
- `embedserve_gpu_memory_reserved_bytes`: `134217728 / 134217728 / 1023410176 / 1023410176 / 889192448`
- `process_resident_memory_bytes`: `1191837696 / 1259864064 / 1259864064` (`end_minus_start=68026368`)
- VRAM drift threshold for the stability gate (`embedserve_gpu_memory_allocated_bytes`): `256 MiB` (`268435456` bytes)

### Caveats

- The stability pass/fail gate uses allocated VRAM drift. Reserved VRAM remains in the report as diagnostic telemetry.
- CUDA reserved memory can remain above the allocation floor because PyTorch caches memory aggressively.
- CPU-mode runs do not emit GPU memory gauges and should be reported as GPU telemetry not applicable.
- The milestone acceptance check is based on post-warmup end-minus-start drift, not on a fitted trend line.
- This run passed the allocation-based stability gate: allocated VRAM ended exactly at the baseline (`end_minus_start=0`) with no failed requests, no invalid responses, and no GPU OOMs.
- Reserved VRAM ended `889192448` bytes above baseline while allocated VRAM returned to baseline, which is consistent with the expected PyTorch CUDA caching behavior and is not treated as a failure.
