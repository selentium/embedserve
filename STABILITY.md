# Stability Report

## One-Hour Stability Run

Use this section format for the canonical Milestone 7 report.

### Environment

- Date:
- Hardware:
- GPU driver / CUDA runtime:
- Model:
- Revision:
- Device:
- DType:
- Batching settings:
- Tokenization settings:

### Command

```bash
make load-test \
  LOAD_TEST_HARDWARE_ID="gpu-box-01" \
  LOAD_TEST_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2" \
  LOAD_TEST_MODEL_REVISION="826711e54e001c83835913827a843d8dd0a1def9"
```

### Summary

- Duration:
- Concurrency:
- Request volume:
- Status counts:
- Failure counts by category:
- Metric deltas:

### Memory Trend

- `embedserve_gpu_memory_allocated_bytes`: start / min / max / end / end-minus-start
- `embedserve_gpu_memory_reserved_bytes`: start / min / max / end / end-minus-start
- `process_resident_memory_bytes`: start / max / end
- VRAM drift threshold for the stability gate (`embedserve_gpu_memory_allocated_bytes`): `256 MiB` (`268435456` bytes)

### Caveats

- The stability pass/fail gate uses allocated VRAM drift. Reserved VRAM remains in the report as diagnostic telemetry.
- CUDA reserved memory can remain above the allocation floor because PyTorch caches memory aggressively.
- CPU-mode runs do not emit GPU memory gauges and should be reported as GPU telemetry not applicable.
- The milestone acceptance check is based on post-warmup end-minus-start drift, not on a fitted trend line.
