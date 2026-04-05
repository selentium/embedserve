# Performance and Stability

EmbedServe keeps three distinct live-server harnesses:

- batching verification
- benchmark runs
- sustained stability runs

## Batching Verification

Run the acceptance harness:

```bash
make verify-batching
```

The harness validates that concurrent traffic produces real batched execution and captures:

- hardware identifier
- model id and revision
- concurrency and request count
- input shape
- warmup count
- batching settings

## Benchmark Harness

Run the repeatable benchmark harness:

```bash
make bench-10k BENCH_10K_ARGS="--json"
```

The benchmark profile captures:

- hardware identifier
- label such as `batched` or `no_batching`
- model and revision
- device and dtype
- tokenization settings
- batching settings
- total texts, request shape, concurrency, timeout, and warmup policy
- whether the run used warm cache, warm model, and warm container

Canonical benchmark reporting format lives in [Benchmark Report](../../BENCHMARK.md).

## Stability Load Harness

Run the sustained-load harness:

```bash
make load-test LOAD_TEST_HARDWARE_ID="gpu-box-01"
```

The default one-hour profile uses:

- `LOAD_TEST_DURATION_SECONDS=3600`
- `LOAD_TEST_CONCURRENCY=32`
- `LOAD_TEST_WARMUP_REQUESTS=64`
- `LOAD_TEST_INPUTS_PER_REQUEST=2`
- `LOAD_TEST_INPUT_TOKEN_COUNT=24`
- `LOAD_TEST_METRICS_POLL_INTERVAL_SECONDS=5`
- `LOAD_TEST_MAX_VRAM_DRIFT_BYTES=268435456`

The harness:

- waits for `/readyz`
- performs warmup requests before measurement
- drives fixed concurrency for the configured duration
- polls `/metrics` for failure counters, process RSS, and CUDA memory
- exits `0` on pass, `1` on stability failure, and `2` on operational failure

Canonical stability reporting format lives in [Stability Report](../../STABILITY.md).

## Choosing the Right Tool

- Use batching verification to confirm the batching path is active and compatible with a target config.
- Use the benchmark harness for throughput and latency comparisons across commits or settings.
- Use the load harness for long-duration health and memory drift checks.
