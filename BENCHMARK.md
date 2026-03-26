# Benchmark Report

This document is the stable benchmark report for EmbedServe.
Use it to compare runs across commits with a fixed methodology and fixed table format.

## Methodology

- Harness: `python scripts/bench_10k.py --json`
- Endpoint under test: `POST /embed`
- Corpus: deterministic tokenizer-stable synthetic texts generated from the pinned model tokenizer
- Canonical measured profile: `10000` total texts, `1` input per request, `24` tokenizer tokens per input, `32` concurrent requests, `200` sequential warmup requests
- Warmup policy: run `warmup_requests` sequentially before the measured phase; warmup failures abort the benchmark
- Latency method: client-side request latency from just before `POST /embed` until response JSON validation completes
- Percentiles: nearest-rank `p50`, `p95`, and `p99` over successful `200` responses only
- Throughput method: report both `texts_per_second` and `requests_per_second` from successful measured requests only
- Error accounting: `timeout_count` is client-side timeouts only; `error_count` includes non-`200` HTTP responses, transport failures, and malformed `200` payloads
- No-batching comparison mode: rerun the same benchmark with the same host, model, revision, dtype, tokenization settings, concurrency, timeout, and warmup policy, changing only `MAX_BATCH_SIZE=1`
- Repeatability band: back-to-back same-config runs on an otherwise idle host are acceptable when `texts_per_second` and `p95` differ by no more than `10%`

## Canonical Commands

Benchmark the normal batched profile:

```bash
make bench-10k BENCH_10K_ARGS="--json"
```

Benchmark the no-batching comparison profile against a server started with `MAX_BATCH_SIZE=1`:

```bash
make bench-10k \
  BENCH_10K_LABEL=no_batching \
  BENCH_10K_MAX_BATCH_SIZE=1 \
  BENCH_10K_ARGS="--json"
```

`bench_10k.py` validates the live server's `/readyz` model, tokenization, and batching settings before it records a run.

## Environment Metadata

| Field | Value |
| --- | --- |
| Hardware ID | `rtx3050-4gb-laptop` |
| Hardware Description | 12th Gen Intel(R) Core(TM) i7-12650H host with NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM), driver 560.35.03 |
| Model ID | `sentence-transformers/all-MiniLM-L6-v2` |
| Model Revision | `826711e54e001c83835913827a843d8dd0a1def9` |
| Device | `cuda:0` |
| Dtype | `float16` |
| Max Length | `512` |
| Truncate | `true` |
| Warm Cache | `true` |
| Warm Model | `true` |
| Warm Container | `true` |

## Results Table

Record one row per committed run and keep the columns stable.

| Date | Commit | Profile | Hardware ID | Model | Revision | Device | Dtype | Concurrency | Total Texts | Inputs/Request | Token Count | Max Length | Truncate | Max Batch Size | Max Batch Tokens | Batch Timeout ms | Max Queue Size | Request Timeout ms | Warm Cache | Warm Model | Warm Container | Duration s | Texts/s | Requests/s | p50 ms | p95 ms | p99 ms | Error Count | Timeout Count | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-26 | 4e8bcf5a809e | batched | rtx3050-4gb-laptop | sentence-transformers/all-MiniLM-L6-v2 | 826711e54e001c83835913827a843d8dd0a1def9 | cuda:0 | float16 | 32 | 10000 | 1 | 24 | 512 | true | 128 | 8192 | 2 | 1024 | 5000 | true | true | true | 33.235 | 300.888 | 300.888 | 78.154 | 263.516 | 399.685 | 0 | 0 | `OUTPUT_DTYPE=float32`; throughput `+5.48%` vs `no_batching`, p95 `+67.62%` |
| 2026-03-26 | 4e8bcf5a809e | no_batching | rtx3050-4gb-laptop | sentence-transformers/all-MiniLM-L6-v2 | 826711e54e001c83835913827a843d8dd0a1def9 | cuda:0 | float16 | 32 | 10000 | 1 | 24 | 512 | true | 1 | 8192 | 2 | 1024 | 5000 | true | true | true | 35.057 | 285.251 | 285.251 | 108.580 | 157.207 | 306.267 | 0 | 0 | Same config as `batched`, only `MAX_BATCH_SIZE=1`; `OUTPUT_DTYPE=float32` |

## Comparison Notes

- Compute throughput delta versus `no_batching` as `((batched_texts_per_second / no_batching_texts_per_second) - 1) * 100`.
- Compute latency delta versus `no_batching` for `p95` as `((batched_p95_ms / no_batching_p95_ms) - 1) * 100`.
- If a comparison is published, both source rows must use the same commit, host, warmed-state flags, and corpus shape.
- Measured on `2026-03-26` at commit `4e8bcf5a809e`: batched throughput was `+5.48%` versus `no_batching`, while batched `p95` latency was `+67.62%` higher.

## Run Notes

- Keep the benchmark host otherwise idle during reported runs.
- Keep `MODEL_ID`, `MODEL_REVISION`, `DEVICE`, `DTYPE`, `MAX_LENGTH`, `TRUNCATE`, and all batching settings fixed when comparing commits.
- Save the raw JSON output from `bench_10k.py` alongside local experiment notes before copying summarized values into this report.
- `2026-03-26` calibration run: `1000` texts, `50` warmup requests, `32` concurrency, zero errors/timeouts before the recorded `10k` runs.
- Raw JSON from the recorded runs was saved under `/tmp/embedserve-benchmarks-20260326/`.
