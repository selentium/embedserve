# Troubleshooting

## `/healthz` is `200` but `/readyz` is `503`

Meaning:

- The process is up, but model initialization or warmup failed.

What to check:

- the `reason` and `detail` fields from `/readyz`
- container logs from `make docker-logs`
- `DEVICE` and `DTYPE` compatibility
- model id and pinned revision correctness

## `reason=device_unavailable`

Likely causes:

- Docker cannot access the GPU
- NVIDIA Container Toolkit is not configured
- requested `cuda:N` device does not exist

What to check:

- `nvidia-smi` on the host
- `docker run --rm --gpus all ... nvidia-smi`
- `DEVICE` override values

## `reason=dtype_unsupported`

Likely cause:

- the selected dtype is not supported on the chosen device

Common example:

- `float16` on CPU is not supported

## `reason=model_load_failed` or `reason=tokenizer_incompatible`

Likely causes:

- invalid or unavailable model revision
- first-start network access failure to Hugging Face
- incompatible tokenizer/model artifacts

What to check:

- outbound connectivity on first startup
- exact `MODEL_ID` and `MODEL_REVISION`
- logs around runtime initialization

## `POST /embed` returns `503 {"detail":"Batch queue is full"}`

Meaning:

- load exceeds the configured queue capacity

What to check:

- `MAX_BATCH_QUEUE_SIZE`
- request concurrency
- `embedserve_batch_queue_depth`
- `embedserve_batch_overload_rejections_total`

## `POST /embed` returns `503 {"detail":"Embedding request timed out"}`

Meaning:

- the request waited longer than `BATCH_REQUEST_TIMEOUT_MS` for batch completion

What to check:

- `BATCH_REQUEST_TIMEOUT_MS`
- model speed and host saturation
- `embedserve_batch_request_timeouts_total`
- whether queue depth is staying elevated

## GPU memory looks high after load

Important distinction:

- `embedserve_gpu_memory_allocated_bytes` is the primary signal for active allocation drift
- `embedserve_gpu_memory_reserved_bytes` can remain high because PyTorch caches CUDA memory

Use the stability harness and report format before treating reserved memory growth alone as a leak.

## The first startup is slow

Likely cause:

- the pinned model revision is being downloaded into the Hugging Face cache

What to check:

- whether the `hf-cache` Docker volume already exists
- outbound network access on first boot
- logs showing download progress
