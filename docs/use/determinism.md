# Determinism and Compatibility

EmbedServe applies a best-effort numerical-stability policy at startup.

## Runtime Policy

- `random.seed(0)`
- `torch.manual_seed(0)`
- `torch.cuda.manual_seed_all(0)` when CUDA is available
- `torch.backends.cudnn.deterministic=True`
- `torch.backends.cudnn.benchmark=False`
- `torch.use_deterministic_algorithms(True, warn_only=True)` when supported

## What Is Guaranteed

- The service aims for bounded numerical drift under a fixed environment.
- The policy is intended to improve repeatability for the same host and runtime stack.

## What Is Not Guaranteed

- Bitwise-identical outputs are not promised.
- Results are not portable across different GPUs, drivers, PyTorch versions, transformers versions, or other low-level runtime changes.

## Conditions For Meaningful Verification

Keep all of the following fixed:

- same host and accelerator
- same `MODEL_ID` and `MODEL_REVISION`
- same runtime settings: `DEVICE`, `DTYPE`, `MAX_LENGTH`, `TRUNCATE`, `NORMALIZE_EMBEDDINGS`, `OUTPUT_DTYPE`
- warmed runtime before measurement

## Verification Command

Run live verification against a started server:

```bash
make verify-determinism
```

Intentional failure example:

```bash
make verify-determinism VERIFY_DETERMINISM_ARGS="--min-cosine-similarity 1.0001"
```

The verifier exits non-zero on threshold or operational failure.

## Compatibility Reminder

Use this policy as a bounded-repeatability tool, not as a promise that outputs match across machines or upgrades.
