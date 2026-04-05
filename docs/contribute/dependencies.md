# Dependency Management

The repository uses human-edited `.in` files and committed `.txt` lockfiles.

## Files You Edit

- `requirements.in`: direct runtime dependencies
- `dev-requirements.in`: direct development tooling dependencies
- `requirements.cpu-linux.in`: local Linux CPU Torch overlay input
- `requirements.cuda-linux.in`: local Linux CUDA Torch overlay input

## Generated Lockfiles

- `requirements.txt`: portable runtime base lockfile
- `dev-requirements.txt`: pinned development tooling lockfile
- `requirements.cpu-linux.txt`: Linux CPU Torch overlay lockfile
- `requirements.cuda-linux.txt`: Linux CUDA Torch overlay lockfile

## Generation Model

1. `pip-compile` resolves each `.in` file to a lockfile.
2. `scripts/filter_portable_requirements.py` keeps the base runtime lockfile portable.
3. `scripts/filter_torch_overlay_requirements.py` writes the Linux CPU overlay on top of `requirements.txt`.
4. `scripts/filter_cuda_overlay_requirements.py` writes the Linux CUDA overlay on top of `requirements.txt`.

## Commands

Regenerate lockfiles without intentionally upgrading pinned versions:

```bash
make deps-compile
```

Refresh pinned versions intentionally:

```bash
make deps-upgrade
```

## Rules

- Do not hand-edit generated `.txt` lockfiles unless you are debugging the generation workflow itself.
- Keep the Docker image and local bootstrap flow aligned with the lockfile model.
- If you change dependency behavior, update the docs and any Docker artifact tests that encode the contract.
