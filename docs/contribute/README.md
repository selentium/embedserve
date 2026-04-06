# Contributor Guide

EmbedServe has a small public API surface but several operational guarantees, so contributor docs focus on staying aligned with the actual contract.

## Start Here

- [Local Setup](getting-started.md)
- [Architecture](architecture.md)
- [Testing and Verification](testing.md)
- [Dependency Management](dependencies.md)
- [Worktrees](worktrees.md)
- [Documentation Maintenance](documentation.md)

## Repo Layout

- `app/`: service implementation
- `tests/`: unit and integration-style test coverage
- `scripts/`: live verification, benchmark, load-test, and helper scripts
- `docker/`: Dockerfile and Compose workflow
- `docs/use/`: public user and operator docs

## Contributor Expectations

- Keep public docs aligned with the implementation and tests.
- Treat metric names and labels as compatibility-sensitive once documented.
- Preserve the pinned dependency workflow unless intentionally changing it.
- Prefer explicit, measurable operational claims over vague descriptions.
