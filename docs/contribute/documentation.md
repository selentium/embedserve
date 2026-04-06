# Documentation Maintenance

Documentation in this repository is part of the product contract, not an afterthought.

## Public Docs Rules

- Keep the root `README.md` short and link outward to canonical docs.
- Keep user-facing reference material in `docs/use/`.
- Keep contributor workflow and architecture material in `docs/contribute/`.

## Canonical Sources

- API contract: `docs/use/api.md`
- configuration surface: `docs/use/configuration.md`
- deployment workflow: `docs/use/deployment.md`
- contributor workflow: `docs/contribute/`
- benchmark data and format: `docs/benchmarks/BENCHMARK.md`
- stability data and format: `docs/benchmarks/STABILITY.md`

Avoid maintaining the same reference data in multiple places unless one copy is explicitly a summary that links to the canonical page.

## Required Updates When Behavior Changes

Update docs in the same change when you alter:

- public endpoints or payloads
- environment variables or defaults
- documented metric names, labels, or semantics
- Docker workflow expectations
- benchmark or stability claims

## Review Standard

Before considering a docs-affecting change complete, check that:

- the implementation, tests, and docs describe the same behavior
- user-facing docs remain understandable on their own
- contributor docs still reflect the actual bootstrap and verification workflow
