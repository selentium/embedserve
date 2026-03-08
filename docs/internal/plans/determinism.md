## Milestone 3 Plan: Determinism Policy and Live Verification

### Summary

Implement Milestone 3 as a measured numerical-stability policy (not bitwise determinism), enforced by a live-server verifier script and documented with explicit environment assumptions.
No HTTP API contract changes; additions are runtime policy behavior, CLI verification tooling, README policy wording, and deterministic validation tests.

### Public interfaces and contract changes

- Add a new CLI tool: `scripts/verify_determinism.py`.
- Add a new developer command in `Makefile`: `make verify-determinism`.
- Update README title and determinism wording so public claims match enforced policy boundaries.
- No changes to `/embed`, `/readyz`, `/metrics`, request/response schemas, or existing settings env vars.

### Implementation plan

1. Runtime best-effort determinism policy in inference startup path.
- Apply deterministic toggles with safe behavior:
`random.seed(0)`, `torch.manual_seed(0)`, `torch.cuda.manual_seed_all(0)` when CUDA is available, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, and `torch.use_deterministic_algorithms(True, warn_only=True)` when available.
- Keep policy explicitly best effort: do not fail startup purely because strict bitwise determinism cannot be guaranteed.
- Emit structured startup log fields indicating determinism policy activation and effective mode (numerical_stability, seed, deterministic algorithms enabled/unsupported).

2. Live verification script behavior (`scripts/verify_determinism.py`).
- Default endpoint: `http://127.0.0.1:8000/embed`. Send repeated identical payloads.
- Default corpus: built-in fixed mixed text set (8-16 strings with varied lengths/chars), with optional `--inputs-file` override.
- Default verifier settings: `--iterations 25`, `--max-abs-diff 2e-3`, `--min-cosine-similarity 0.9995`, `--timeout-seconds 10`.
- Compute metrics against the first successful response as baseline:
`max_abs_diff` is maximum absolute float delta across all coordinates and iterations; `min_cosine_similarity` is minimum cosine similarity per input vector between baseline and each repeated run.
- Enforce schema/dimension consistency across runs; treat mismatches as verification failure.
- Exit codes: `0` pass, `1` threshold failure, `2` operational/protocol failure (HTTP non-200, timeout, malformed payload/response).
- Output modes: human-readable summary by default, `--json` emits machine-readable results including config, measured metrics, pass/fail, and failure reason.

3. Developer workflow integration.
- Add `make verify-determinism` to run the script with overridable base URL, thresholds, and timeout via args/env passthrough.
- Do not add a pytest wrapper target; keep verification as an explicit live-server command.

4. README and milestone alignment.
- Update determinism section with exact claim boundaries: bounded numerical drift under fixed environment, no bitwise guarantee, and non-portability across hardware/driver/library changes.
- Document required conditions for meaningful checks (same host, same model+revision, same settings, warmed runtime).
- Add explicit commands for:
standard pass run with defaults and intentional fail run (for example impossible cosine threshold > 1.0) to prove non-zero exit behavior.

### Test plan

- Add automated unit tests for deterministic-policy helper behavior (seed calls, backend flags, graceful handling when APIs are unavailable).
- Add script tests for metric math and pass/fail gating using mocked HTTP responses (no real server required in default suite).
- Add script CLI tests for defaults and argument overrides (iterations, thresholds, timeout, `--json`, `--inputs-file`).
- Add script failure-path tests: non-200 response, timeout/transport error, shape mismatch, and malformed response payload.
- Keep existing API tests unchanged except minimal updates if startup logging/policy hooks alter test setup expectations.
- Manual acceptance checks for milestone sign-off:
1. Start live server.
2. Run `make verify-determinism` and confirm exit `0` on known-good setup.
3. Run intentional-fail command and confirm non-zero exit.
4. Confirm README text matches enforced policy and default thresholds.

### Assumptions and defaults

- Determinism scope is numerical stability only, not bitwise reproducibility.
- No new public env vars are introduced for determinism tuning in this milestone.
- Script is runtime-supported and uses `httpx` from `requirements.txt`.
- Verification is sequential (no concurrency/batching) to isolate determinism signal from future Milestone 4 behavior.
- Default thresholds are intentionally environment-agnostic for this milestone and can be overridden via CLI flags.
