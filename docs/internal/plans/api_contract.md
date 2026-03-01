# Milestone 1 Implementation Plan: Repo Bootstrap and API Contract

  ## Summary

  Milestone 1 should deliver a bootable FastAPI service with a stable HTTP contract, structured logs, Prometheus metrics, and a deterministic stub implementation for POST /embed. The goal is to lock the
  public API before any model loading or GPU inference work begins.

  This milestone should not attempt real tokenization, model loading, batching, CUDA integration, determinism guarantees, or Docker runtime behavior beyond creating the expected directory layout. The
  implementation should optimize for a clean contract, low bootstrap complexity, and easy replacement of the stub embedder in Milestone 2.

  ## Locked Product Decisions

  These decisions are fixed for Milestone 1 and should not be revisited during implementation:

  - POST /embed accepts a JSON object with a single required field: inputs, which must be a list of strings.
  - Single-string request bodies are not accepted in Milestone 1.
  - POST /embed returns 200 OK with a schema-valid stub success response, not 501.
  - Multi-input ordering is stable: response order matches request order, and each item includes its original index.
  - Validation is all-or-nothing: if any input is invalid, the entire request fails.
  - Validation failures use FastAPI/Pydantic-style 422 responses. Do not introduce a custom error envelope in this milestone.
  - Empty lists, empty strings, whitespace-only strings, and oversized input lists are rejected with 422.
  - Success responses include full metadata from day one: model, revision, dim, and usage.tokens.
  - GET /readyz returns 200 in Milestone 1 once the app is booted, because readiness only reflects API availability in stub mode.
  - README wording must explicitly state that Milestone 1 exposes the API contract and stub embeddings only; it does not yet provide real model-backed inference or determinism guarantees.

  ## Scope

  ### In scope

  - Runtime dependency bootstrap for FastAPI service
  - App package structure under app/
  - Placeholder directory structure for scripts/, tests/, and docker/
  - HTTP routes:
      - GET /healthz
      - GET /readyz
      - GET /metrics
      - POST /embed
  - Request/response schemas
  - Structured JSON logging
  - Prometheus metrics
  - Smoke and contract-oriented API tests
  - README updates for temporary Milestone 1 behavior

  ### Out of scope

  - Hugging Face integration
  - Tokenizer usage
  - Real embeddings
  - Dynamic batching
  - Determinism verification logic
  - GPU readiness checks
  - Production Docker image behavior

  ## Target Repository Layout

  Create only the files needed for Milestone 1. Do not create future-only placeholder modules unless they are required to keep package structure valid.

  app/
    __init__.py
    main.py
    settings.py
    schemas.py
    logging.py
    metrics.py
    deps.py
    engine/
      __init__.py
      embedder.py
  tests/
    __init__.py
    test_health.py
    test_metrics.py
    test_embed.py
  scripts/
    .gitkeep
  docker/
    .gitkeep
  docs/internal/plans/
    api_contract.md

  Notes:

  - Keep app/engine/embedder.py as the only engine module in this milestone.
  - Do not add model.py, batcher.py, or utils.py yet unless implementation strictly requires them.
  - docker/ and scripts/ only need to exist structurally in Milestone 1.

  ## Runtime Dependencies

  Populate requirements.txt with the minimal runtime stack:

  - fastapi
  - uvicorn
  - prometheus-client
  - pydantic-settings

  Do not add logging or metrics helper libraries unless there is a concrete gap. Use stdlib logging and prometheus_client directly.

  dev-requirements.txt, pyproject.toml, and Makefile are already aligned for linting, typing, and tests, so only extend them if a concrete Milestone 1 need appears.

  ## Public API Contract

  ### GET /healthz

  Purpose:

  - Liveness only

  Response:

  - Status: 200
  - Body:

  {
    "status": "ok"
  }

  Rules:

  - No dependency on model state
  - No readiness semantics
  - Keep this endpoint trivial and stable

  ### GET /readyz

  Purpose:

  - Milestone 1 readiness for HTTP surface in stub mode

  Response:

  - Status: 200
  - Body:

  {
    "status": "ready",
    "mode": "stub"
  }

  Rules:

  - Always returns 200 once the app is successfully booted
  - Do not pretend GPU or model readiness exists yet
  - The mode: "stub" field is important so operators and tests can distinguish contract-ready from model-ready behavior

  ### GET /metrics

  Purpose:

  - Prometheus text exposition

  Response:

  - Status: 200
  - Content type: Prometheus text exposition
  - Body contains custom app metrics plus default process/python metrics

  Rules:

  - Use a single registry per app instance
  - Expose custom metrics and default collectors from the same registry
  - Ensure tests can instantiate the app multiple times without duplicate collector registration issues

  ### POST /embed

  #### Request schema

  Content type:

  - application/json

  Body shape:

  {
    "inputs": ["first text", "second text"]
  }

  Rules:

  - inputs is required
  - inputs must be a list
  - inputs must contain at least 1 item
  - inputs maximum length is controlled by env var MAX_INPUTS_PER_REQUEST
  - Each item must be a string
  - Each string must be non-empty after trimming surrounding whitespace for validation
  - Extra top-level fields are forbidden

  Validation behavior:

  - Reject empty list with 422
  - Reject empty string with 422
  - Reject whitespace-only string with 422
  - Reject list longer than MAX_INPUTS_PER_REQUEST with 422
  - Reject mixed-validity payloads with 422
  - Reject unknown fields with 422

  #### Success response schema

  Status:

  - 200

  Body shape:

  {
    "data": [
      {
        "index": 0,
        "embedding": [0.123, -0.456, 0.789]
      }
    ],
    "model": "stub-model",
    "revision": "milestone1-stub",
    "dim": 8,
    "usage": {
      "tokens": 0
    }
  }

  Rules:

  - data length must equal len(inputs)
  - data[i].index == i
  - data[i].embedding must be a list of floats
  - All embeddings must have length dim
  - Response order must exactly match request order
  - usage.tokens is always 0 in Milestone 1 because tokenization does not exist yet
  - model and revision come from settings
  - dim comes from the stub embedder constant

  #### Validation failure response

  Use FastAPI/Pydantic-style 422 structure.

  Target contract shape:

  {
    "detail": [
      {
        "loc": ["body", "inputs"],
        "msg": "validation error message",
        "type": "validation_error_type"
      }
    ]
  }

  Rules:

  - Do not add a project-specific error envelope
  - For env-driven max-input validation, preserve the same top-level detail list contract even if the check is performed outside the base Pydantic field definition
  - Do not return partial-success bodies

  ## Internal Types and Interfaces

  Implement these logical types in app/schemas.py:

  - EmbedRequest
      - inputs: list[str]
      - extra = "forbid"
      - validator rejects blank/whitespace-only items
  - EmbeddingItem
      - index: int
      - embedding: list[float]
  - UsageInfo
      - tokens: int
  - EmbedResponse
      - data: list[EmbeddingItem]
      - model: str
      - revision: str
      - dim: int
      - usage: UsageInfo
  - HealthResponse
      - status: Literal["ok"]
  - ReadyResponse
      - status: Literal["ready"]
      - mode: Literal["stub"]

  Implement these runtime interfaces in app/engine/embedder.py:

  - Embedder protocol or base class
      - embed(inputs: list[str]) -> EmbedResponse | tuple[data, dim]
  - StubEmbedder
      - deterministic
      - stateless after initialization
      - configured with model, revision, and fixed dim

  Do not introduce async worker abstractions yet. A synchronous stub embedder wrapped by an async FastAPI route is sufficient.

  ## Settings Contract

  Implement Settings in app/settings.py using pydantic-settings.

  Environment variables for Milestone 1:

  - MODEL_ID
      - default: stub-model
  - MODEL_REVISION
      - default: milestone1-stub
  - LOG_LEVEL
      - default: INFO
  - MAX_INPUTS_PER_REQUEST
      - default: 64

  Rules:

  - Load settings once via cached dependency or lifespan bootstrap
  - Keep settings names uppercase to match README style
  - Do not add model, batching, tokenization, or output env vars yet unless they are explicitly documented as inactive placeholders in README
  - If MAX_INPUTS_PER_REQUEST < 1, fail app startup with a settings validation error

  ## Stub Embedding Behavior

  Use a deterministic algorithm so tests can assert stability without snapshot brittleness.

  Implementation contract:

  - Fixed embedding dimension: 8
  - For each input string:
      - compute SHA-256 over UTF-8 bytes
      - derive 8 float values from the digest
      - scale values into a stable numeric range such as [-1.0, 1.0]
      - round to a small fixed precision such as 6 decimal places before returning
  - Same input must always return the same embedding within the same code version
  - Different inputs should usually produce different vectors
  - No randomness
  - No external dependencies
  - No normalization requirement in Milestone 1

  This is intentionally a stub, not a semantic embedding.

  ## Application Wiring

  Implement create_app() in app/main.py and expose module-level app = create_app().

  Use FastAPI lifespan to bootstrap shared state:

  - load settings
  - configure logging
  - create metrics registry
  - create stub embedder
  - store shared objects under app.state

  Routes:

  - /healthz
      - returns HealthResponse
  - /readyz
      - returns ReadyResponse
  - /metrics
      - returns Prometheus exposition from the app registry
  - /embed
      - validates request body
      - enforces MAX_INPUTS_PER_REQUEST
      - calls StubEmbedder
      - returns EmbedResponse

  Dependency wiring:

  - app/deps.py should expose simple accessors for settings, registry, and embedder from request.app.state
  - Do not use global singletons for the metrics registry

  ## Metrics Plan

  Implement metrics in app/metrics.py.

  Custom metrics to expose:

  - embedserve_http_requests_total
      - type: counter
      - labels: method, route, status_code
  - embedserve_http_request_duration_seconds
      - type: histogram
      - labels: method, route
  - embedserve_app_ready
      - type: gauge
      - labels: mode
      - value: 1 for mode="stub"

  Also register default collectors:

  - process metrics
  - platform metrics
  - Python GC metrics

  Rules:

  - Use route templates, not raw URLs, for the route label
  - Keep label cardinality low
  - Record metrics for all public endpoints, including /metrics
  - Set embedserve_app_ready{mode="stub"} 1 during startup
  - Do not add batching or model-specific metrics yet

  ## Logging Plan

  Implement structured JSON logging in app/logging.py using stdlib logging.

  Log shape for request/access logs:

  - timestamp
  - level
  - logger
  - event
  - request_id
  - method
  - path
  - status_code
  - duration_ms

  Rules:

  - Generate a request ID per HTTP request in middleware
  - Emit one access log entry per request
  - Include exceptions in JSON logs for unhandled server errors
  - Disable or avoid duplicate default Uvicorn access logs if they conflict with custom structured logs
  - Respect LOG_LEVEL

  Do not add distributed tracing or external logging backends in this milestone.

  ## Implementation Sequence

  ### 1. Bootstrap dependencies and directories

  - Add runtime dependencies to requirements.txt
  - Create app/, app/engine/, tests/, scripts/, and docker/
  - Add __init__.py where needed
  - Keep placeholder directories minimal with .gitkeep

  ### 2. Add settings and schema modules

  - Implement Settings
  - Implement request/response models
  - Validate blank strings and config sanity

  ### 3. Add stub embedder

  - Implement deterministic StubEmbedder
  - Hard-code dim = 8
  - Return metadata from settings
  - Return usage.tokens = 0

  ### 4. Add logging and metrics plumbing

  - Create JSON logging config
  - Create metrics registry and helpers
  - Add request middleware for timing, request IDs, logs, and metric updates

  ### 5. Build FastAPI app and routes

  - Add create_app()
  - Bootstrap shared state in lifespan
  - Implement all four routes
  - Ensure uvicorn app.main:app works without additional code changes

  ### 6. Add tests

  - Add endpoint smoke tests
  - Add contract tests for /embed
  - Add env override test for MAX_INPUTS_PER_REQUEST

  ### 7. Update README and internal docs

  - Update README to describe stub-mode behavior clearly
  - Align endpoint documentation with actual request and response shapes
  - Note that real inference and determinism land in later milestones

  ## Test Cases and Scenarios

  At minimum, implement the following tests.

  ### Health and readiness

  - GET /healthz returns 200 and {"status":"ok"}
  - GET /readyz returns 200 and {"status":"ready","mode":"stub"}

  ### Metrics

  - GET /metrics returns 200
  - Response content type is Prometheus-compatible
  - Response body contains:
      - embedserve_http_requests_total
      - embedserve_http_request_duration_seconds
      - embedserve_app_ready
  - Response body also contains at least one default process or Python metric

  ### Embed success

  - Single valid input returns 200
  - Multi-input request returns 200
  - data preserves request order
  - data[i].index == i
  - Every embedding length equals dim
  - Top-level model, revision, dim, and usage.tokens are present
  - Repeating the same request twice returns identical stub embeddings

  ### Embed validation

  - Missing inputs returns 422
  - Non-list inputs returns 422
  - Empty list returns 422
  - Empty string item returns 422
  - Whitespace-only item returns 422
  - Extra top-level field returns 422
  - Mixed-validity list returns 422

  ### Config-driven input cap

  - 2 inputs succeeds

  ## README Updates

  Update README so Milestone 1 does not over-claim capabilities.

  Required wording changes:

  - State clearly that current /embed behavior is a deterministic stub used to lock the API contract
  - State clearly that real transformer inference begins in Milestone 2
  - Replace or soften any wording that implies current GPU-backed inference already exists
  - Replace or soften determinism wording so it does not imply model-backed repeatability is already implemented
  - Keep the long-term product direction, but mark it as planned rather than already shipped

  README endpoint section should document the actual Milestone 1 /embed request and response shape.

  ## Acceptance Criteria

  Milestone 1 is complete when all of the following are true:

  - uvicorn app.main:app starts successfully from the repo root
  - GET /healthz returns 200
  - GET /readyz returns 200 with mode: "stub"
  - GET /metrics returns valid Prometheus exposition
  - POST /embed returns a schema-valid stub success response
  - Validation failures return 422 with FastAPI-style detail
  - Pytest suite passes locally
  - README describes current behavior accurately

  ## Assumptions and Defaults

  These defaults are chosen unless later explicitly changed by a new planning pass:

  - Request shape is object-based, not bare array-based
  - inputs accepts list-only, not string-or-list dual mode
  - Success responses use 200 OK
  - Validation is full-request rejection, not partial success
  - Validation failures use FastAPI/Pydantic-style 422
  - MAX_INPUTS_PER_REQUEST is public config in Milestone 1
  - Default MAX_INPUTS_PER_REQUEST is 64
  - Default MODEL_ID is stub-model
  - Default MODEL_REVISION is milestone1-stub
  - Stub embedding dimension is 8
  - usage.tokens is always 0 until tokenizer work exists
  - Readiness in Milestone 1 means “HTTP contract is up in stub mode,” not “model is loaded”
