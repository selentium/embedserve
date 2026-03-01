# Milestone 2 Implementation Plan: Model Loading, Tokenization, and Single-Request Inference

  ## Summary

  Milestone 2 should replace the stub embedder with a real Hugging Face encoder pipeline while preserving the existing /embed success schema and request validation rules from Milestone 1. The service
  should boot even when model initialization fails, but readiness must now mean “the configured tokenizer and model loaded successfully, a warmup inference succeeded, and the target device is usable.”

  This milestone remains intentionally narrow. It does not add batching, request timeouts, retry loops, stub fallback, Docker reproducibility, or determinism guarantees beyond standard inference hygiene
  (model.eval() and torch.inference_mode()).

  ## Locked Product Decisions

  - Baseline model compatibility is limited to standard Hugging Face sentence-transformer style encoder models that expose token embeddings through last_hidden_state.
  - Use transformers plus torch directly. Do not add the sentence-transformers package in this milestone.
  - Pooling is fixed to attention-mask-aware mean pooling. It is not configurable in Milestone 2.
  - The process boots even if model or device initialization fails.
  - Missing required runtime dependencies such as torch or transformers is a hard startup failure, not an unready state.
  - GET /healthz remains a pure liveness endpoint and still returns 200.
  - GET /readyz becomes a true readiness endpoint and returns 503 when the model runtime is not ready.
  - POST /embed returns 503 while the service is unready. Do not fall back to stub embeddings.
  - usage.tokens means the total number of tokens actually fed into the model after truncation and special-token insertion, summed across all inputs and excluding padding.
  - There is no automatic retry after initialization failure. Recovery requires fixing config/environment and restarting the process.
  - NORMALIZE_EMBEDDINGS and OUTPUT_DTYPE are part of the stable public configuration surface and must be documented in README.md.
  - DTYPE is the single runtime floating-point precision control for model loading and inference in Milestone 2. There is no separate compute-dtype or mixed-precision setting yet.
  - First successful startup may require live Hugging Face network access if the configured model revision is not already cached locally.
  - MODEL_REVISION is syntactically enforced as a full 40-character git commit hash during settings validation.
  - Request processing remains all-or-nothing. Mixed-validity payloads still fail the whole request.

  ## Scope

  ### In scope

  - Real model and tokenizer loading from Hugging Face using pinned MODEL_ID and MODEL_REVISION
  - Device and dtype selection
  - Tokenization with configurable max length and truncation behavior
  - Forward pass, pooling, optional L2 normalization, output dtype conversion
  - Readiness state and startup failure handling
  - Updated response metadata and token counting
  - Unit and API tests covering new behavior
  - README updates for real inference behavior and config surface

  ### Out of scope

  - Dynamic batching
  - Stub fallback on failure
  - Background model reload or retry
  - Per-request timeout controls
  - Determinism verification policy
  - Docker or CUDA image pinning
  - Performance tuning beyond correctness and basic safety

  ## Public API and Config Contract

  ### Existing request contract preserved

  POST /embed keeps the Milestone 1 request body:

  {
    "inputs": ["first text", "second text"]
  }

  These rules stay unchanged:

  - inputs is required and must be a list of strings
  - Empty lists are rejected
  - Empty or whitespace-only strings are rejected
  - Extra top-level fields are rejected
  - Validation failures use FastAPI-style 422
  - Response order must match request order
  - Every response still includes X-Request-ID

  ### New and updated environment variables

  Keep existing variables:

  - MODEL_ID
  - MODEL_REVISION
  - LOG_LEVEL
  - MAX_INPUTS_PER_REQUEST

  Add these variables and treat them as public config:

  - DEVICE
      - Default: cpu
      - Allowed forms: cpu, cuda, cuda:N
  - DTYPE
      - Default: float32
      - Allowed values: float32, float16, bfloat16
  - MAX_LENGTH
      - Default: 512
      - Must be >= 1
  - TRUNCATE
      - Default: true
  - NORMALIZE_EMBEDDINGS
      - Default: true
  - OUTPUT_DTYPE
      - Default: float32
      - Allowed values: float32, float16

  Validation split:

  - Pure schema/env validation still fails startup immediately.
      - Examples: invalid boolean, invalid enum string, MAX_LENGTH < 1, MAX_INPUTS_PER_REQUEST < 1, MODEL_REVISION not matching a full 40-character hexadecimal commit hash
  - Runtime-dependent validation marks the service unready instead of crashing startup.
      - Examples: unavailable CUDA device, unsupported dtype/device combo, model load failure, tokenizer/model incompatibility
  - Missing required Python dependencies still fails startup immediately.
      - Examples: ImportError for torch or transformers

  ### Default model assumption

  Use a small sentence-transformer model as the documented development default.

  - MODEL_ID default should become sentence-transformers/all-MiniLM-L6-v2
  - MODEL_REVISION must be an exact Hugging Face commit hash pinned during implementation
  - MODEL_REVISION must match ^[0-9a-fA-F]{40}$
  - Reject branch names, tags, and short SHAs at settings validation time
  - Do not leave MODEL_REVISION as a branch or tag name
  - The implementation should pass the revision string through unchanged and surface it in responses exactly as configured

  ### /readyz response contract

  Change readiness from stub-only to model-runtime-aware.

  Ready response:

  - HTTP 200
  - Body shape:

  {
    "status": "ready",
    "mode": "model",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "EXACT_COMMIT_HASH",
    "device": "cpu",
    "dtype": "float32"
  }

  Unready response:

  - HTTP 503
  - Body shape:

  {
    "status": "not_ready",
    "mode": "model",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "EXACT_COMMIT_HASH",
    "device": "cuda:0",
    "dtype": "float16",
    "reason": "device_unavailable",
    "detail": "CUDA device cuda:0 is not available"
  }

  Rules:

  - reason is a closed enum, not a free-form string
  - Allowed reason values: initialization_failed, device_unavailable, dtype_unsupported, model_load_failed, tokenizer_incompatible, warmup_failed
  - Public reason values should map from internal failures as follows:
      - device selection or CUDA availability failures -> device_unavailable
      - unsupported DEVICE plus DTYPE combinations -> dtype_unsupported
      - Hugging Face download, revision resolution, or model construction failures -> model_load_failed
      - tokenizer construction failures or tokenizer/model input-shape incompatibilities -> tokenizer_incompatible
      - warmup tokenization or warmup forward-pass failures after successful construction -> warmup_failed
      - use initialization_failed only for unexpected runtime-initialization failures that do not fit the narrower categories
  - detail should be a short sanitized message, not a traceback dump
  - Full exception details go to logs only
  - mode should be model for both ready and unready cases
  - Preserve X-Request-ID on both 200 and 503

  ### /embed success contract

  Keep the Milestone 1 success shape unchanged:

  {
    "data": [
      {
        "index": 0,
        "embedding": [0.1, 0.2, 0.3]
      }
    ],
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "EXACT_COMMIT_HASH",
    "dim": 384,
    "usage": {
      "tokens": 8
    }
  }

  Rules:

  - model and revision echo config values
  - dim is the pooled embedding width
  - usage.tokens is the sum of non-padding tokens actually processed by the model after truncation
  - embedding values remain JSON numbers, regardless of OUTPUT_DTYPE
  - OUTPUT_DTYPE affects numeric precision before serialization, not the wire format

  ### /embed unready behavior

  While unready, POST /embed returns:

  - HTTP 503
  - Body: standard FastAPI-style error payload via HTTPException

  Example:

  {
    "detail": "Model is not ready: initialization_failed"
  }

  Do not introduce a custom error envelope for this path.

  Validation precedence clarification:

  - Preserve Milestone 1 precedence for request-shape validation and MAX_INPUTS_PER_REQUEST only
  - Schema validation failures and over-input-count failures still return 422 even when runtime.ready is false
  - Tokenizer-dependent validation such as TRUNCATE=false overlength checks runs only when runtime.ready is true
  - If runtime.ready is false, do not attempt tokenizer-backed length checks just to decide between 422 and 503
  - This avoids requiring a usable tokenizer or embedder in the unready state

  ## Tokenization and Inference Semantics

  ### Effective max length

  Compute an effective max length per runtime:

  - Start from configured MAX_LENGTH
  - If tokenizer.model_max_length is a sane finite cap, use min(MAX_LENGTH, tokenizer.model_max_length)
  - If the tokenizer reports a sentinel “effectively unbounded” value, use configured MAX_LENGTH directly

  Do not silently exceed the configured cap.

  ### Overlength input behavior

  If TRUNCATE=true:

  - Tokenize with truncation enabled
  - Truncate each input independently to the effective max length
  - Return success if all other validation passes

  If TRUNCATE=false:

  - Perform a length-check pass before padded batch construction and before model forward
  - The length-check path should avoid large padded tensor allocation for obviously oversize inputs
  - It may inspect items individually or use tokenizer facilities that return lengths without creating model inputs
  - If any input would exceed the effective max length after special tokens are added, reject the whole request with 422
  - Error location should point to the offending item when possible: ["body", "inputs", <index>]
  - Error message should include both actual token length and limit
  - Raise RequestValidationError so the response stays in the existing FastAPI 422 shape instead of introducing a custom error body

  ### Pooling

  Use standard attention-mask-aware mean pooling over last_hidden_state.

  Formula:

  - Multiply token embeddings by expanded attention mask
  - Sum across the sequence dimension
  - Divide by the count of non-padding tokens
  - Clamp the denominator to avoid division by zero
  - Include special tokens if they are present in the non-padding mask
  - Exclude padding tokens only

  ### Normalization

  If NORMALIZE_EMBEDDINGS=true:

  - Apply L2 normalization after pooling
  - Use an epsilon guard so near-zero vectors do not create NaNs

  If NORMALIZE_EMBEDDINGS=false:

  - Return raw pooled vectors

  ### Output dtype conversion

  DTYPE semantics for Milestone 2:

  - DTYPE controls the model runtime dtype for loading and inference
  - There is no separate compute dtype, autocast mode, or mixed-precision policy in this milestone
  - Integer tokenizer outputs remain integer tensors; floating-point model outputs remain in DTYPE until the final output cast
  - Unsupported DEVICE plus DTYPE combinations are treated as initialization failures and must be caught before the service is marked ready

  After pooling and optional normalization:

  - Move embeddings to CPU
  - Cast to OUTPUT_DTYPE
  - Convert to list[list[float]]

  Rules:

  - OUTPUT_DTYPE=float16 is allowed even on CPU because it is only the final returned precision
  - Precision tests should compare expected rounded numeric effects, not Python type identity
  - Do not round values manually in the service

  ## Internal Architecture

  ## File and module shape

  Keep the repo structure small and incremental.

  Update existing modules:

  - app/settings.py
  - app/schemas.py
  - app/main.py
  - app/deps.py
  - app/metrics.py
  - app/logging.py
  - app/engine/embedder.py

  Add one new runtime-state module:

  - app/runtime.py

  Do not add batching or worker modules in Milestone 2.

  ## Runtime state model

  Create a small runtime-state container in app/runtime.py to hold:

  - ready: bool
  - mode: Literal["model"]
  - model_id: str
  - revision: str
  - device: str
  - dtype: str
  - reason: str | None
  - detail: str | None
  - embedder: Embedder | None

  This runtime object becomes the single source of truth for /readyz and /embed.

  ## Embedder implementation

  Replace StubEmbedder with a real TransformerEmbedder that owns:

  - tokenizer
  - model
  - resolved device
  - effective max length
  - truncation setting
  - normalization setting
  - output dtype
  - a process-local inference lock

  Implementation rules:

  - Call AutoTokenizer.from_pretrained(..., revision=..., use_fast=True, trust_remote_code=False)
  - Call AutoModel.from_pretrained(..., revision=..., trust_remote_code=False, torch_dtype=resolved_dtype)
  - Move the model to the resolved device before warmup and request inference
  - Move tokenized model inputs to the resolved device before each forward pass
  - Set model.eval()
  - Run inference under torch.inference_mode()
  - Pass the tokenizer output dictionary directly into the model
  - Require last_hidden_state to exist
  - Derive dim from the pooled output width, not from a hard-coded constant

  To keep the default test suite offline and deterministic:

  - Isolate model/tokenizer construction behind a small loader/helper boundary that can be replaced in tests
  - Route startup initialization through that helper rather than calling transformers constructors inline from the lifespan body
  - Default tests should use fake tokenizer/model objects and monkeypatched loaders; they must not download weights

  ## Startup initialization flow

  In lifespan:

  1. Load settings.
  2. Configure logging.
  3. Create metrics registry.
  4. Initialize readiness gauge to 0 for mode="model".
  5. Attempt runtime initialization inside a guarded helper.
  6. On success:
      - build tokenizer and model
      - validate device and dtype compatibility
      - run one warmup inference on a short fixed string
      - store a ready runtime with embedder
      - set readiness gauge to 1
      - log one structured initialization-success event
  7. On failure:
      - store an unready runtime with embedder=None
      - keep readiness gauge at 0
      - log one structured initialization-failure event with traceback

  Dependency loading rule:

  - ImportError for required runtime packages such as torch or transformers aborts startup immediately
  - Initialization failures that happen after required imports succeed transition the service to unready runtime state instead

  Warmup is mandatory. Readiness should not mean “weights downloaded” only; it should mean the actual tokenization plus forward path works.
  First successful startup may download model artifacts from Hugging Face if they are not already cached.

  ## Route wiring

  GET /healthz

  - unchanged
  - pure liveness only

  GET /readyz

  - read runtime state
  - return 200 with ready schema if runtime.ready
  - return 503 with unready schema otherwise
  - document both 200 and 503 responses in route metadata/OpenAPI, not just in prose

  POST /embed

  - validate request body as today
  - enforce MAX_INPUTS_PER_REQUEST
  - preserve Milestone 1 validation precedence for malformed payloads and over-input-count payloads: those still return 422 even if runtime.ready is false
  - short-circuit with 503 if runtime.ready is false
  - execute model inference through the embedder
  - preserve response ordering and metadata shape
  - document the 503 response in route metadata/OpenAPI

  ## Concurrency policy

  Milestone 2 is single-request inference only, so correctness should win over throughput.

  - Serialize inference with a single process-local threading.Lock or equivalent thread-safe synchronous lock inside the embedder
  - Tokenization, forward pass, pooling, normalization, and output conversion should all occur under that lock
  - Do not attempt concurrent forwards on the same shared model instance yet
  - Do not implement request merging or micro-batching

  This keeps behavior predictable until Milestone 4 introduces batching explicitly.

  ## Async boundary

  Do not run heavy PyTorch work directly on the event loop.

  - Keep the embedder API synchronous
  - Invoke it from the route through fastapi.concurrency.run_in_threadpool
  - Leave cancellation handling simple in Milestone 2
  - If a client disconnects mid-inference, the in-flight inference may still complete internally; explicit cancellation control is deferred to Milestone 4

  ## Metrics and Logging Changes

  ## Metrics

  Keep existing HTTP metrics and the existing readiness metric name.

  Changes:

  - embedserve_app_ready{mode="model"} should be 1 when ready and 0 when unready
  - Remove stub-mode assumptions from tests and docs
  - Keep embedserve_http_requests_total and embedserve_http_request_duration_seconds unchanged

  Do not add new model-specific metrics in this milestone. That avoids expanding the compatibility surface prematurely.

  ## Logging

  Add structured startup events in addition to the existing access logs.

  Implementation note:

  - app/logging.py must be updated so these startup fields are actually emitted by the JSON formatter; the current formatter only includes a small fixed field set plus exception details

  Initialization success log fields:

  - event
  - model
  - revision
  - device
  - dtype
  - max_length
  - truncate
  - normalize_embeddings
  - output_dtype

  Initialization failure log fields:

  - same fields as above
  - exception_type
  - exception_message
  - exception_traceback

  Access logging stays as implemented in Milestone 1.

  ## Edge Cases and Failure Modes

  The implementation plan should explicitly handle these cases.

  - DEVICE=cuda or cuda:N but CUDA is unavailable: boot unready
  - DTYPE=float16 or bfloat16 on unsupported hardware: boot unready
  - torch or transformers cannot be imported: fail startup immediately
  - MODEL_REVISION is not a full 40-character hexadecimal commit hash: fail settings validation and abort startup
  - MODEL_ID exists but MODEL_REVISION does not resolve: boot unready
  - tokenizer loads but model forward output has no last_hidden_state: boot unready
  - tokenizer cannot produce padded batched inputs cleanly: boot unready
  - TRUNCATE=false and one item exceeds max length: reject the entire request with 422
  - TRUNCATE=false length checking must not allocate large padded model-input tensors before rejection
  - TRUNCATE=true and only some items exceed max length: truncate only those items and succeed
  - MAX_INPUTS_PER_REQUEST still applies before inference
  - request validation errors keep returning 422 even when the runtime is unready
  - this 422-before-503 rule applies only to request-shape validation and MAX_INPUTS_PER_REQUEST, not tokenizer-dependent overlength checks
  - blank-string validation remains trim-aware for rejection only; accepted strings are not modified
  - usage.tokens must be deterministic for the same tokenizer settings and request payload
  - normalization must not produce NaNs
  - output conversion must always happen on CPU before JSON serialization
  - there is no partial success format
  - there is no stub fallback
  - there is no auto-retry after failed initialization

  ## Implementation Sequence

  The order matters because the current codebase creates settings and runtime state inside lifespan, imports the embedder from app.main/app.deps, and the default test suite constructs TestClient(create_app()) with no network or GPU assumptions.

  Recommended order:

  1. Add the runtime-state module and dependency seam first.
      - Introduce app/runtime.py and a get_runtime dependency
      - Move app.state.embedder toward app.state.runtime so routes can branch on readiness without requiring an embedder object to exist
      - Keep existing stub behavior temporarily while this seam lands
  2. Update metrics helpers so model-mode readiness can be represented as both 0 and 1 before changing startup behavior.
      - Initialize embedserve_app_ready{mode="model"} to 0 during startup
      - Add or replace helpers so unready state is exported explicitly, not only the ready case
  3. Add startup-initialization helper boundaries before replacing the embedder implementation.
      - Keep model/tokenizer construction behind a narrow loader or initializer function
      - Make that helper injectable or monkeypatchable from tests
      - Defer torch/transformers imports to the runtime-initialization path so missing dependencies behave as startup failures rather than surprising module-import failures
  4. Extend Settings with DEVICE, DTYPE, MAX_LENGTH, TRUNCATE, NORMALIZE_EMBEDDINGS, OUTPUT_DTYPE, and strict MODEL_REVISION validation.
      - Do this after the test seam exists so the suite does not accidentally start downloading models or require GPU/online setup just from default startup
  5. Add ready and unready response schemas and update route wiring to use runtime state.
      - Update /readyz to return 200 or 503 from runtime state
      - Update /embed to preserve request-shape validation and MAX_INPUTS_PER_REQUEST checks before readiness checks
      - Remove route-time assumptions that app.state.embedder is always present
  6. Replace StubEmbedder with TransformerEmbedder behind the loader/helper seam.
      - Keep the synchronous embedder API
      - Add the process-local inference lock
      - Implement tokenization, forward, pooling, normalization, and output conversion
  7. Add guarded startup initialization and mandatory warmup.
      - Build tokenizer/model through the helper
      - Validate device/dtype compatibility
      - Run warmup
      - Store either ready runtime with embedder or unready runtime with reason/detail
  8. Update logging so startup initialization events actually emit the planned structured fields.
  9. Update tests to cover ready and unready startup, tokenizer behavior, /embed 503 behavior, and OpenAPI response metadata through fakes and monkeypatching.
      - This should happen before changing the documented default model config used in examples
  10. Update route metadata/OpenAPI so documented 200 and 503 responses match runtime behavior.
  11. Update README.md to document the new config surface, ready/unready behavior, truncation semantics, and the limited determinism claim.
  12. Keep requirements unpinned in the same style as the current repo, but add torch and transformers. Version pinning remains a Milestone 5 responsibility.

  ## Test Plan

  Add or update tests so Milestone 2 is fully covered without requiring internet or GPU in the default suite.

  ### Settings and startup tests

  - valid new env vars create settings successfully
  - invalid enum values fail settings validation
  - invalid MAX_LENGTH fails settings validation
  - non-hash MODEL_REVISION values fail settings validation
  - invalid CUDA target causes startup to boot unready, not crash
  - model loader exception causes startup to boot unready, not crash
  - successful initialization sets embedserve_app_ready{mode="model"} 1

  ### /readyz tests

  - ready runtime returns 200
  - ready body includes status, mode, model, revision, device, dtype
  - failed initialization returns 503
  - unready body includes reason enum value and short detail
  - both paths include X-Request-ID
  - OpenAPI advertises both the 200 and 503 readiness responses

  ### /embed success tests

  - a valid request returns 200
  - response model, revision, and dim reflect runtime config
  - response ordering still matches request ordering
  - usage.tokens equals the sum of non-padding processed tokens
  - MODEL_REVISION is echoed unchanged when set to a commit-hash-like value
  - NORMALIZE_EMBEDDINGS=true yields vectors with norm approximately 1.0
  - NORMALIZE_EMBEDDINGS=false yields non-normalized vectors
  - OUTPUT_DTYPE=float16 changes returned numeric precision relative to float32
  - repeated requests with the same fake model return stable outputs for the same runtime config

  ### /embed failure tests

  - unready runtime returns 503
  - malformed payloads still return 422 when runtime is unready
  - TRUNCATE=false and overlength input returns 422
  - TRUNCATE=false failure points to offending input index
  - over-input-count behavior from Milestone 1 still returns 422
  - all error paths preserve X-Request-ID
  - OpenAPI advertises the 503 /embed response

  ### Tokenization behavior tests

  Use a fake tokenizer and fake model so tests are deterministic and offline.

  - truncation enabled shortens long sequences to effective max length
  - truncation disabled rejects long sequences before model forward
  - truncation disabled length checking does not create padded batch model inputs for oversized requests
  - usage.tokens uses post-truncation lengths, not pre-truncation lengths
  - effective max length respects both configured MAX_LENGTH and tokenizer cap
  - padding tokens are excluded from mean pooling and token counts

  ### Optional non-default tests

  - a @pytest.mark.gpu smoke test may be added later for CUDA environments, but it must not be required for the default local suite
  - no test in the default suite should download real model weights

  ### Test harness requirements

  - the app startup path must expose a narrow seam for injecting a fake runtime initializer or fake model/tokenizer loader
  - TestClient-based tests should be able to exercise both ready and unready startup paths without network access
  - startup tests should verify that first-boot network access is a runtime behavior only, not a requirement for the default automated suite

  ## README and Documentation Updates

  Update README.md so it no longer describes the service as stub-only.

  Required changes:

  - document the real inference path and that Milestone 1 stub behavior is gone
  - add the full environment-variable table for Milestone 2
  - document exact semantics of TRUNCATE, MAX_LENGTH, NORMALIZE_EMBEDDINGS, and OUTPUT_DTYPE
  - document that MODEL_REVISION should be an exact commit hash
  - document that first startup may download model artifacts from Hugging Face if they are not already cached
  - update /readyz examples for both ready and unready states
  - keep the determinism note narrow: Milestone 2 does not yet guarantee strict reproducibility
  - note that batching is not implemented yet and inference is serialized per process

  ## Assumptions and Defaults Chosen

  - Default model family is sentence-transformer style encoder models only
  - Default documented dev model is sentence-transformers/all-MiniLM-L6-v2
  - The exact default MODEL_REVISION hash is a factual lookup to perform during implementation, not a design choice to defer
  - DEVICE default is cpu
  - DTYPE default is float32
  - MAX_LENGTH default is 512
  - TRUNCATE default is true
  - NORMALIZE_EMBEDDINGS default is true
  - OUTPUT_DTYPE default is float32
  - MAX_INPUTS_PER_REQUEST stays at its Milestone 1 value unless separate performance evidence justifies changing it
  - sentence-transformers dependency, auto-retry, batching, and stub fallback are all explicitly excluded from Milestone 2
