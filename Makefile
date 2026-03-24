.PHONY: help bootstrap-dev worktree-create worktree-remove start format lint typecheck test verify-determinism verify-batching bench-10k audit hadolint pre-commit-install pre-commit-run check

VENV := ./venv/bin
PRE_COMMIT_HOME := /tmp/pre-commit-cache
PYTHON ?= python3
WORKTREE ?=
BASE ?= HEAD
WORKTREE_PATH ?=
SETUP ?= 1
COPY_ENV ?= 1
FORCE ?= 0
DELETE_BRANCH ?= 0
VERIFY_DETERMINISM_URL ?= http://127.0.0.1:8000/embed
VERIFY_DETERMINISM_ITERATIONS ?= 25
VERIFY_DETERMINISM_MAX_ABS_DIFF ?= 2e-3
VERIFY_DETERMINISM_MIN_COSINE_SIMILARITY ?= 0.9995
VERIFY_DETERMINISM_TIMEOUT_SECONDS ?= 10
VERIFY_DETERMINISM_ARGS ?=
VERIFY_BATCHING_EMBED_URL ?= http://127.0.0.1:8000/embed
VERIFY_BATCHING_METRICS_URL ?= http://127.0.0.1:8000/metrics
VERIFY_BATCHING_HARDWARE_ID ?= local-dev
VERIFY_BATCHING_MODEL_ID ?= sentence-transformers/all-MiniLM-L6-v2
VERIFY_BATCHING_MODEL_REVISION ?= 826711e54e001c83835913827a843d8dd0a1def9
VERIFY_BATCHING_CONCURRENCY ?= 32
VERIFY_BATCHING_TOTAL_REQUESTS ?= 256
VERIFY_BATCHING_WARMUP_REQUESTS ?= 32
VERIFY_BATCHING_INPUTS_PER_REQUEST ?= 2
VERIFY_BATCHING_INPUT_TOKEN_COUNT ?= 24
VERIFY_BATCHING_TIMEOUT_SECONDS ?= 10
VERIFY_BATCHING_MAX_BATCH_SIZE ?= 128
VERIFY_BATCHING_MAX_BATCH_TOKENS ?= 8192
VERIFY_BATCHING_BATCH_TIMEOUT_MS ?= 2
VERIFY_BATCHING_MAX_BATCH_QUEUE_SIZE ?= 1024
VERIFY_BATCHING_BATCH_REQUEST_TIMEOUT_MS ?= 5000
VERIFY_BATCHING_ARGS ?=
BENCH_10K_EMBED_URL ?= http://127.0.0.1:8000/embed
BENCH_10K_HARDWARE_ID ?= local-dev
BENCH_10K_LABEL ?= batched
BENCH_10K_CONCURRENCY ?= 32
BENCH_10K_TOTAL_TEXTS ?= 10000
BENCH_10K_WARMUP_REQUESTS ?= 200
BENCH_10K_INPUTS_PER_REQUEST ?= 1
BENCH_10K_INPUT_TOKEN_COUNT ?= 24
BENCH_10K_TIMEOUT_SECONDS ?= 10
BENCH_10K_MODEL_ID ?= sentence-transformers/all-MiniLM-L6-v2
BENCH_10K_MODEL_REVISION ?= 826711e54e001c83835913827a843d8dd0a1def9
BENCH_10K_DEVICE ?= cpu
BENCH_10K_DTYPE ?= float32
BENCH_10K_MAX_LENGTH ?= 512
BENCH_10K_TRUNCATE ?= true
BENCH_10K_MAX_BATCH_SIZE ?= 128
BENCH_10K_MAX_BATCH_TOKENS ?= 8192
BENCH_10K_BATCH_TIMEOUT_MS ?= 2
BENCH_10K_MAX_BATCH_QUEUE_SIZE ?= 1024
BENCH_10K_BATCH_REQUEST_TIMEOUT_MS ?= 5000
BENCH_10K_WARM_CACHE ?= true
BENCH_10K_WARM_MODEL ?= true
BENCH_10K_WARM_CONTAINER ?= true
BENCH_10K_ARGS ?=

help:
	@echo "Available targets:"
	@echo "  bootstrap-dev     Create local venv, install deps, install pre-commit hooks"
	@echo "  worktree-create   Create a sibling git worktree on a new branch"
	@echo "  worktree-remove   Remove a git worktree safely"
	@echo "  start             Start API server (loads .env if present)"
	@echo "  format            Format Python code and apply safe lint fixes"
	@echo "  lint              Run Ruff lint checks"
	@echo "  typecheck         Run mypy type checks"
	@echo "  test              Run pytest (with coverage from pyproject.toml)"
	@echo "  verify-determinism Run live determinism verification against /embed"
	@echo "  verify-batching   Run live batching acceptance verification"
	@echo "  bench-10k         Run live benchmark harness against /embed"
	@echo "  audit             Audit dev dependencies for known vulnerabilities"
	@echo "  hadolint          Lint Dockerfiles if present"
	@echo "  pre-commit-install Install git pre-commit hooks"
	@echo "  pre-commit-run    Run configured pre-commit hooks on all files"
	@echo "  check             Run lint + typecheck + test"

bootstrap-dev:
	$(PYTHON) -m venv venv
	$(VENV)/python -m pip install --upgrade pip
	$(VENV)/python -m pip install -r requirements.txt -r dev-requirements.txt
	$(MAKE) pre-commit-install

worktree-create:
	@WORKTREE_NAME="$(WORKTREE)" \
	WORKTREE_PATH="$(WORKTREE_PATH)" \
	BASE_REF="$(BASE)" \
	COPY_ENV="$(COPY_ENV)" \
	SETUP="$(SETUP)" \
	SETUP_COMMAND='$(MAKE) -C "$$WORKTREE_PATH" bootstrap-dev PYTHON="$(PYTHON)"' \
	bash scripts/worktree.sh create

worktree-remove:
	@WORKTREE_NAME="$(WORKTREE)" \
	WORKTREE_PATH="$(WORKTREE_PATH)" \
	FORCE="$(FORCE)" \
	DELETE_BRANCH="$(DELETE_BRANCH)" \
	bash scripts/worktree.sh remove

start:
	@set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	$(VENV)/uvicorn "$${UVICORN_APP:-app.main:app}" --host "$${UVICORN_HOST:-127.0.0.1}" --port "$${UVICORN_PORT:-8000}"

format:
	$(VENV)/ruff format .
	$(VENV)/ruff check --fix .

lint:
	$(VENV)/ruff check .

typecheck:
	$(VENV)/mypy .

test:
	$(VENV)/pytest

verify-determinism:
	$(VENV)/python scripts/verify_determinism.py \
		--url "$(VERIFY_DETERMINISM_URL)" \
		--iterations "$(VERIFY_DETERMINISM_ITERATIONS)" \
		--max-abs-diff "$(VERIFY_DETERMINISM_MAX_ABS_DIFF)" \
		--min-cosine-similarity "$(VERIFY_DETERMINISM_MIN_COSINE_SIMILARITY)" \
		--timeout-seconds "$(VERIFY_DETERMINISM_TIMEOUT_SECONDS)" \
		$(VERIFY_DETERMINISM_ARGS)

verify-batching:
	$(VENV)/python scripts/verify_batching.py \
		--embed-url "$(VERIFY_BATCHING_EMBED_URL)" \
		--metrics-url "$(VERIFY_BATCHING_METRICS_URL)" \
		--hardware-id "$(VERIFY_BATCHING_HARDWARE_ID)" \
		--model-id "$(VERIFY_BATCHING_MODEL_ID)" \
		--model-revision "$(VERIFY_BATCHING_MODEL_REVISION)" \
		--concurrency "$(VERIFY_BATCHING_CONCURRENCY)" \
		--total-requests "$(VERIFY_BATCHING_TOTAL_REQUESTS)" \
		--warmup-requests "$(VERIFY_BATCHING_WARMUP_REQUESTS)" \
		--inputs-per-request "$(VERIFY_BATCHING_INPUTS_PER_REQUEST)" \
		--input-token-count "$(VERIFY_BATCHING_INPUT_TOKEN_COUNT)" \
		--timeout-seconds "$(VERIFY_BATCHING_TIMEOUT_SECONDS)" \
		--max-batch-size "$(VERIFY_BATCHING_MAX_BATCH_SIZE)" \
		--max-batch-tokens "$(VERIFY_BATCHING_MAX_BATCH_TOKENS)" \
		--batch-timeout-ms "$(VERIFY_BATCHING_BATCH_TIMEOUT_MS)" \
		--max-batch-queue-size "$(VERIFY_BATCHING_MAX_BATCH_QUEUE_SIZE)" \
		--batch-request-timeout-ms "$(VERIFY_BATCHING_BATCH_REQUEST_TIMEOUT_MS)" \
		$(VERIFY_BATCHING_ARGS)

bench-10k:
	$(VENV)/python scripts/bench_10k.py \
		--embed-url "$(BENCH_10K_EMBED_URL)" \
		--hardware-id "$(BENCH_10K_HARDWARE_ID)" \
		--label "$(BENCH_10K_LABEL)" \
		--concurrency "$(BENCH_10K_CONCURRENCY)" \
		--total-texts "$(BENCH_10K_TOTAL_TEXTS)" \
		--warmup-requests "$(BENCH_10K_WARMUP_REQUESTS)" \
		--inputs-per-request "$(BENCH_10K_INPUTS_PER_REQUEST)" \
		--input-token-count "$(BENCH_10K_INPUT_TOKEN_COUNT)" \
		--timeout-seconds "$(BENCH_10K_TIMEOUT_SECONDS)" \
		--model-id "$(BENCH_10K_MODEL_ID)" \
		--model-revision "$(BENCH_10K_MODEL_REVISION)" \
		--device "$(BENCH_10K_DEVICE)" \
		--dtype "$(BENCH_10K_DTYPE)" \
		--max-length "$(BENCH_10K_MAX_LENGTH)" \
		--truncate "$(BENCH_10K_TRUNCATE)" \
		--max-batch-size "$(BENCH_10K_MAX_BATCH_SIZE)" \
		--max-batch-tokens "$(BENCH_10K_MAX_BATCH_TOKENS)" \
		--batch-timeout-ms "$(BENCH_10K_BATCH_TIMEOUT_MS)" \
		--max-batch-queue-size "$(BENCH_10K_MAX_BATCH_QUEUE_SIZE)" \
		--batch-request-timeout-ms "$(BENCH_10K_BATCH_REQUEST_TIMEOUT_MS)" \
		--warm-cache "$(BENCH_10K_WARM_CACHE)" \
		--warm-model "$(BENCH_10K_WARM_MODEL)" \
		--warm-container "$(BENCH_10K_WARM_CONTAINER)" \
		$(BENCH_10K_ARGS)

audit:
	$(VENV)/pip-audit -r dev-requirements.txt

hadolint:
	@if command -v hadolint >/dev/null 2>&1; then \
		find . -type f \( -name 'Dockerfile' -o -name '*.Dockerfile' \) -print0 | xargs -0 -r hadolint ; \
	else \
		echo "hadolint not found on PATH"; \
		exit 1; \
	fi

pre-commit-install:
	PRE_COMMIT_HOME=$(PRE_COMMIT_HOME) $(VENV)/pre-commit install

pre-commit-run:
	PRE_COMMIT_HOME=$(PRE_COMMIT_HOME) $(VENV)/pre-commit run --all-files

check: lint typecheck test
