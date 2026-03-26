.PHONY: help bootstrap-dev deps-compile deps-upgrade worktree-create worktree-remove start format lint typecheck test verify-determinism verify-batching bench-10k load-test audit hadolint pre-commit-install pre-commit-run docker-build docker-up docker-down docker-logs docker-ps docker-health docker-test-request check

VENV := ./venv/bin
PRE_COMMIT_HOME := /tmp/pre-commit-cache
PYTHON ?= python3
TORCH_VARIANT ?= auto
PIP_COMPILE := $(VENV)/python -m piptools compile
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
DOCKER_COMPOSE_FILE ?= docker/docker-compose.yml
DOCKER_COMPOSE := docker compose -f $(DOCKER_COMPOSE_FILE)
DOCKER_SERVICE ?= embedserve
EMBEDSERVE_PORT ?= 8000
DOCKER_INTERNAL_HEALTH_URL ?= http://127.0.0.1:8000/readyz
DOCKER_HEALTH_TIMEOUT_SECONDS ?= 300
DOCKER_HEALTH_POLL_INTERVAL_SECONDS ?= 5
DOCKER_INTERNAL_EMBED_URL ?= http://127.0.0.1:8000/embed
DOCKER_TEST_INPUTS_JSON ?= ["hello world","docker smoke test"]
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
LOAD_TEST_EMBED_URL ?= http://127.0.0.1:8000/embed
LOAD_TEST_METRICS_URL ?= http://127.0.0.1:8000/metrics
LOAD_TEST_READY_URL ?= http://127.0.0.1:8000/readyz
LOAD_TEST_DURATION_SECONDS ?= 3600
LOAD_TEST_CONCURRENCY ?= 32
LOAD_TEST_WARMUP_REQUESTS ?= 64
LOAD_TEST_INPUTS_PER_REQUEST ?= 2
LOAD_TEST_INPUT_TOKEN_COUNT ?= 24
LOAD_TEST_TIMEOUT_SECONDS ?= 10
LOAD_TEST_METRICS_POLL_INTERVAL_SECONDS ?= 5
LOAD_TEST_MAX_VRAM_DRIFT_BYTES ?= 268435456
LOAD_TEST_HARDWARE_ID ?= local-dev
LOAD_TEST_MODEL_ID ?= sentence-transformers/all-MiniLM-L6-v2
LOAD_TEST_MODEL_REVISION ?= 826711e54e001c83835913827a843d8dd0a1def9
LOAD_TEST_MAX_BATCH_SIZE ?= 128
LOAD_TEST_MAX_BATCH_TOKENS ?= 8192
LOAD_TEST_BATCH_TIMEOUT_MS ?= 2
LOAD_TEST_MAX_BATCH_QUEUE_SIZE ?= 1024
LOAD_TEST_BATCH_REQUEST_TIMEOUT_MS ?= 5000
LOAD_TEST_ARGS ?=

help:
	@echo "Available targets:"
	@echo "  bootstrap-dev     Create local venv, install deps, select Linux Torch overlay, install pre-commit hooks"
	@echo "  deps-compile      Regenerate pinned requirements from *.in sources"
	@echo "  deps-upgrade      Refresh pinned requirements to latest compatible versions"
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
	@echo "  load-test         Run sustained stability and memory monitoring load test"
	@echo "  audit             Audit runtime and dev dependency lockfiles for known vulnerabilities"
	@echo "  hadolint          Lint Dockerfiles if present"
	@echo "  pre-commit-install Install git pre-commit hooks"
	@echo "  pre-commit-run    Run configured pre-commit hooks on all files"
	@echo "  docker-build      Build the Docker image via docker compose"
	@echo "  docker-up         Start the Docker service in the background"
	@echo "  docker-down       Stop the Docker compose stack"
	@echo "  docker-logs       Tail Docker service logs"
	@echo "  docker-ps         Show Docker compose service status"
	@echo "  docker-health     Poll /readyz until the Docker service is ready"
	@echo "  docker-test-request Send a sample POST /embed request to Docker"
	@echo "  check             Run lint + typecheck + test"

bootstrap-dev:
	$(PYTHON) -m venv venv
	$(VENV)/python -m pip install --upgrade pip
	$(VENV)/python -m pip install -r requirements.txt -r dev-requirements.txt
	@if [ "$$(uname -s)" = "Linux" ]; then \
		case "$(TORCH_VARIANT)" in \
			auto) \
				if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then \
					torch_requirements=requirements.cuda-linux.txt; \
				else \
					torch_requirements=requirements.cpu-linux.txt; \
				fi \
				;; \
			cpu) torch_requirements=requirements.cpu-linux.txt ;; \
			cuda) torch_requirements=requirements.cuda-linux.txt ;; \
			*) \
				echo "Unsupported TORCH_VARIANT=$(TORCH_VARIANT); expected auto, cpu, or cuda" >&2; \
				exit 1 \
				;; \
		esac; \
		echo "Installing Linux Torch overlay: $$torch_requirements"; \
		$(VENV)/python -m pip install -r "$$torch_requirements"; \
	fi
	$(MAKE) pre-commit-install

deps-compile:
	@set -e; \
	tmp_requirements="$$(mktemp)"; \
	trap 'rm -f "$$tmp_requirements"' EXIT; \
	$(PIP_COMPILE) --output-file "$$tmp_requirements" requirements.in; \
	$(VENV)/python scripts/filter_portable_requirements.py "$$tmp_requirements" requirements.txt
	$(PIP_COMPILE) --output-file dev-requirements.txt dev-requirements.in
	@set -e; \
	tmp_requirements="$$(mktemp)"; \
	trap 'rm -f "$$tmp_requirements"' EXIT; \
	$(PIP_COMPILE) --output-file "$$tmp_requirements" requirements.cpu-linux.in; \
	$(VENV)/python scripts/filter_torch_overlay_requirements.py "$$tmp_requirements" requirements.cpu-linux.txt requirements.txt
	@set -e; \
	tmp_requirements="$$(mktemp)"; \
	trap 'rm -f "$$tmp_requirements"' EXIT; \
	$(PIP_COMPILE) --output-file "$$tmp_requirements" requirements.cuda-linux.in; \
	$(VENV)/python scripts/filter_cuda_overlay_requirements.py "$$tmp_requirements" requirements.cuda-linux.txt requirements.txt

deps-upgrade:
	@set -e; \
	tmp_requirements="$$(mktemp)"; \
	trap 'rm -f "$$tmp_requirements"' EXIT; \
	$(PIP_COMPILE) --upgrade --output-file "$$tmp_requirements" requirements.in; \
	$(VENV)/python scripts/filter_portable_requirements.py "$$tmp_requirements" requirements.txt
	$(PIP_COMPILE) --upgrade --output-file dev-requirements.txt dev-requirements.in
	@set -e; \
	tmp_requirements="$$(mktemp)"; \
	trap 'rm -f "$$tmp_requirements"' EXIT; \
	$(PIP_COMPILE) --upgrade --output-file "$$tmp_requirements" requirements.cpu-linux.in; \
	$(VENV)/python scripts/filter_torch_overlay_requirements.py "$$tmp_requirements" requirements.cpu-linux.txt requirements.txt
	@set -e; \
	tmp_requirements="$$(mktemp)"; \
	trap 'rm -f "$$tmp_requirements"' EXIT; \
	$(PIP_COMPILE) --upgrade --output-file "$$tmp_requirements" requirements.cuda-linux.in; \
	$(VENV)/python scripts/filter_cuda_overlay_requirements.py "$$tmp_requirements" requirements.cuda-linux.txt requirements.txt

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

load-test:
	$(VENV)/python scripts/load_test.py \
		--embed-url "$(LOAD_TEST_EMBED_URL)" \
		--metrics-url "$(LOAD_TEST_METRICS_URL)" \
		--ready-url "$(LOAD_TEST_READY_URL)" \
		--duration-seconds "$(LOAD_TEST_DURATION_SECONDS)" \
		--concurrency "$(LOAD_TEST_CONCURRENCY)" \
		--warmup-requests "$(LOAD_TEST_WARMUP_REQUESTS)" \
		--inputs-per-request "$(LOAD_TEST_INPUTS_PER_REQUEST)" \
		--input-token-count "$(LOAD_TEST_INPUT_TOKEN_COUNT)" \
		--timeout-seconds "$(LOAD_TEST_TIMEOUT_SECONDS)" \
		--metrics-poll-interval-seconds "$(LOAD_TEST_METRICS_POLL_INTERVAL_SECONDS)" \
		--max-vram-drift-bytes "$(LOAD_TEST_MAX_VRAM_DRIFT_BYTES)" \
		--hardware-id "$(LOAD_TEST_HARDWARE_ID)" \
		--model-id "$(LOAD_TEST_MODEL_ID)" \
		--model-revision "$(LOAD_TEST_MODEL_REVISION)" \
		--max-batch-size "$(LOAD_TEST_MAX_BATCH_SIZE)" \
		--max-batch-tokens "$(LOAD_TEST_MAX_BATCH_TOKENS)" \
		--batch-timeout-ms "$(LOAD_TEST_BATCH_TIMEOUT_MS)" \
		--max-batch-queue-size "$(LOAD_TEST_MAX_BATCH_QUEUE_SIZE)" \
		--batch-request-timeout-ms "$(LOAD_TEST_BATCH_REQUEST_TIMEOUT_MS)" \
		$(LOAD_TEST_ARGS)

audit:
	$(VENV)/pip-audit -r requirements.txt
	$(VENV)/pip-audit -r requirements.cpu-linux.txt
	$(VENV)/pip-audit -r requirements.cuda-linux.txt
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

docker-build:
	@$(DOCKER_COMPOSE) build $(DOCKER_SERVICE)

docker-up:
	@$(DOCKER_COMPOSE) up --detach $(DOCKER_SERVICE)

docker-down:
	@$(DOCKER_COMPOSE) down

docker-logs:
	@$(DOCKER_COMPOSE) logs --tail=100 --follow $(DOCKER_SERVICE)

docker-ps:
	@$(DOCKER_COMPOSE) ps

docker-health:
	@container_id="$$( $(DOCKER_COMPOSE) ps -q $(DOCKER_SERVICE) )"; \
	if [ -z "$$container_id" ]; then \
		echo "No container found for service $(DOCKER_SERVICE)" >&2; \
		exit 1; \
	fi; \
	if [ "$$(docker inspect --format '{{.State.Running}}' "$$container_id")" != "true" ]; then \
		echo "Service $(DOCKER_SERVICE) is not running" >&2; \
		exit 1; \
	fi; \
	start_time="$$(date +%s)"; \
	while :; do \
		health_status="$$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$$container_id")"; \
		if [ "$$health_status" = "healthy" ]; then \
			$(DOCKER_COMPOSE) exec -T $(DOCKER_SERVICE) curl --fail --silent "$(DOCKER_INTERNAL_HEALTH_URL)"; \
			exit 0; \
		fi; \
		if [ "$$health_status" = "unhealthy" ]; then \
			echo "Service $(DOCKER_SERVICE) reported unhealthy" >&2; \
			exit 1; \
		fi; \
		if [ "$$(( $$(date +%s) - start_time ))" -ge "$(DOCKER_HEALTH_TIMEOUT_SECONDS)" ]; then \
			echo "Timed out waiting for service $(DOCKER_SERVICE) to become healthy" >&2; \
			exit 1; \
		fi; \
		sleep "$(DOCKER_HEALTH_POLL_INTERVAL_SECONDS)"; \
	done

docker-test-request:
	@$(DOCKER_COMPOSE) exec -T \
		-e DOCKER_INTERNAL_EMBED_URL="$(DOCKER_INTERNAL_EMBED_URL)" \
		-e DOCKER_TEST_INPUTS_JSON='$(DOCKER_TEST_INPUTS_JSON)' \
		$(DOCKER_SERVICE) python -c 'exec("""import json\nimport os\nfrom urllib import request\n\npayload = json.dumps({\"inputs\": json.loads(os.environ[\"DOCKER_TEST_INPUTS_JSON\"])}).encode(\"utf-8\")\nreq = request.Request(\n    os.environ[\"DOCKER_INTERNAL_EMBED_URL\"],\n    data=payload,\n    headers={\"Content-Type\": \"application/json\"},\n    method=\"POST\",\n)\nwith request.urlopen(req, timeout=10) as response:\n    print(response.read().decode(\"utf-8\"))\n""")'

check: lint typecheck test
