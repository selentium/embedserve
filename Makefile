.PHONY: help start format lint typecheck test verify-determinism audit hadolint pre-commit-install pre-commit-run check

VENV := ./venv/bin
PRE_COMMIT_HOME := /tmp/pre-commit-cache
VERIFY_DETERMINISM_URL ?= http://127.0.0.1:8000/embed
VERIFY_DETERMINISM_ITERATIONS ?= 25
VERIFY_DETERMINISM_MAX_ABS_DIFF ?= 2e-3
VERIFY_DETERMINISM_MIN_COSINE_SIMILARITY ?= 0.9995
VERIFY_DETERMINISM_TIMEOUT_SECONDS ?= 10
VERIFY_DETERMINISM_ARGS ?=

help:
	@echo "Available targets:"
	@echo "  start             Start API server (loads .env if present)"
	@echo "  format            Format Python code and apply safe lint fixes"
	@echo "  lint              Run Ruff lint checks"
	@echo "  typecheck         Run mypy type checks"
	@echo "  test              Run pytest (with coverage from pyproject.toml)"
	@echo "  verify-determinism Run live determinism verification against /embed"
	@echo "  audit             Audit dev dependencies for known vulnerabilities"
	@echo "  hadolint          Lint Dockerfiles if present"
	@echo "  pre-commit-install Install git pre-commit hooks"
	@echo "  pre-commit-run    Run configured pre-commit hooks on all files"
	@echo "  check             Run lint + typecheck + test"

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
