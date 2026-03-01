.PHONY: help format lint typecheck test audit hadolint pre-commit-install pre-commit-run check

VENV := ./venv/bin
PRE_COMMIT_HOME := /tmp/pre-commit-cache

help:
	@echo "Available targets:"
	@echo "  format            Format Python code and apply safe lint fixes"
	@echo "  lint              Run Ruff lint checks"
	@echo "  typecheck         Run mypy type checks"
	@echo "  test              Run pytest (with coverage from pyproject.toml)"
	@echo "  audit             Audit dev dependencies for known vulnerabilities"
	@echo "  hadolint          Lint Dockerfiles if present"
	@echo "  pre-commit-install Install git pre-commit hooks"
	@echo "  pre-commit-run    Run configured pre-commit hooks on all files"
	@echo "  check             Run lint + typecheck + test"

format:
	$(VENV)/ruff format .
	$(VENV)/ruff check --fix .

lint:
	$(VENV)/ruff check .

typecheck:
	$(VENV)/mypy .

test:
	$(VENV)/pytest

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
