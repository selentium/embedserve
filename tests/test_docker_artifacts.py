from __future__ import annotations

import re
from pathlib import Path

import yaml  # type: ignore[import-untyped,unused-ignore]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_requirements_files_are_exact_pins() -> None:
    requirement_pattern = re.compile(
        r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_.,-]+\])?==[A-Za-z0-9_.+-]+$"
    )

    for relative_path in ("requirements.txt", "dev-requirements.txt"):
        lines = _read_lines(REPO_ROOT / relative_path)
        assert lines, f"{relative_path} should not be empty"
        assert all(requirement_pattern.fullmatch(line) for line in lines), relative_path


def test_dependency_input_files_capture_human_edited_direct_dependencies() -> None:
    runtime_lines = _read_lines(REPO_ROOT / "requirements.in")
    dev_lines = _read_lines(REPO_ROOT / "dev-requirements.in")
    docker_lines = _read_lines(REPO_ROOT / "docker" / "requirements.cuda-linux.in")

    assert "fastapi" in runtime_lines
    assert "pydantic" in runtime_lines
    assert "pydantic-settings" in runtime_lines
    assert "torch" in runtime_lines
    assert "transformers" in runtime_lines
    assert "uvicorn" in runtime_lines

    assert "-c requirements.txt" in dev_lines
    assert "pip-tools" in dev_lines
    assert "pytest" in dev_lines
    assert "PyYAML" in dev_lines

    assert "--index-url https://download.pytorch.org/whl/cu126" in docker_lines
    assert "--extra-index-url https://pypi.org/simple" in docker_lines
    assert "-r ../requirements.txt" in docker_lines
    assert "torch==2.9.1+cu126" in docker_lines


def test_generated_requirements_files_include_pip_compile_header() -> None:
    for relative_path in (
        "requirements.txt",
        "dev-requirements.txt",
        "docker/requirements.cuda-linux.txt",
    ):
        contents = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "pip-compile" in contents, relative_path


def test_docker_requirements_extend_base_runtime_with_linux_cuda_pins() -> None:
    requirement_pattern = re.compile(
        r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_.,-]+\])?==[A-Za-z0-9_.+-]+$"
    )

    lines = _read_lines(REPO_ROOT / "docker" / "requirements.cuda-linux.txt")

    assert lines[0] == "--index-url https://download.pytorch.org/whl/cu126"
    assert lines[1] == "--extra-index-url https://pypi.org/simple"
    assert lines[2] == "-r ../requirements.txt"
    assert all(requirement_pattern.fullmatch(line) for line in lines[3:])
    assert "torch==2.9.1+cu126" in lines
    assert "nvidia-cudnn-cu12==9.10.2.21" in lines
    assert "triton==3.5.1" in lines


def test_portable_runtime_requirements_exclude_linux_cuda_only_packages() -> None:
    lines = _read_lines(REPO_ROOT / "requirements.txt")

    for requirement in (
        "torch==2.9.1+cu126",
        "nvidia-cublas-cu12==12.6.4.1",
        "nvidia-cudnn-cu12==9.10.2.21",
        "triton==3.5.1",
    ):
        assert requirement not in lines


def test_portable_requirements_filter_script_excludes_cuda_runtime_dependencies() -> None:
    filter_script = (REPO_ROOT / "scripts" / "filter_portable_requirements.py").read_text(
        encoding="utf-8"
    )

    assert '_EXCLUDED_PREFIXES = ("nvidia-",)' in filter_script
    assert '_EXCLUDED_NAMES = {"triton"}' in filter_script


def test_cuda_overlay_filter_script_keeps_only_cuda_runtime_dependencies() -> None:
    filter_script = (REPO_ROOT / "scripts" / "filter_cuda_overlay_requirements.py").read_text(
        encoding="utf-8"
    )

    assert '_KEEP_PREFIXES = ("nvidia-",)' in filter_script
    assert '_KEEP_NAMES = {"torch", "triton"}' in filter_script
    assert "-r ../requirements.txt\\n" in filter_script


def test_pyproject_targets_python_310() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'target-version = "py310"' in pyproject
    assert 'python_version = "3.10"' in pyproject


def test_dockerfile_uses_cuda_runtime_and_readiness_healthcheck() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04" in dockerfile
    assert "python3-venv" in dockerfile
    assert "HF_HOME=/var/cache/huggingface" in dockerfile
    assert (
        "COPY docker/requirements.cuda-linux.txt /app/docker/requirements.cuda-linux.txt"
        in dockerfile
    )
    assert "--index-url https://download.pytorch.org/whl/cu126" in dockerfile
    assert "pip install \\" in dockerfile
    assert "--requirement /app/docker/requirements.cuda-linux.txt" in dockerfile
    assert "HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=12" in dockerfile
    assert "http://127.0.0.1:8000/readyz" in dockerfile
    assert "USER appuser" in dockerfile
    assert 'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]' in dockerfile


def test_compose_service_requests_gpu_and_cache_volume() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker" / "docker-compose.yml").read_text("utf-8"))

    service = compose["services"]["embedserve"]

    assert service["build"] == {"context": "..", "dockerfile": "docker/Dockerfile"}
    assert service["gpus"] == "all"
    assert service["ports"] == ["${EMBEDSERVE_PORT:-8000}:8000"]
    assert service["volumes"] == ["hf-cache:/var/cache/huggingface"]
    assert service["environment"]["DEVICE"] == "${DEVICE:-cuda}"
    assert service["environment"]["MODEL_REVISION"] == (
        "${MODEL_REVISION:-826711e54e001c83835913827a843d8dd0a1def9}"
    )
    assert compose["volumes"] == {"hf-cache": None}


def test_makefile_exposes_docker_wrapper_targets() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    for target in (
        "deps-compile",
        "deps-upgrade",
        "docker-build",
        "docker-up",
        "docker-down",
        "docker-logs",
        "docker-ps",
        "docker-health",
        "docker-test-request",
    ):
        assert re.search(rf"^{re.escape(target)}:", makefile, flags=re.MULTILINE), target
        assert target in makefile.split("help:", maxsplit=1)[1]

    assert "@$(DOCKER_COMPOSE) build $(DOCKER_SERVICE)" in makefile
    assert "@$(DOCKER_COMPOSE) up --detach $(DOCKER_SERVICE)" in makefile
    assert (
        'scripts/filter_portable_requirements.py "$$tmp_requirements" requirements.txt' in makefile
    )
    assert (
        'scripts/filter_cuda_overlay_requirements.py "$$tmp_requirements" '
        "docker/requirements.cuda-linux.txt"
    ) in makefile
    assert "$(PIP_COMPILE) --output-file dev-requirements.txt dev-requirements.in" in makefile
    assert (
        "$(PIP_COMPILE) --upgrade --output-file dev-requirements.txt dev-requirements.in"
        in makefile
    )
    assert "docker inspect --format" in makefile
    assert "@$(DOCKER_COMPOSE) exec -T \\" in makefile
    assert '-e DOCKER_INTERNAL_EMBED_URL="$(DOCKER_INTERNAL_EMBED_URL)"' in makefile
    assert "-e DOCKER_TEST_INPUTS_JSON='$(DOCKER_TEST_INPUTS_JSON)'" in makefile
    assert "$(DOCKER_SERVICE) python -c" in makefile


def test_dockerignore_and_env_example_capture_local_docker_contract() -> None:
    dockerignore = _read_lines(REPO_ROOT / ".dockerignore")
    env_example = _read_lines(REPO_ROOT / ".env.example")

    assert "venv/" in dockerignore
    assert ".env" in dockerignore
    assert "tests/" in dockerignore
    assert "DEVICE=cpu" in env_example
    assert "EMBEDSERVE_PORT=8000" in env_example
