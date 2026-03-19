FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/var/cache/huggingface \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

ARG APT_CA_CERTIFICATES_VERSION=2024*
ARG APT_CURL_VERSION=7.81.*
ARG APT_PYTHON3_VERSION=3.10.*
ARG APT_PYTHON3_PIP_VERSION=22.0.*
ARG APT_PYTHON3_VENV_VERSION=3.10.*

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates=${APT_CA_CERTIFICATES_VERSION} \
        curl=${APT_CURL_VERSION} \
        python3=${APT_PYTHON3_VERSION} \
        python3-pip=${APT_PYTHON3_PIP_VERSION} \
        python3-venv=${APT_PYTHON3_VENV_VERSION} \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY docker/requirements.cuda-linux.txt /app/docker/requirements.cuda-linux.txt

RUN pip install \
        --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple \
        --requirement /app/docker/requirements.cuda-linux.txt

COPY app /app/app

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser \
    && mkdir -p "${HF_HOME}" \
    && chown -R appuser:appuser /app "${HF_HOME}"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=12 \
    CMD curl --fail --silent http://127.0.0.1:8000/readyz >/dev/null || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
