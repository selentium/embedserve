# EmbedServe

Production-oriented embedding inference service designed for high-throughput and efficient GPU utilization.

Built with FastAPI, embedserve uses dynamic batching and asynchronous request handling to maximize performance while maintaining predictable latency. The service includes readiness/liveness probes, Prometheus metrics, and reproducible model loading for reliable operation in production environments.

### Key features

- Dynamic batching for improved throughput under load
- Deterministic embedding outputs (stable across runs)
- GPU-optimized inference pipeline
- Health checks (`/healthz`, `/readyz`) and Prometheus metrics
- Docker-based deployment for NVIDIA GPU environments

## What It Guarantees

- `POST /embed` returns real transformer embeddings for one or more input strings.
- `GET /healthz` is liveness only and returns `200` while the process is up.
- `GET /readyz` returns `200` only after model initialization and warmup succeed.
- Dynamic batching is always on for successful runtime operation and preserves FIFO request submission order.
- Configuration is environment-driven, with a pinned default model revision and explicit batching controls.
- Determinism is best-effort numerical stability under a fixed environment, not bitwise reproducibility.
- Docker deployment targets a local Linux host with an NVIDIA GPU and persists the Hugging Face cache across restarts.
- Prometheus metrics expose HTTP, readiness, batching, failure, and CUDA memory telemetry.

## Quickstart

Host prerequisites:

- Docker Engine with Docker Compose v2
- NVIDIA GPU with a driver compatible with CUDA 12.6
- NVIDIA Container Toolkit configured for Docker

Bring the service up:

```bash
make docker-build
make docker-up
make docker-health
```

Send a request:

```bash
curl http://127.0.0.1:8000/embed \
  -H 'content-type: application/json' \
  -d '{"inputs":["hello world","embed this"]}'
```

Stop the stack:

```bash
make docker-down
```

## Documentation

Start here for full docs:

- [Documentation Home](docs/README.md)

User and operator docs:

- [Overview](docs/use/overview.md)
- [Quickstart](docs/use/quickstart.md)
- [API Reference](docs/use/api.md)
- [Configuration Reference](docs/use/configuration.md)
- [Deployment](docs/use/deployment.md)
- [Operations and Monitoring](docs/use/operations.md)
- [Determinism and Compatibility](docs/use/determinism.md)
- [Performance and Stability](docs/use/performance.md)
- [Troubleshooting](docs/use/troubleshooting.md)

Contributor docs:

- [Contributor Guide](docs/contribute/README.md)
- [Local Setup](docs/contribute/getting-started.md)
- [Architecture](docs/contribute/architecture.md)
- [Testing and Verification](docs/contribute/testing.md)
- [Dependency Management](docs/contribute/dependencies.md)
- [Worktrees](docs/contribute/worktrees.md)
- [Documentation Maintenance](docs/contribute/documentation.md)

Reference artifacts:

- [Benchmark Report](docs/benchmarks/BENCHMARK.md)
- [Stability Report](docs/benchmarks/STABILITY.md)

## Contributing

For a local developer setup:

```bash
make bootstrap-dev
make start
```

The contributor docs cover the repo layout, testing commands, dependency lockfile workflow, and worktree helpers in more detail.
