# Lightning AI Inference Sizer

Internal MLSE sizing tool: explains model memory fit, workload demand, GPUs
per replica, serving replicas, spare capacity, fleet size, and the evidence
supporting latency and capacity claims.

**Status: Estimated-mode prototype.** No GPU benchmarks have been run;
calibration is pending (see
[`docs/benchmark-collection.md`](docs/benchmark-collection.md)). Branding uses
provisional assets pending the official Lightning brand kit.

## Quick links

| What | Where |
| --- | --- |
| Tool README (architecture, run instructions, model support matrix) | [`inference-sizer/README.md`](inference-sizer/README.md) |
| FastAPI app + frontend sources | [`inference-sizer/`](inference-sizer/) |
| Verified model catalog notes (per-preset sources, derivations, not-found list) | [`docs/model-catalog-notes.md`](docs/model-catalog-notes.md) |
| Completion report | [`docs/completion-report.md`](docs/completion-report.md) |
| Container / Compose / Kubernetes | [`Dockerfile`](Dockerfile) · [`docker-compose.yml`](docker-compose.yml) · [`k8s/`](k8s/) |

## Run

```bash
docker compose up -d --build        # or: docker build -t inference-sizer . && docker run -p 8000:8000
open http://localhost:8000
```

Tests: `PYTHONPATH=inference-sizer .venv/bin/python -m pytest tests/ -q` (74).