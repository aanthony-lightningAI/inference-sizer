# Lightning AI Inference Sizer (prototype)

Internal MLSE sizing tool: explains model memory fit, workload demand,
GPUs per replica, serving replicas, spare capacity, fleet size, and the
evidence supporting latency and capacity claims.

Status: **Estimated-mode prototype.** No GPU benchmarks have been run;
calibration is pending (see `docs/benchmark-collection.md`). Branding uses
provisional assets scraped from lightning.ai pending the official brand kit.

## Layout

- `server.py` — FastAPI app (`/api/health`, `/api/catalog`, `/api/size`) that also
  serves the built frontend from `web/dist` under one origin.
- `sizer/` — calculation package shared by the API and CLI.
  - `schemas.py` versioned request/result models
  - `normalize.py` model normalization + provenance
  - `hardware.py` / `data/hardware.json` hardware catalog with source metadata
  - `engine.py` pure sizing logic
  - `benchmarks.py` versioned benchmark profile adapter
- `web/` — Vite/JS frontend (no framework).
- `tests/` — regression and integration checks (repo root).
- `docs/` — baseline failure record, benchmark collection instructions.
- `fixtures/benchmarks/` — synthetic benchmark profiles (marked synthetic; never
  real calibration evidence).

## Run (dev)

```bash
python3 -m venv .venv && .venv/bin/pip install -r inference-sizer/requirements.txt
cd inference-sizer/web && npm install && npm run build && cd ../..
cd inference-sizer && ../.venv/bin/python -m uvicorn server:app --reload --port 8000
```

Open http://127.0.0.1:8000 — the API and the built frontend share one origin.

## Run (container)

```bash
docker build -t inference-sizer . && docker run -p 8000:8000 inference-sizer
```

## CLI

```bash
cd inference-sizer && ../.venv/bin/python -m sizer.cli            # defaults
../.venv/bin/python -m sizer.cli request.json                     # from file
../.venv/bin/python -m sizer.cli --export scenario.json           # full export
```

## Sample requests

Health:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Catalog (versioned presets, hardware profiles with source metadata):

```bash
curl -s http://127.0.0.1:8000/api/catalog | python3 -m json.tool | head -50
```

Size (70B GQA on HGX B200, interactive chat):

```bash
curl -s http://127.0.0.1:8000/api/size -H 'content-type: application/json' -d '{
  "schema_version": 1,
  "customer": "Example Co",
  "model": {"preset_id": "llama_3_1_70b", "weight_format": "bf16", "kv_dtype": "bf16"},
  "workload": {"mean_input_tokens": 2048, "p95_input_tokens": 8192,
               "mean_output_tokens": 512, "p95_output_tokens": 2048,
               "peak_rps": 20, "peak_inflight": 160,
               "ttft_ms_p95": 800, "decode_ms_p95": 40},
  "deployment": {"hardware_profile_id": "hgx_b200",
                 "max_gpus_per_replica": 8, "spare_replicas": 1}
}' | python3 -m json.tool | head -60
```

See `docs/baseline-failures.md` for the sizing-engine regressions this rewrite
fixes, and `docs/benchmark-collection.md` for turning an Estimated result into
a Benchmark-calibrated one.