# Lightning AI Inference Sizer (prototype)

Internal MLSE sizing tool: explains model memory fit, workload demand,
GPUs per replica, serving replicas, spare capacity, fleet size, and the
evidence supporting latency and capacity claims.

Status: **Estimated-mode prototype.** No GPU benchmarks have been run;
calibration is pending (see `docs/benchmark-collection.md`). Branding uses
provisional assets scraped from lightning.ai pending the official brand kit —
never represent them as approved Lightning branding.

## Architecture

- `server.py` — FastAPI app (`/api/health`, `/api/catalog`, `/api/size`) that
  also serves the built frontend from `web/dist` under one origin. CORS is
  restricted to `CORS_ORIGINS` (comma-separated env var; empty = same-origin).
- `sizer/` — calculation package shared by the API and CLI.
  - `schemas.py` versioned request/result models (`schema_version: 1`)
  - `normalize.py` model normalization + provenance (preset / config.json / manual / inferred)
  - `hardware.py` + `data/hardware.json` seven-profile catalog with source metadata and independent statuses
  - `engine.py` pure sizing logic (validation, per-device memory, KV, latency bounds, fleet)
  - `benchmarks.py` `benchmark_profile/v1` adapter + compatibility matching
- `web/` — Vite/JS frontend, no framework. Brand tokens centralized in `web/src/theme/`.
- `tests/` — regression + integration checks (51).
- `docs/` — baseline failure record, benchmark collection instructions.
- `fixtures/benchmarks/` — synthetic profiles (marked synthetic; never calibration evidence).
- `samples/` — ready-to-run request scenarios.

## Run (dev)

```bash
python3 -m venv .venv && .venv/bin/pip install -r inference-sizer/requirements.txt
cd inference-sizer/web && npm ci && npm run build && cd ../..
cd inference-sizer && ../.venv/bin/python -m uvicorn server:app --reload --port 8000
```

Open http://127.0.0.1:8000 — API and frontend on one origin.

## Run (container)

```bash
docker build -t inference-sizer . && docker run -p 8000:8000 inference-sizer
```

## CLI (same calculation path as the API)

```bash
cd inference-sizer && ../.venv/bin/python -m sizer.cli ../samples/70b_chat_hgx_b200.json
../.venv/bin/python -m sizer.cli ../samples/70b_chat_hgx_b200.json --export scenario.json
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

## Sample scenarios

| File | What it demonstrates |
| --- | --- |
| `samples/70b_chat_hgx_b200.json` | 70B GQA interactive chat on HGX B200 |
| `samples/coding_assistant_27b_dgx_h100.json` | 27-31B coding assistant; active-user cadence + calls-per-task conversion |
| `samples/rag_long_context_hgx_b300.json` | Long-context RAG; multi-GPU replica; allocation size distinct from GPUs/replica |
| `samples/smoke_8b_synthetic_fixture.json` | 8B smoke test with a synthetic benchmark: adapter accepts it but it stays Estimated (synthetic fixtures are never calibration evidence) |

## Sizing behavior (summary)

- KV bytes/token (MHA/GQA) = 2 x layers x kv_heads x head_dim x KV element
  bytes, over the FULL retained input plus generated context. Prefix hit rate
  gives zero physical memory-sharing credit and only reduces prefill compute.
- input p95 + output p95 is a conservative tail envelope, not the p95 of total
  length; hard context limits apply for admission.
- Per-device memory fit is validated (weights sharded by TP; KV sharded or
  replicated). Runtime, CUDA graphs, activations and reserve are separate terms.
- Candidates failing their own optimistic TTFT/decode bound are rejected with
  the failing SLO named; a passing bound never proves p95 SLO compliance.
- Serving replicas respect both request-rate and concurrent demand; the
  operating factor applies once; spares are added separately and excluded from
  serving capacity; fleet = GPUs/replica x (serving + spare).
- Evidence (`Estimated` / `Benchmark calibrated`) is independent of
  feasibility (`Feasible` / `Infeasible` / `Unsupported`). Unsupported
  architectures (MLA, MoE) never receive recommendations.

See `docs/baseline-failures.md` for the sizing-engine regressions this rewrite
fixes, and `docs/benchmark-collection.md` for turning an Estimated result into
a Benchmark-calibrated one.

## Known limitations

- MLA/MoE recommendations, multimodal/hybrid architectures, KV offload,
  session-retention simulation, validated disaggregated serving, automated GPU
  provisioning, and commercial pricing are deferred.
- Rubin (Vera Rubin NVL72) and H200 NVL are comparison-only until availability,
  topology and compute values are verified; GB300 compute values are unsourced.
- The `vera_rubin_nvl72` / `h200_nvl` profiles have `compute_tflops: null`;
  latency bounds are UNVERIFIED (not failed) for those profiles.
- All latency and throughput numbers without an imported benchmark are
  optimistic ceiling estimates (linear bandwidth scaling, no queueing).