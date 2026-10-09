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
  - `schemas.py` versioned request/result models (`schema_version: 2`; v1
    payloads are migrated on import)
  - `normalize.py` model normalization + provenance (preset / config.json / manual / inferred);
    classifies dense, MLA, MLA-hybrid and DeltaNet/sparse-hybrid architectures
  - `hardware.py` + `data/hardware.json` seven-profile hardware catalog and a
    24-preset model catalog with per-preset provenance (`spec_confidence`,
    `source_ref`, family)
  - `engine.py` pure sizing logic (validation, per-device memory, three KV-cache
    paths — MHA/GQA, MLA, vendor override — plus fixed per-sequence KV terms,
    latency bounds, fleet)
  - `benchmarks.py` `benchmark_profile/v1` adapter + compatibility matching
- `web/` — Vite/JS frontend, no framework. Brand tokens centralized in `web/src/theme/`.
- `tests/` — regression + integration checks (74).
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
# plain docker
docker build -t inference-sizer . && docker run -p 8000:8000 inference-sizer

# docker compose (same image, with healthcheck)
docker compose up -d --build
```

## Run (kubernetes)

Manifests live in `k8s/` (Deployment + ClusterIP Service; frontend and API are
served from one origin, so one container image suffices):

```bash
docker build -t inference-sizer:latest .   # push to your registry for a real cluster
kubectl apply -f k8s/
kubectl port-forward svc/inference-sizer 8000:80   # local access
```

The Deployment runs 2 replicas as a non-root user with readiness/liveness
probes on `/api/health`. `CORS_ORIGINS` is empty by default (same-origin only);
set it in the container env to allow other origins.

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

## Model support matrix

Three KV-cache paths, dispatched per model (`model.kv_model`):

- **mha_gqa** — `2 × layers-with-KV × kv_heads × head_dim × kv_bytes` (covers
  MHA, GQA and MQA). For hybrid DeltaNet/sparse models only the
  full-attention layers (`attention_kv_layers`) count toward the growing term;
  sliding-window caps and linear-attention state are a fixed per-sequence term.
- **mla** — `layers-with-KV × (kv_lora_rank + qk_rope_head_dim) × kv_bytes`,
  replicated across TP (matches vLLM/Mooncake MLA layouts).
- **override** — vendor- or config-derived bytes/token, cited per preset
  (`kv_bytes_per_token_override` + `kv_override_source`); used where the
  architecture (CSA/HCA, compressed sparse pools, unified K/V) is not modeled
  first-principles.

MoE: memory fit uses **total** params (all experts resident); prefill compute
uses **active** params; decode ITL is bounded with total-param reads
(conservative at serving batch). Expert parallelism is not modeled.

| Family | Presets | KV path | Spec confidence |
| --- | --- | --- | --- |
| Llama | 3.1 8B / 70B / 405B | mha_gqa | published |
| Qwen (2.5) | 32B | mha_gqa | published |
| Qwen (3.5) | 27B, 35B-A3B, 122B-A10B | hybrid GQA subset + fixed DeltaNet state | published |
| Qwen (3.6) | 27B | hybrid GQA subset + fixed DeltaNet state | published |
| Qwen (3.8) | 27B | hybrid GQA subset + fixed DeltaNet state | published |
| Qwen (3.8) | Flash-Next | override (sparse-pool, ratio 4) + fixed DeltaNet state | published |
| DeepSeek | V3 (MLA) | mla | published |
| DeepSeek | V4 Flash 0731 | override (CSA/HCA) | published |
| DeepSeek | V4.1 Flash | override (CED, derived 1/8-of-V4) | **unverified** |
| GLM | 5, 5.1, 5.2 | mla | 5.2 published; 5 / 5.1 **unverified** (family-inherited) |
| GLM | 5.3 Flash | mla (11 full layers) + fixed KDA state | published |
| Kimi | K2.6, K2.7-Code | mla | published |
| Kimi | K3 | mla (24 gated-MLA layers) + fixed KDA state | published |
| Inkling | Inkling, Inkling-Small | mha_gqa (global layers) + fixed sliding-window | published |
| Gemma (4) | 31B, 26B-A4B | override (unified K/V) + fixed sliding-window | published |

Per-model sources, derivations, and the requested combinations that were
verified **not to exist** (e.g. Qwen3.8-35B/122B/235B; Muse Spark 1.3 is
proprietary) are documented in [`docs/model-catalog-notes.md`](../docs/model-catalog-notes.md).

## Sizing behavior (summary)

- KV bytes/token follow one of the three paths above (MHA/GQA, MLA, override),
  over the FULL retained input plus generated context; hybrid models add a
  fixed per-sequence term. Prefix hit rate gives zero physical memory-sharing
  credit and only reduces prefill compute.
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
  architectures (MLA without `kv_lora_rank`, hybrids without their fixed-term
  constants) never receive recommendations — the failure names the missing field.

See `docs/baseline-failures.md` for the sizing-engine regressions this rewrite
fixes, and `docs/benchmark-collection.md` for turning an Estimated result into
a Benchmark-calibrated one.

## Known limitations

- MLA is modeled as the compressed single-stream cache (`kv_lora_rank +
  rope`); CSA/HCA, compressed sparse pools and unified-K/V attention are not
  modeled first-principles — they use cited vendor/config-derived overrides.
  Linear/DeltaNet recurrent state and sliding-window caps use a fixed
  per-sequence term with the dtype stated per preset (assumed where the
  vendor does not publish it).
- MoE is modeled with all experts resident (no expert parallelism/offload);
  decode ITL assumes total-param reads. Speculative decoding (DSpark/MTP) is
  not modeled.
- Presets with `spec_confidence: "unverified"` (DSv4.1-Flash KV; GLM-5/5.1
  family-inherited specs) are flagged in the UI and sized with derived values.
- KV offload/tiering, session-retention simulation, validated disaggregated
  serving, automated GPU provisioning, and commercial pricing are deferred.
- Rubin (Vera Rubin NVL72) and H200 NVL are comparison-only until availability,
  topology and compute values are verified; GB300 compute values are unsourced.
- The `vera_rubin_nvl72` / `h200_nvl` profiles have `compute_tflops: null`;
  latency bounds are UNVERIFIED (not failed) for those profiles.
- All latency and throughput numbers without an imported benchmark are
  optimistic ceiling estimates (linear bandwidth scaling, no queueing).