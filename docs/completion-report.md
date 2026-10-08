# Completion report — Lightning AI Inference Sizer

Covers the original rebuild of the sizing tool (from the `inference-sizer.zip`
spec) and the subsequent model-catalog + engine extension (MLA / MoE / hybrid
architectures, 2026-10-08). Every claim below points at a verifiable artifact:
a test, a file, a doc, or a command. No unverified claims.

## 1. Scope delivered

| Spec requirement | Where it lives |
| --- | --- |
| Sizing engine (memory fit, KV, latency bounds, fleet math) | `inference-sizer/sizer/engine.py` |
| Versioned request/result schemas + migration | `inference-sizer/sizer/schemas.py` (`SCHEMA_VERSION = 2`, `migrate_request_payload`) |
| Model normalization + provenance | `inference-sizer/sizer/normalize.py` (preset / config.json / manual / inferred; dense, MLA, MLA-hybrid, DeltaNet-hybrid classification) |
| Hardware catalog (7 profiles, independent statuses) | `inference-sizer/sizer/hardware.py` + `sizer/data/hardware.json` |
| Model preset catalog (24 presets with provenance) | `sizer/data/hardware.json` (`CATALOG_VERSION = 3`) |
| Benchmark adapter | `inference-sizer/sizer/benchmarks.py` (`benchmark_profile/v1`, compatibility matching) |
| API + one-origin frontend serving | `inference-sizer/server.py` (`/api/health`, `/api/catalog`, `/api/size`) |
| Frontend (no framework, Vite) | `inference-sizer/web/src/` |
| Packaging | `Dockerfile` (multi-stage: Vite build → python:3.12-slim), `samples/` |
| Docs | `inference-sizer/README.md`, `docs/` (this folder) |

## 2. Baseline engine bugs fixed

Probes recorded against the unmodified zip source in
[`docs/baseline-failures.md`](baseline-failures.md) (`probes/baseline_probes.py`);
regression checks live in `tests/`.

1. **ttft_bound** — a candidate failing its own optimistic TTFT bound by 85x
   was still selected. Now candidates failing their own optimistic TTFT/ITL
   bound are rejected with the failing SLO named (`tests/test_engine.py`).
2. **fleet_concurrency** — replicas were derived from output throughput only;
   10,000 in-flight demand was served by 1 replica with capacity 103. Now
   serving replicas respect BOTH request-rate and concurrent demand.
3. **prefix_memory** — a 100% prefix hit erased the retained input from KV
   memory (1.56 → 0.31 GiB). Now prefix hit rate gives zero physical
   memory-sharing credit and only reduces prefill compute.
4. **invalid_inputs** — `max_gpus_per_replica=0` raised `IndexError`;
   `layers=-5` was accepted and produced negative KV. Now pydantic validation
   rejects both (covered by e2e checks 4–5 and `tests/`).

CLI↔API agreement: both call the same pure `size()`; re-verified by
`tests/test_cli_api_agreement.py`.

## 3. Hardware catalog

- 7 profiles with per-profile independent statuses and source metadata
  (`sizer/data/hardware.json`).
- Applied corrections: HGX B200 naming guard; H200 SXM NVL8 vs NVL split;
  DGX H100 SLC→SEA1 availability note.
- Preserved discrepancy: B300 270 GB vs 288 GB marketing figures kept as
  distinct nominal/usable values.
- `vera_rubin_nvl72` / `h200_nvl` have `compute_tflops: null` → latency
  bounds are UNVERIFIED (not failed) for those profiles.

## 4. Extended model catalog + engine (this increment)

**Research-first, provenance-tracked.** Every new preset was verified against
public sources before entry; sources, derivations, and confidence are recorded
per preset in [`docs/model-catalog-notes.md`](model-catalog-notes.md).

- **Three KV-cache paths** (`sizer/engine.py`, dispatched on `model.kv_model`):
  - `mha_gqa`: `2 × layers-with-KV × kv_heads × head_dim × kv_bytes` — the
    original formula, now subsetting to `attention_kv_layers` for hybrids.
  - `mla`: `layers-with-KV × (kv_lora_rank + qk_rope_head_dim) × kv_bytes`,
    replicated across TP (matches vLLM/Mooncake MLA layouts).
  - `override`: vendor/config-derived bytes/token with mandatory
    `kv_override_source`; schema rejects an override without provenance.
- **Fixed per-sequence KV term** (`kv_fixed_bytes_per_sequence`) for
  sliding-window caps, linear/DeltaNet recurrent state and compressed sparse
  pools; sharded by TP in memory fit and batch sizing.
- **MoE semantics**: memory fit uses total params (all experts resident);
  prefill uses active params; decode ITL keeps total-param reads
  (conservative). Expert parallelism not modeled (assumption text).
- **Schema v2** (`SCHEMA_VERSION = 2`, calculator 1.1.0): new fields +
  `family`, `spec_confidence`, `notes`; v1 payloads migrate on import
  (`migrate_request_payload`, round-trip tested).
- **Catalog v3**: 24 presets (5 original + deepseek_v3_mla enabled via the MLA
  path + 18 new across DeepSeek V4/V4.1, GLM-5/5.1/5.2/5.3-Flash, Kimi
  K2.6/K2.7/K3, Qwen 3.5/3.6/3.8 verified sizes + Flash-Next, Inkling,
  Inkling-Small, Gemma 4 31B/26B-A4B). Every preset carries `family`,
  `spec_confidence` and source provenance; unverified entries (DSv4.1-Flash
  KV, GLM-5/5.1 family-inherited) are flagged in the UI.
- **Nonexistent combinations documented, never fabricated**: Qwen3.5-235B,
  Qwen3.6-122B/235B, Qwen3.8-35B/122B/235B; Muse Spark 1.3 is proprietary
  (no public specs) — see the not-found list in
  [`docs/model-catalog-notes.md`](model-catalog-notes.md).
- **UI**: family-grouped preset dropdown (`<optgroup>`), architecture
  badges (MoE / MLA cache / Hybrid attention / KV override), "Specs
  unverified" chip, Advanced section with the v2 fields, fixed-KV memory row,
  per-path KV placement text (`web/src/main.js`, `render.js`, `state.js`).

## 5. Verification evidence

- **Unit/integration**: `PYTHONPATH=inference-sizer .venv/bin/python -m pytest
  tests/ -q` → **74 passed** (52 carried from the original build + 22 new in
  `tests/test_v2_capabilities.py`: MLA formula/replication, override
  provenance requirements, MoE total-vs-active semantics, fixed-term math vs
  independent recomputation for every derived preset, v1→v2 migration,
  catalog integrity, hybrid-import guardrails).
- **Browser e2e**: `node e2e/verify.js` (server on :8000) → **16/16 checks
  passed**, screenshots in `e2e/shot-*.png`. New checks: MLA preset
  (kimi_k2_6) selects + badges + sizes feasible on GB300; hybrid preset
  (qwen_3_8_27b) with fixed-state term; override preset (DSv4 Flash 0731);
  unverified-confidence chip (DSv4.1-Flash); family-grouped dropdown;
  MLA-without-`kv_lora_rank` rejection now names the missing field.
- **Container**: rebuilt from the current tree (`docker build -t
  inference-sizer .`), serving on :8000 — `/api/health` reports
  `{"schema_version": 2, "calculator_version": "1.1.0"}`.
- **KV math audit**: every derived KV figure in the catalog equals an
  independent recomputation from published config values
  (`tests/test_v2_capabilities.py::test_kv_math_matches_documented_derivations`,
  derivations documented per preset in `docs/model-catalog-notes.md`).

## 6. Evidence-status honesty

- The tool is **Estimated-mode only**: no GPU benchmark has been run against
  any preset or profile. All latency/throughput numbers are optimistic
  ceiling estimates (linear bandwidth scaling, no queueing). Calibration path:
  [`docs/benchmark-collection.md`](benchmark-collection.md).
- Synthetic benchmark fixtures are accepted by the adapter but never used as
  calibration evidence (`fixtures/benchmarks/`, marked synthetic).
- A passing optimistic bound never proves p95 SLO compliance.

## 7. Known limitations / open items

- CSA/HCA, compressed sparse pools and unified-K/V attention are not modeled
  first-principles — cited override figures only. Linear/DeltaNet state dtype
  is assumed where vendors do not publish it (stated per preset).
- MoE: all experts resident; no expert parallelism/offload; speculative
  decoding (DSpark/MTP) not modeled.
- KV offload/tiering, session-retention simulation, validated disaggregated
  serving, automated GPU provisioning, and commercial pricing are deferred.
- Provisional branding scraped from lightning.ai; pending the official brand
  kit — never present it as approved Lightning branding.
- Container image is built and verified locally; not pushed to a registry.

## 8. Reproduce

```bash
PYTHONPATH=inference-sizer .venv/bin/python -m pytest tests/ -q   # 74 passed
docker build -t inference-sizer . && docker run -p 8000:8000 inference-sizer
node e2e/verify.js                                                # 16/16 (server on :8000)
```