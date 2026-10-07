# Benchmark collection and calibration instructions

Goal: turn an **Estimated** sizing result into a **Benchmark-calibrated** one for
one compatible model / GPU / engine configuration. This is a separate job from
running the calculator; it needs GPU serving and load generation access.

## What calibrates what

A benchmark profile (`benchmark_profile/v1`, see
`sizer/benchmarks.py`) calibrates a scenario ONLY when every material field
matches: model architecture + checkpoint revision + parameter count,
quantization, weight format, KV dtype, engine name/version, hardware
SKU/profile, GPUs per replica (parallelism), token-length distribution, and
prefix behavior. Any material change strips the calibrated label — profiles
never extrapolate across models, GPUs, engines, or workloads.

Synthetic fixtures (files marked `"synthetic": true`, e.g.
`fixtures/benchmarks/synthetic_hgx_b200_70b.json`) exercise the adapter and
matching logic but are **excluded from real calibration evidence** and must
never be presented as measured GPU results.

## Collection procedure (vLLM)

1. **Serve the exact configuration.** Same model revision, quantization, dtype
   flags, tensor-parallel degree, and engine version that the scenario uses:

   ```bash
   vllm serve <model> \
     --tensor-parallel-size <TP> \
     --dtype <weight dtype> \
     --kv-cache-dtype <kv dtype> \
     --version  # record the exact version for the profile
   ```

2. **Verify memory and KV capacity.** From engine logs, record observed
   per-device allocatable memory and the engine-reported KV cache capacity.
   Enter the observed memory as `observed_memory_gb` in the sizer. Startup cache
   capacity is a memory observation, NOT a throughput or latency guarantee.

3. **Sweep concurrency and arrival rate** with the GuideLLM benchmarking CLI
   (or `vllm bench serve`) using FIXED token distributions matching the
   workload (mean/p95 input and output lengths):

   ```bash
   # example: fixed distribution, sweep arrival rate
   guidellm benchmark --target http://localhost:8000 \
     --model <model> \
     --data "prompt_tokens={mean:2048,p95:8192},output_tokens={mean:512,p95:2048}" \
     --rate-type sweep --max-seconds 60
   ```

   Find the highest arrival rate / concurrency where the configured SLOs hold.

4. **Test cold and warm prefixes separately.** A warm-prefix run measures
   prefix reuse; a cold run measures worst-case prefill. Record
   `prefix_behavior` (`cold` | `warm` | `mixed`) — the adapter rejects a cold
   profile for a scenario that assumes prefix hits.

5. **Record a full run**, not a spot check: duration, request count, completion
   and error rates, SLO miss rate, and the jointly measured capacity —
   sustainable rps AND concurrent generations measured together (never rps from
   one run and concurrency from another).

## Profile fields (benchmark_profile/v1)

`profile_id, synthetic, source, collected_at, model_ref, checkpoint_revision,
architecture, attn, total_params, quantization, weight_format,
compute_precision, kv_dtype, engine_name, engine_version, hardware_sku,
hardware_profile_id, gpus_per_replica, token_length_distribution,
prefix_behavior, arrival_rate_rps, concurrency, duration_s, request_count,
latency_objectives, metrics`.

Metric definitions to preserve: **p95 TTFT**, the chosen **decode metric** (ITL
percentiles are token-level; average time-per-output-token is a different
number — record what you measured), **completion rate**, **error rate**,
**SLO miss rate**, **sustainable rps / concurrency**, and **goodput** (requests
per second meeting the configured latency objectives).

## Applying a profile

- API: include it inline in the sizing request (`benchmark` field +
  `benchmark_profile_id`).
- UI: "Import benchmark JSON…" in step 4. Status flips to **Benchmark
  calibrated** only on a full material match; otherwise the result stays
  **Estimated** with the mismatch list.
- CLI: place the profile in the request JSON (`benchmark` key) or import a full
  scenario export that carries it.