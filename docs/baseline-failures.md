# Baseline failure probes (unmodified engine, 2026-10-07)

Recorded against the unmodified source extracted from `inference-sizer.zip`
(commit `08bd435`) with fastapi 0.142.4 / pydantic 2.13.5 / Python 3.12.13.
Reproduce with `PYTHONPATH=inference-sizer .venv/bin/python probes/baseline_probes.py`.

Default request: Llama 3.1 70B preset (70B, 80 layers, GQA 8x128), B200 entry,
rps=20, inflight=160, in 2048/8192, out 512/2048, prefix_hit=0.4, ttft=800 ms,
itl=40 ms, max 8 GPUs/replica, margin 1.

## ttft_bound

`ttft_ms=1` → **fleet_gpus: 4**, estimated prefill **85.0 ms**, `ttft_ok: False`.
The candidate that fails its own optimistic bound by 85x is still selected; the
result presents 4 fleet GPUs with no SLO failure surfaced as a rejection.

## fleet_concurrency

`rps=0.1, inflight=10000` → **replicas: 1** (serving), **fleet_gpus: 2**. The
selected replica has a decode-concurrency capacity of **103** requests, far below
the stated 10,000 in-flight demand. Replicas are derived from output throughput
only; concurrent demand never constrains the fleet.

## prefix_memory

`prefix_hit=0.0` → p95 KV per sequence **1.56 GiB**.
`prefix_hit=1.0` → p95 KV per sequence **0.31 GiB**.
A full prefix hit **erases the retained input context** from the KV memory
calculation (only 512 output tokens remain). Longer retained context reduces
memory demand.

## invalid_inputs

`max_gpus_per_replica=0` → **IndexError: list index out of range** (raised by
`_powers(0)` indexing `out[-1]`).
`layers=-5` → **accepted**; produces `kv_bytes_per_token: -10240` (negative KV).

## CLI / API agreement

`python -m sizer.cli` (stdin/defaults) and `POST /api/size` with the same
`SizeRequest` both call `size()`; outputs are identical for the default request
and for the ttft_bound probe (both use the same pure function — agreement
confirmed by construction and re-verified after each engine change via
`tests/test_api.py::test_cli_api_agreement`).