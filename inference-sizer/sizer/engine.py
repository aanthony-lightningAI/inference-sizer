"""Inference GPU sizing engine.

Memory fit sets GPUs per replica. Peak goodput at the latency SLO sets replica count.
Disaggregated prefill/decode is a mode flag, not a separate model.
"""

from __future__ import annotations

from math import ceil
from typing import Literal

from pydantic import BaseModel, Field


class SizeRequest(BaseModel):
    customer: str = "Customer"
    tier: str = "Interactive chat"
    params_b: float = Field(70, description="Total parameters, billions")
    active_b: float = Field(70, description="Active parameters, billions. Same as total for dense models.")
    layers: int = 80
    kv_heads: int = 8
    head_dim: int = 128
    attn: Literal["gqa", "mla"] = "gqa"
    mla_dim: int = Field(576, description="MLA cached elements per layer: kv_lora_rank + qk_rope_head_dim")
    weight_bytes: float = Field(1.0, description="Bytes per weight: 2 BF16, 1 FP8, 0.5 FP4")
    kv_bytes: float = Field(1.0, description="Bytes per KV element")
    rps: float = 20
    inflight: int = 160
    in_mean: int = 2048
    in_p95: int = 8192
    out_mean: int = 512
    out_p95: int = 2048
    prefix_hit: float = Field(0.4, ge=0, le=1)
    ttft_ms: float = 800
    itl_ms: float = 40
    util: float = Field(0.65, gt=0, le=1)
    margin_replicas: int = 1
    gpu_name: str = "B200 192GB"
    hbm_gb: float = 192
    bw_tb_s: float = 8
    flops_tflops: float = 4500
    bw_eff: float = Field(0.65, gt=0, le=1)
    flops_eff: float = Field(0.45, gt=0, le=1)
    max_gpus_per_replica: int = 8
    overhead_gb: float = 2
    disagg: bool = False
    link_gb_s: float = 50
    link_eff: float = Field(0.5, gt=0, le=1)


PRESETS = [
    {"name": "Llama 3.1 8B", "params_b": 8, "active_b": 8, "layers": 32, "kv_heads": 8, "head_dim": 128, "attn": "gqa"},
    {"name": "Llama 3.1 70B", "params_b": 70, "active_b": 70, "layers": 80, "kv_heads": 8, "head_dim": 128, "attn": "gqa"},
    {"name": "Llama 3.1 405B", "params_b": 405, "active_b": 405, "layers": 126, "kv_heads": 8, "head_dim": 128, "attn": "gqa"},
    {"name": "Qwen2.5 32B", "params_b": 32, "active_b": 32, "layers": 64, "kv_heads": 8, "head_dim": 128, "attn": "gqa"},
    {"name": "DeepSeek-V3 class (MLA)", "params_b": 671, "active_b": 37, "layers": 61, "kv_heads": 128, "head_dim": 128, "attn": "mla", "mla_dim": 576},
]

GPUS = [
    {"name": "H100 SXM 80GB", "hbm_gb": 80, "bw_tb_s": 3.35, "flops": {2: 989, 1: 1979, 0.5: 1979}},
    {"name": "H200 141GB", "hbm_gb": 141, "bw_tb_s": 4.8, "flops": {2: 989, 1: 1979, 0.5: 1979}},
    {"name": "B200 192GB", "hbm_gb": 192, "bw_tb_s": 8.0, "flops": {2: 2250, 1: 4500, 0.5: 9000}},
    {"name": "B300 288GB", "hbm_gb": 288, "bw_tb_s": 8.0, "flops": {2: 2500, 1: 5000, 0.5: 10000}},
]


def _gib(n: float) -> float:
    return n / (1024 ** 3)


def _powers(max_gpus: int) -> list[int]:
    out = []
    n = 1
    while n <= max_gpus:
        out.append(n)
        n *= 2
    if out[-1] != max_gpus:
        out.append(max_gpus)
    return out


def kv_bytes_per_token(req: SizeRequest) -> int:
    if req.attn == "mla":
        elements = req.mla_dim
    else:
        elements = 2 * req.kv_heads * req.head_dim
    return int(elements * req.layers * req.kv_bytes)


def effective_tokens(req: SizeRequest, p95: bool) -> float:
    inp = req.in_p95 if p95 else req.in_mean
    out = req.out_p95 if p95 else req.out_mean
    return inp * (1 - req.prefix_hit) + out


def _decode_fit(req: SizeRequest, gpus: int, batch: int, ctx: float) -> dict:
    weight = req.params_b * 1e9 * req.weight_bytes
    usable = req.hbm_gb * (1024 ** 3) * gpus - req.overhead_gb * (1024 ** 3) * gpus - weight
    kv_budget = max(0.0, usable * 0.88)
    kv_need = kv_bytes_per_token(req) * ctx * batch
    mem_ok = kv_need <= kv_budget and usable > 0
    agg_bw = gpus * req.bw_tb_s * 1e12 * req.bw_eff
    step = (weight + kv_need) / agg_bw if agg_bw else float("inf")
    goodput = batch / step if step else 0
    return {"mem_ok": mem_ok, "step": step, "goodput": goodput, "kv_need": kv_need, "kv_budget": kv_budget}


def _max_batch(req: SizeRequest, gpus: int, ctx: float) -> int:
    lo, hi, fit = 1, max(req.inflight, 1), 0
    itl = req.itl_ms / 1000
    while lo <= hi:
        mid = (lo + hi) // 2
        got = _decode_fit(req, gpus, mid, ctx)
        if got["mem_ok"] and got["step"] <= itl:
            fit = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return fit


def _weights_fit(req: SizeRequest, gpus: int) -> bool:
    weight = req.params_b * 1e9 * req.weight_bytes
    cap = req.hbm_gb * (1024 ** 3) * gpus * 0.92
    overhead = req.overhead_gb * (1024 ** 3) * gpus
    return weight + overhead <= cap


def size_colocated(req: SizeRequest) -> dict:
    ctx_mem = effective_tokens(req, True)
    ctx_thru = effective_tokens(req, False)
    peak_out = req.rps * req.out_mean
    best = None
    notes = []
    for gpus in _powers(req.max_gpus_per_replica):
        if not _weights_fit(req, gpus):
            continue
        batch = _max_batch(req, gpus, ctx_mem)
        if not batch:
            continue
        fit = _decode_fit(req, gpus, batch, ctx_thru)
        prefill_flops = 2 * req.active_b * 1e9 * req.in_mean * (1 - req.prefix_hit)
        prefill_time = prefill_flops / (gpus * req.flops_tflops * 1e12 * req.flops_eff)
        replicas = ceil(peak_out / (fit["goodput"] * req.util)) if fit["goodput"] else 10**9
        rec = {
            "gpus_per_replica": gpus,
            "concurrency": batch,
            "goodput_tok_s": fit["goodput"],
            "prefill_ms": prefill_time * 1000,
            "ttft_ok": prefill_time <= req.ttft_ms / 1000,
            "replicas": replicas,
            "fleet_gpus": gpus * (replicas + req.margin_replicas),
            "kv_per_seq_gib": _gib(kv_bytes_per_token(req) * ctx_mem),
        }
        if best is None or rec["fleet_gpus"] < best["fleet_gpus"] or (
            rec["fleet_gpus"] == best["fleet_gpus"] and rec["ttft_ok"] and not best["ttft_ok"]
        ):
            best = rec
    if best is None:
        notes.append(
            "No replica up to max GPUs fits weights plus p95 KV inside the inter-token SLO. "
            "Raise max GPUs per replica, cut p95 context, or lower KV dtype."
        )
    return {"replica": best, "notes": notes, "peak_output_tok_s": peak_out, "ctx_p95": ctx_mem, "ctx_mean": ctx_thru}


def size_disagg(req: SizeRequest) -> dict:
    ctx_mem = effective_tokens(req, True)
    peak_out = req.rps * req.out_mean
    peak_in = req.rps * req.in_mean * (1 - req.prefix_hit)
    notes = []
    decode = None
    for gpus in _powers(req.max_gpus_per_replica):
        if not _weights_fit(req, gpus):
            continue
        batch = _max_batch(req, gpus, ctx_mem)
        if not batch:
            continue
        fit = _decode_fit(req, gpus, batch, effective_tokens(req, False))
        replicas = ceil(peak_out / (fit["goodput"] * req.util)) if fit["goodput"] else 10**9
        rec = {
            "gpus_per_replica": gpus,
            "concurrency": batch,
            "goodput_tok_s": fit["goodput"],
            "replicas": replicas,
            "fleet_gpus": gpus * (replicas + req.margin_replicas),
        }
        if decode is None or rec["fleet_gpus"] < decode["fleet_gpus"]:
            decode = rec
    prefill = None
    for gpus in _powers(req.max_gpus_per_replica):
        if not _weights_fit(req, gpus):
            continue
        flops_per_token = 2 * req.active_b * 1e9
        tok_s = (gpus * req.flops_tflops * 1e12 * req.flops_eff) / flops_per_token
        replicas = ceil(peak_in / (tok_s * req.util)) if tok_s else 10**9
        one = (flops_per_token * req.in_p95 * (1 - req.prefix_hit)) / (gpus * req.flops_tflops * 1e12 * req.flops_eff)
        kv_move = (kv_bytes_per_token(req) * ctx_mem) / (req.link_gb_s * 1e9 * req.link_eff)
        ttft = one + kv_move
        rec = {
            "gpus_per_replica": gpus,
            "goodput_input_tok_s": tok_s,
            "replicas": replicas,
            "prefill_ms": one * 1000,
            "kv_transfer_ms": kv_move * 1000,
            "ttft_ms": ttft * 1000,
            "ttft_ok": ttft <= req.ttft_ms / 1000,
            "fleet_gpus": gpus * (replicas + req.margin_replicas),
        }
        if prefill is None or rec["fleet_gpus"] < prefill["fleet_gpus"]:
            prefill = rec
    if decode is None:
        notes.append("Decode pool does not fit at the ITL SLO within max GPUs per replica.")
    if prefill and not prefill["ttft_ok"]:
        notes.append("Prefill plus KV transfer exceeds the TTFT SLO. Add prefill GPUs or a faster KV link.")
    fleet = None
    if decode and prefill:
        fleet = decode["fleet_gpus"] + prefill["fleet_gpus"]
    return {"decode": decode, "prefill": prefill, "notes": notes, "fleet_gpus": fleet, "peak_input_tok_s": peak_in}


def size(req: SizeRequest) -> dict:
    notes = []
    if req.attn == "gqa" and req.kv_heads >= 32 and req.params_b > 200:
        notes.append("High KV-head count on a very large model. If this is MLA, switch attention or the cache is overstated.")
    coloc = size_colocated(req)
    dis = size_disagg(req) if req.disagg else None
    fleet = dis["fleet_gpus"] if dis and dis["fleet_gpus"] is not None else (coloc["replica"]["fleet_gpus"] if coloc["replica"] else None)
    return {
        "customer": req.customer,
        "tier": req.tier,
        "gpu": req.gpu_name,
        "mode": "disaggregated" if req.disagg else "colocated",
        "fleet_gpus": fleet,
        "weight_gib": _gib(req.params_b * 1e9 * req.weight_bytes),
        "kv_bytes_per_token": kv_bytes_per_token(req),
        "peak_output_tok_s": req.rps * req.out_mean,
        "colocated": coloc,
        "disaggregated": dis,
        "notes": notes + coloc["notes"] + (dis["notes"] if dis else []),
        "assumptions": [
            "KV bytes/token = elements/layer × layers × KV bytes.",
            "GQA elements = 2 × kv heads × head dim. MLA elements = latent + RoPE.",
            "Decode step reads all weights plus live KV. ITL = that read / effective HBM bandwidth.",
            "Prefill FLOPs ≈ 2 × active params × uncached input tokens.",
            "Replicas = ceil(peak tok/s / (goodput × utilization)).",
            "Ceiling model until bandwidth and compute efficiency are replaced with a measured replica.",
        ],
    }
