"""Pure sizing logic for dense (MHA/GQA/MQA), MLA, MoE and hybrid-attention
decoder models, dispatched on the KV-cache path (kv_model).

Shared by the API and the CLI. All quantities are bytes internally.

Key properties (each backed by a regression test):
- Prefix hit rate gives ZERO physical memory-sharing credit; retained-context
  memory is monotonic in context length. Prefix credit applies to prefill
  compute only.
- input p95 + output p95 is a conservative tail envelope, not the p95 of total
  length. Hard max-context limits apply for admission.
- Per-device memory is validated; pooled HBM is not treated as interchangeable.
- A candidate failing its own optimistic latency bound is rejected and the
  failing SLO is named. A passing bound never proves p95 SLO compliance.
- Serving replicas respect both request-rate and concurrent-request demand.
  Spares are added separately and excluded from serving capacity.
"""

from __future__ import annotations

from math import ceil

from .hardware import device_memory_bytes, get_profile
from .schemas import (
    FORMAT_BYTES,
    CandidateRecord,
    DeploymentSpec,
    Feasibility,
    MemoryComponents,
    ModelSpec,
    SizeRequest,
    SizingResult,
    WorkloadSpec,
)


def kv_bytes_per_token(model: ModelSpec) -> int:
    """KV bytes per token per sequence, dispatched on the KV-cache path:

    - mha_gqa: 2 (K and V) x kv_layers x kv_heads x head_dim x KV element bytes
    - mla:     kv_layers x (kv_lora_rank + qk_rope_head_dim) x KV element bytes
      (compressed latent + RoPE channel, one copy — replicated across TP)
    - override: vendor/config-published figure (hybrid sparse attention etc.)
    """
    if model.kv_model == "mha_gqa":
        return int(
            2 * model.effective_kv_layers() * model.kv_heads * model.head_dim
            * FORMAT_BYTES[model.kv_dtype]
        )
    if model.kv_model == "mla":
        return int(
            model.effective_kv_layers()
            * (model.kv_lora_rank + (model.qk_rope_head_dim or 0))
            * FORMAT_BYTES[model.kv_dtype]
        )
    if model.kv_model == "override":
        assert model.kv_bytes_per_token_override is not None  # schema-enforced
        return int(model.kv_bytes_per_token_override)
    raise ValueError(f"unknown kv_model {model.kv_model!r}")  # unreachable: KVModel literal is exhaustive


def context_lengths(workload: WorkloadSpec) -> tuple[int, int, str]:
    """Returns (tail envelope, mean context, label).

    Tail envelope = p95_input + p95_output, capped by the hard context limit.
    """
    tail_raw = workload.p95_input_tokens + workload.p95_output_tokens
    label = (
        "conservative tail envelope (input p95 + output p95); not the p95 of total length"
    )
    return tail_raw, workload.mean_input_tokens + workload.mean_output_tokens, label


def kv_sharding(kv_heads: int, tp: int) -> tuple[str, int]:
    """Returns (mode, kv_heads_per_device). kv heads below TP are replicated."""
    if kv_heads % tp == 0:
        return "sharded", kv_heads // tp
    if kv_heads < tp:
        return "replicated", kv_heads  # every device holds every KV head
    return "invalid", 0


def _kv_bytes_per_token_per_device(model: ModelSpec, tp: int) -> int:
    """Per-device growing-KV bytes/token. MHA/GQA shards kv heads across TP
    (replicating when heads < TP); MLA and override caches hold ONE compressed
    stream that serving engines replicate across TP — never sharded here."""
    if model.kv_model in ("mla", "override"):
        return kv_bytes_per_token(model)
    mode, heads_per_dev = kv_sharding(model.kv_heads, tp)
    if mode == "invalid":
        raise ValueError(
            f"kv_heads={model.kv_heads} is not divisible by TP={tp} and not replicable (< TP)"
        )
    return int(
        2 * model.effective_kv_layers() * heads_per_dev * model.head_dim
        * FORMAT_BYTES[model.kv_dtype]
    )


def _kv_fixed_per_device(model: ModelSpec, tp: int) -> int:
    """Per-device share of the per-sequence constant KV-side memory (sliding-
    window caps, linear/DeltaNet recurrent state). Head-parallel state shards
    across TP like KV heads."""
    fixed = model.kv_fixed_bytes_per_sequence or 0
    return int(fixed / tp)


def _weights_bytes_per_device(model: ModelSpec, tp: int) -> float:
    return model.total_params * model.weight_bytes_each() / tp


def _candidate_memory(
    model: ModelSpec, dep: DeploymentSpec, profile, device_mem_bytes: int, mem_source: str,
    tp: int, kv_per_tok_per_dev: int, kv_fixed_per_dev: int, batch: int, ctx_tail: int,
) -> MemoryComponents:
    weights = _weights_bytes_per_device(model, tp)
    runtime = dep.runtime_overhead_gb * 1e9
    graphs = dep.cuda_graphs_gb * 1e9
    activations = dep.activations_workspace_gb * 1e9
    reserve = dep.reserve_fraction * device_mem_bytes
    kv_p95 = kv_per_tok_per_dev * ctx_tail * batch
    kv_fixed = kv_fixed_per_dev * batch
    total = weights + runtime + graphs + activations + reserve + kv_p95 + kv_fixed
    return MemoryComponents(
        weights_bytes=weights,
        kv_p95_bytes_per_replica=kv_per_tok_per_dev * ctx_tail * batch * tp,  # replica total
        kv_fixed_bytes_per_replica=kv_fixed * tp,
        runtime_overhead_bytes=runtime,
        cuda_graphs_bytes=graphs,
        activations_workspace_bytes=activations,
        reserve_bytes=reserve,
        total_bytes_per_replica=total * tp,
        total_bytes_per_device=total,
        device_memory_bytes=device_mem_bytes,
        device_memory_source=mem_source,  # type: ignore[arg-type]
    )


def _decode_itl_ms(
    model: ModelSpec, dep: DeploymentSpec, profile, tp: int,
    kv_per_tok_per_dev: int, kv_fixed_per_dev: int, batch: int, ctx_mean: int,
) -> float:
    """Optimistic decode-step bound: one step reads all weights plus live KV
    (growing part at mean context, plus the per-sequence constant KV-side term)."""
    agg_bw = tp * profile.hbm_bandwidth_tb_s * 1e12 * dep.bw_eff
    read_bytes = (
        _weights_bytes_per_device(model, tp)
        + kv_per_tok_per_dev * ctx_mean * batch
        + kv_fixed_per_dev * batch
    )
    return read_bytes / agg_bw * 1000.0


def _prefill_ms(
    model: ModelSpec, dep: DeploymentSpec, profile, tp: int, input_tokens: int,
) -> float:
    """Optimistic prefill bound on the given (conservative) input token count."""
    flops = 2 * (model.active_params or model.total_params) * input_tokens
    peak = tp * (profile.compute_tflops or {}).get("bf16", 0) * 1e12 * dep.flops_eff
    return flops / peak * 1000.0 if peak else float("inf")


def _prefill_input_tok_s(model: ModelSpec, dep: DeploymentSpec, profile, tp: int) -> float:
    peak = tp * (profile.compute_tflops or {}).get("bf16", 0) * 1e12 * dep.flops_eff
    flops_per_tok = 2 * (model.active_params or model.total_params)
    return peak / flops_per_tok if flops_per_tok else float("inf")


def _max_batch(
    model: ModelSpec, dep: DeploymentSpec, profile, device_mem_bytes: int,
    tp: int, kv_per_tok_per_dev: int, kv_fixed_per_dev: int, ctx_tail: int, ctx_mean: int,
    batch_cap: int, decode_ms_slo: float,
) -> tuple[int, float, str | None]:
    """Largest batch fitting per-device memory at the tail envelope AND the ITL
    bound. Returns (batch, itl_ms, limit) where limit names the binding factor."""
    fixed = (
        _weights_bytes_per_device(model, tp)
        + dep.runtime_overhead_gb * 1e9
        + dep.cuda_graphs_gb * 1e9
        + dep.activations_workspace_gb * 1e9
        + dep.reserve_fraction * device_mem_bytes
    )
    budget = device_mem_bytes - fixed
    if budget <= 0:
        return 0, float("inf"), "memory"
    # Binary search on batch: both memory (at tail context) and ITL (at mean
    # context) increase monotonically with batch.
    lo, hi, best, itl_best, limit_best = 1, batch_cap, 0, float("inf"), "memory"
    while lo <= hi:
        mid = (lo + hi) // 2
        mem_ok = (
            kv_per_tok_per_dev * ctx_tail * mid + kv_fixed_per_dev * mid <= budget
        )
        itl = _decode_itl_ms(
            model, dep, profile, tp, kv_per_tok_per_dev, kv_fixed_per_dev, mid, ctx_mean
        )
        if mem_ok and itl <= decode_ms_slo:
            best, itl_best, limit_best = mid, itl, "memory"
            lo = mid + 1
        elif not mem_ok:
            hi = mid - 1
            limit_best = "memory"
        else:
            hi = mid - 1
            limit_best = "itl"
    return best, itl_best, limit_best


def _evaluate_candidate(
    req: SizeRequest, tp: int, device_mem_bytes: int, mem_source: str,
    ctx_tail: int, ctx_mean: int, rps: float, inflight: int, batch_cap: int,
) -> CandidateRecord:
    model, dep, workload, profile = (
        req.model, req.deployment, req.workload,
        get_profile(req.deployment.hardware_profile_id),
    )
    reasons: list[str] = []
    limit: str | None = None

    # ---- divisibility and KV placement
    heads = model.attention_heads
    if heads is not None and heads % tp != 0:
        reasons.append(f"attention_heads={heads} not divisible by TP={tp}")
    if model.kv_model == "mha_gqa":
        mode, heads_per_dev = kv_sharding(model.kv_heads, tp)
        if mode == "invalid":
            reasons.append(
                f"kv_heads={model.kv_heads} not divisible by TP={tp} and not replicable (< TP)"
            )
            return CandidateRecord(gpus_per_replica=tp, feasible=False, rejection_reasons=reasons,
                                   kv_per_token_bytes=kv_bytes_per_token(model))
    else:
        # MLA and override caches hold one compressed stream replicated across TP.
        mode, heads_per_dev = "replicated", None
    kv_per_tok_per_dev = _kv_bytes_per_token_per_device(model, tp)
    kv_fixed_per_dev = _kv_fixed_per_device(model, tp)

    # ---- memory fit (fixed terms + weights, before any KV)
    fixed = (
        _weights_bytes_per_device(model, tp)
        + dep.runtime_overhead_gb * 1e9
        + dep.cuda_graphs_gb * 1e9
        + dep.activations_workspace_gb * 1e9
        + dep.reserve_fraction * device_mem_bytes
    )
    if fixed > device_mem_bytes:
        reasons.append(
            f"weights+runtime+graphs+activations+reserve ({fixed / 1e9:.1f} GB/device) "
            f"exceed device memory ({device_mem_bytes / 1e9:.1f} GB)"
        )
        return CandidateRecord(
            gpus_per_replica=tp, feasible=False, rejection_reasons=reasons,
            kv_per_token_bytes=kv_bytes_per_token(model), kv_sharding=mode,
            kv_heads_per_device=heads_per_dev, limit="memory",
        )

    # ---- batch inside ITL bound and memory
    batch, itl_ms, batch_limit = _max_batch(
        model, dep, profile, device_mem_bytes, tp, kv_per_tok_per_dev, kv_fixed_per_dev,
        ctx_tail, ctx_mean, batch_cap, workload.decode_ms_p95,
    )
    if batch == 0:
        reasons.append("no concurrent batch fits memory and the decode-latency bound")
        return CandidateRecord(
            gpus_per_replica=tp, feasible=False, rejection_reasons=reasons,
            kv_per_token_bytes=kv_bytes_per_token(model), kv_sharding=mode,
            kv_heads_per_device=heads_per_dev, limit=batch_limit or "memory",
        )

    memory = _candidate_memory(
        model, dep, profile, device_mem_bytes, mem_source, tp,
        kv_per_tok_per_dev, kv_fixed_per_dev, batch, ctx_tail,
    )

    # ---- optimistic prefill (TTFT) bound on the conservative input tail
    uncached_tail = int(workload.p95_input_tokens * (1 - workload.prefix_hit_rate))
    if profile.compute_tflops is None:
        # Compute values unsourced for this profile: no prefill bound exists.
        # Latency is UNVERIFIED, not failed; memory fit still evaluates.
        prefill_ms = None
        ttft_ok = True
        ttft_unknown = True
    else:
        prefill_ms = _prefill_ms(model, dep, profile, tp, uncached_tail)
        ttft_ok = prefill_ms <= workload.ttft_ms_p95
        ttft_unknown = False

    # ---- capacity under the operating factor, applied once
    util = dep.operating_factor
    mean_uncached = int(workload.mean_input_tokens * (1 - workload.prefix_hit_rate))
    prefill_mean_ms = _prefill_ms(model, dep, profile, tp, mean_uncached) if not ttft_unknown else 0.0
    response_s = (prefill_mean_ms + workload.mean_output_tokens * itl_ms) / 1000.0
    littles_rps = batch / response_s if response_s > 0 else float("inf")
    prefill_tok_s = _prefill_input_tok_s(model, dep, profile, tp) if not ttft_unknown else float("inf")
    prefill_rps = prefill_tok_s / mean_uncached if mean_uncached > 0 else float("inf")
    rps_cap = min(littles_rps, prefill_rps)

    serving_from_rps = ceil(rps / (rps_cap * util)) if rps > 0 else 0
    serving_from_conc = ceil(inflight / (batch * util)) if inflight > 0 else 0
    serving = max(serving_from_rps, serving_from_conc)

    feasible = ttft_ok and serving < 10**6
    limit = None
    if ttft_unknown:
        reasons.append(
            "compute values unsourced for this profile: TTFT bound unverified (memory fit only)"
        )
    elif not ttft_ok:
        reasons.append(
            f"fails the optimistic TTFT bound: prefill estimate {prefill_ms:.0f} ms "
            f"> ttft_ms_p95 {workload.ttft_ms_p95:.0f} ms (optimistic bound, not a measured p95)"
        )
        limit = "ttft"
    if rps > 0 and serving_from_rps >= 10**6:
        reasons.append("request-rate requirement cannot be met by this candidate")
        limit = limit or "prefill_throughput"
    if feasible:
        if serving_from_conc > serving_from_rps:
            limit = "concurrency"
        elif serving_from_rps > 0:
            limit = "prefill_throughput" if prefill_rps <= littles_rps else "request_rate"
        else:
            limit = "memory" if batch_limit == "memory" else "itl"

    return CandidateRecord(
        gpus_per_replica=tp, feasible=feasible, rejection_reasons=reasons,
        memory=memory, kv_per_token_bytes=kv_bytes_per_token(model),
        kv_sharding=mode, kv_heads_per_device=heads_per_dev,
        max_concurrency=batch, prefill_bound_ms=prefill_ms, itl_bound_ms=itl_ms,
        rps_capacity=rps_cap, serving_replicas=serving if feasible else None,
        limit=limit,
    )


def _tp_candidates(dep: DeploymentSpec) -> list[int]:
    out = set()
    n = 1
    while n <= dep.max_gpus_per_replica:
        out.add(n)
        n *= 2
    for topo in (8, 72):
        if topo <= dep.max_gpus_per_replica:
            out.add(topo)
    return sorted(out)


def _traffic(req: SizeRequest) -> tuple[float, int, list[str], dict]:
    """Resolve request rate and concurrent demand; surface inconsistencies."""
    w = req.workload
    warnings: list[str] = []
    derived: dict = {}
    rps = w.peak_rps
    if w.derived_rps() is not None:
        derived_rps = w.derived_rps()
        derived = {
            "active_users": w.active_users,
            "requests_per_active_per_hour": w.requests_per_active_per_hour,
            "calls_per_task": w.calls_per_task or 1.0,
            "derived_rps": derived_rps,
        }
        if rps == 0:
            rps = derived_rps
            derived["used"] = "derived from active-user cadence x calls per task"
        elif abs(derived_rps - rps) / max(derived_rps, rps) > 0.1:
            warnings.append(
                f"Explicit peak_rps ({rps}) differs by >10% from the rate derived from "
                f"active-user cadence ({derived_rps:.3f}); using explicit peak_rps."
            )
    inflight = w.peak_inflight
    return rps, inflight, warnings, derived


def size(req: SizeRequest, benchmark: dict | None = None) -> SizingResult:
    benchmark = benchmark if benchmark is not None else req.benchmark
    profile = get_profile(req.deployment.hardware_profile_id)
    result = SizingResult(
        customer=req.customer, tier=req.tier, request=req, hardware_profile=profile,
    )
    result.assumptions = [
        "KV bytes/token by path — MHA/GQA: 2 x kv_layers x kv_heads x head_dim x KV element "
        "bytes; MLA: kv_layers x (kv_lora_rank + qk_rope_head_dim) x KV element bytes, "
        "replicated across TP; override: cited published/derived figure.",
        "KV memory accounts for the FULL retained input plus generated context; prefix "
        "hit rate gives no physical memory-sharing credit and only reduces prefill compute.",
        "input p95 + output p95 is a conservative tail envelope, not the p95 of total length.",
        "Per-device memory fit is validated (weights sharded by TP; growing KV sharded or "
        "replicated per path; per-sequence constant KV-side memory sharded across TP).",
        "Decode ITL and prefill times are optimistic bounds (linear bandwidth scaling, no "
        "queueing or prefill/decode contention); passing does not prove p95 SLO compliance.",
        "Response duration for the request-rate capacity uses Little's law: batch / (prefill_mean + out_mean x ITL).",
        f"Operating factor {req.deployment.operating_factor} applied once to capacity before rounding.",
        "Spare replicas are added after serving replicas and excluded from serving capacity.",
    ]
    model = req.model
    if model.kv_fixed_bytes_per_sequence:
        result.assumptions.append(
            f"Per-sequence constant KV-side memory of {model.kv_fixed_bytes_per_sequence:,} bytes "
            "(sliding-window caps and/or linear-attention recurrent state) is added to every "
            "concurrent sequence; derivation and confidence are documented in the catalog."
        )
    if model.kv_model == "override":
        result.assumptions.append(
            f"KV cache uses the cited per-token figure ({model.kv_bytes_per_token_override:,} B) "
            f"because the cache layout ({model.attn}) is not covered by closed-form formulas; "
            f"source: {model.kv_override_source}."
        )
    if model.num_experts is not None:
        result.assumptions.append(
            f"MoE: all {model.num_experts} routed experts are resident (weights = total "
            f"params); prefill compute uses active params ({(model.active_params or 0) / 1e9:.1f}B); "
            "decode ITL conservatively reads full weights each step; expert parallelism "
            "(EP) is not modeled — experts are assumed TP-sharded."
        )

    # ---- architecture support: unsupported only when the KV path cannot compute.
    # ModelSpec validation makes incomplete MLA/override specs unconstructible,
    # so this is a defensive re-check, not the primary gate.
    try:
        kv_bytes_per_token(model)
    except (ValueError, AssertionError, TypeError) as e:
        result.architecture_status = "unsupported"
        result.feasibility = Feasibility.UNSUPPORTED
        result.rejection_reasons.append(f"KV cache path unusable: {e}")
        result.notes.append("Unsupported architectures never receive sizing recommendations.")
        return result

    # ---- workload: zero traffic vs zero capacity
    rps, inflight, warnings, derived = _traffic(req)
    result.warnings.extend(warnings)
    result.traffic_derived = derived
    zero_traffic = rps == 0 and inflight == 0
    if zero_traffic:
        result.notes.append(
            "Zero traffic declared: no serving capacity required. This is distinct from a "
            "candidate with zero effective capacity."
        )

    tail_raw, ctx_mean, envelope_label = context_lengths(req.workload)
    max_ctx = req.model.max_context_tokens
    ctx_tail = tail_raw
    if max_ctx and tail_raw > max_ctx:
        result.warnings.append(
            f"Tail envelope ({tail_raw} tokens) exceeds the hard context limit ({max_ctx}); "
            f"memory uses the context limit and admission limits apply beyond it."
        )
        ctx_tail = max_ctx
    result.notes.append(f"Memory context: {envelope_label}.")
    if req.model.attention_heads is None:
        result.notes.append(
            "attention_heads not provided: parallelism divisibility could not be verified; "
            "treated as an unverified assumption."
        )

    device_mem_bytes, mem_source = device_memory_bytes(profile, req.deployment.observed_memory_gb)

    # A compatible, non-synthetic benchmark restricts candidates to the MEASURED
    # parallelism; measured capacity never extrapolates across TP degrees.
    bench_profile = None
    if benchmark is not None:
        from .benchmarks import compatibility_report, parse_profile

        try:
            bp = parse_profile(benchmark)
        except Exception as e:
            result.warnings.append(f"Benchmark profile rejected: {e}")
            bp = None
        if bp is not None and not bp.synthetic and (
            req.benchmark_profile_id is None or bp.profile_id == req.benchmark_profile_id
        ):
            mismatches = compatibility_report(bp, req)
            if not mismatches:
                bench_profile = bp
            else:
                result.warnings.append(
                    "Benchmark profile is not compatible with this request; status remains "
                    "Estimated. Material mismatches: " + "; ".join(mismatches)
                )
                result.benchmark = {"profile_id": bp.profile_id, "mismatches": mismatches}

    if bench_profile is not None:
        tp_list = [bench_profile.gpus_per_replica]
    else:
        tp_list = _tp_candidates(req.deployment)
    batch_cap = inflight if inflight > 0 else 256
    candidates: list[CandidateRecord] = []
    for tp in tp_list:
        if req.deployment.allocation_size_gpus and tp > req.deployment.allocation_size_gpus:
            continue  # provider never allocates fewer GPUs than the block size
        candidates.append(_evaluate_candidate(
            req, tp, device_mem_bytes, mem_source, ctx_tail, ctx_mean,
            rps, inflight, batch_cap,
        ))
    result.rejection_reasons.extend(
        f"TP={c.gpus_per_replica}: {r}" for c in candidates if not c.feasible for r in c.rejection_reasons
    )

    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        result.feasibility = Feasibility.INFEASIBLE
        if not result.rejection_reasons:
            result.rejection_reasons.append("no candidate satisfies memory and latency bounds")
        return result

    feasible.sort(key=lambda c: (c.gpus_per_replica * (c.serving_replicas or 0), c.gpus_per_replica))
    best = feasible[0]

    serving = 0 if zero_traffic else (best.serving_replicas or 0)
    spare = req.deployment.spare_replicas
    result.selected = best
    result.memory_components = best.memory
    result.serving_replicas = serving
    result.spare_replicas = spare
    result.fleet_gpus = best.gpus_per_replica * (serving + spare)
    result.allocation_size_gpus = req.deployment.allocation_size_gpus
    result.capacity_rps = (best.rps_capacity or 0) * req.deployment.operating_factor * max(serving, 0)
    result.capacity_concurrency = (best.max_concurrency or 0) * max(serving, 0)
    result.feasibility = Feasibility.FEASIBLE
    result.limiting_factors = [f"TP={best.gpus_per_replica} bound by {best.limit}"]
    if zero_traffic:
        result.limiting_factors.append("zero declared traffic: serving replicas = 0")
    if profile.recommendation_mode == "comparison_only":
        result.notes.append(
            f"{profile.name} is comparison-only in this build: it cannot produce a "
            "supported deployment recommendation (availability/qualification unverified)."
        )
    if any("TTFT bound unverified" in r for r in result.rejection_reasons) or (
        best.limit in ("memory", "itl", "concurrency") and profile.compute_tflops is None
    ):
        result.warnings.append(
            "Compute values for this profile are unsourced: latency bounds are UNVERIFIED, "
            "not passed. Memory fit is still evaluated."
        )
    if req.deployment.allocation_size_gpus:
        result.notes.append(
            f"Provider allocation size is {req.deployment.allocation_size_gpus} GPUs; this is "
            f"distinct from the {best.gpus_per_replica} GPUs used by one replica."
        )
    if spare > 0:
        result.notes.append(
            f"Failure policy: {spare} spare replica(s) cover a nominated failure and are "
            f"excluded from normal serving capacity."
        )
    if rps > 0 and inflight > 0:
        result.notes.append(
            "Peak simultaneous demand is an independent constraint; the larger of the "
            "request-rate and concurrent requirements sets serving replicas."
        )

    # ---- evidence: benchmark calibration only via the adapter
    if benchmark is not None or req.benchmark_profile_id:
        from .benchmarks import apply_calibration  # local import keeps engine standalone

        apply_calibration(req, result, benchmark)
    return result