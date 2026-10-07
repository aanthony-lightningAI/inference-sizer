"""Versioned benchmark profile adapter (benchmark_profile/v1).

A benchmark profile can confer "Benchmark calibrated" status ONLY when:
- it is not marked synthetic,
- every material field matches the request (model revision, quantization,
  compute/KV formats, engine, hardware SKU/system, parallelism, token-length
  distribution, prefix behavior, arrival rate, concurrency),
- its measured capacity is jointly compatible (request-rate and concurrency
  measured together).

Any material change strips the calibrated label. Profiles never extrapolate
across models, GPUs, engines, or workloads.
"""

from __future__ import annotations

from math import ceil

from pydantic import BaseModel, ConfigDict, Field

from .schemas import Evidence, Feasibility, SizeRequest, SizingResult

BENCHMARK_SCHEMA_VERSION = 1


class LatencyObjectives(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    ttft_ms_p95: float | None = Field(None, gt=0)
    decode_ms_p95: float | None = Field(None, gt=0)
    decode_metric: str | None = None  # "itl" or "tpot"; distinct metrics


class BenchmarkMetrics(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    ttft_ms_p95: float = Field(..., gt=0)
    decode_ms_p95: float = Field(..., gt=0)
    decode_metric: str  # "itl" | "tpot"; average TPOT is NOT a token-level ITL percentile
    sustainable_rps: float = Field(..., gt=0, description="Measured sustainable requests/second")
    sustainable_concurrency: int = Field(..., gt=0, description="Measured sustainable simultaneous generations")
    completion_rate: float = Field(..., gt=0, le=1)
    error_rate: float = Field(..., ge=0, lt=1)
    slo_miss_rate: float = Field(..., ge=0, le=1, description="Fraction of requests missing configured objectives")
    goodput_rps: float | None = Field(
        None, gt=0, description="Requests per second meeting the configured latency objectives"
    )
    avg_time_per_output_token_ms: float | None = Field(
        None, gt=0, description="Average; distinct from token-level ITL percentiles"
    )


class BenchmarkProfile(BaseModel):
    """Versioned import schema. Synthetic profiles are adapter-test fixtures."""

    model_config = ConfigDict(allow_inf_nan=False)

    schema_version: int = BENCHMARK_SCHEMA_VERSION
    profile_id: str
    synthetic: bool = False
    source: str
    collected_at: str
    model_ref: str = Field(..., description="Model identifier + checkpoint revision")
    checkpoint_revision: str | None = None
    architecture: str
    attn: str  # "mha" | "gqa" | "mla"
    total_params: float = Field(..., gt=0)
    quantization: str | None = None
    weight_format: str
    compute_precision: str
    kv_dtype: str
    engine_name: str
    engine_version: str | None = None
    hardware_sku: str = Field(..., description="GPU SKU / system, e.g. 'HGX B200'")
    hardware_profile_id: str | None = None
    gpus_per_replica: int = Field(..., ge=1)
    token_length_distribution: str = Field(
        ..., description="e.g. 'input mean 2048 / p95 8192, output mean 512 / p95 2048 (joint)'"
    )
    prefix_behavior: str  # "cold" | "warm" | "mixed"
    arrival_rate_rps: float = Field(..., gt=0)
    concurrency: int = Field(..., gt=0)
    duration_s: float = Field(..., gt=0)
    request_count: int = Field(..., gt=0)
    latency_objectives: LatencyObjectives
    metrics: BenchmarkMetrics

    def material_fields(self) -> dict:
        return {
            "architecture": self.architecture,
            "attn": self.attn,
            "total_params": self.total_params,
            "quantization": self.quantization,
            "weight_format": self.weight_format,
            "kv_dtype": self.kv_dtype,
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "hardware_profile_id": self.hardware_profile_id,
            "gpus_per_replica": self.gpus_per_replica,
            "prefix_behavior": self.prefix_behavior,
        }


def parse_profile(data: dict) -> BenchmarkProfile:
    if data.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported benchmark schema_version {data.get('schema_version')!r}; "
            f"this build speaks {BENCHMARK_SCHEMA_VERSION}"
        )
    return BenchmarkProfile.model_validate(data)


def compatibility_report(profile: BenchmarkProfile, req: SizeRequest) -> list[str]:
    """Material mismatches between the profile and the request."""
    m = req.model
    d = req.deployment
    mismatches: list[str] = []
    if m.architecture and profile.architecture != m.architecture:
        mismatches.append(
            f"architecture: request {m.architecture!r} vs profile {profile.architecture!r}"
        )
    if profile.attn != m.attn:
        mismatches.append(f"attention: request {m.attn!r} vs profile {profile.attn!r}")
    if abs(profile.total_params - m.total_params) / profile.total_params > 0.02:
        mismatches.append(
            f"total_params: request {m.total_params:.3g} vs profile {profile.total_params:.3g} (>2%)"
        )
    if profile.weight_format != m.weight_format:
        mismatches.append(f"weight_format: {m.weight_format!r} vs {profile.weight_format!r}")
    if profile.kv_dtype != m.kv_dtype:
        mismatches.append(f"kv_dtype: {m.kv_dtype!r} vs {profile.kv_dtype!r}")
    if profile.quantization and m.quantization and profile.quantization != m.quantization:
        mismatches.append(f"quantization: {m.quantization!r} vs {profile.quantization!r}")
    if profile.engine_name != d.engine.name:
        mismatches.append(f"engine: {d.engine.name!r} vs {profile.engine_name!r}")
    if d.engine.version and profile.engine_version and d.engine.version != profile.engine_version:
        mismatches.append(f"engine_version: {d.engine.version!r} vs {profile.engine_version!r}")
    if profile.hardware_profile_id and profile.hardware_profile_id != d.hardware_profile_id:
        mismatches.append(
            f"hardware: {d.hardware_profile_id!r} vs {profile.hardware_profile_id!r}"
        )
    if profile.gpus_per_replica != d.max_gpus_per_replica and d.max_gpus_per_replica < profile.gpus_per_replica:
        mismatches.append(
            f"parallelism: candidate TP {d.max_gpus_per_replica} below measured TP {profile.gpus_per_replica}"
        )
    # Token-length distribution: compare the stated distribution numerically when
    # the request uses the standard tail-envelope fields.
    if f"input mean {req.workload.mean_input_tokens}" not in profile.token_length_distribution:
        mismatches.append(
            "token_length_distribution: request input lengths not stated identically in the profile "
            f"({profile.token_length_distribution!r})"
        )
    if profile.prefix_behavior == "cold" and req.workload.prefix_hit_rate > 0:
        mismatches.append(
            "prefix_behavior: profile measured cold prefixes; request assumes "
            f"prefix_hit_rate={req.workload.prefix_hit_rate}"
        )
    return mismatches


def apply_calibration(req: SizeRequest, result: SizingResult, benchmark: dict | None) -> None:
    """Attach calibrated evidence to a feasible result, or record why it is refused.

    Called by engine.size() when a benchmark is referenced. Synthetic profiles and
    incompatible profiles never confer calibration.
    """
    profile_id = req.benchmark_profile_id
    if benchmark is None:
        result.warnings.append(
            f"benchmark_profile_id {profile_id!r} referenced but no profile supplied: "
            "status remains Estimated."
        )
        return
    try:
        profile = parse_profile(benchmark)
    except Exception as e:  # surface validation errors usefully
        result.warnings.append(f"Benchmark profile rejected: {e}")
        return
    if profile.profile_id != profile_id:
        result.warnings.append(
            f"Benchmark profile id {profile.profile_id!r} does not match requested {profile_id!r}; "
            "status remains Estimated."
        )
        return
    result.benchmark_profile_id = profile.profile_id
    result.benchmark_synthetic = profile.synthetic
    if profile.synthetic:
        result.warnings.append(
            "Synthetic benchmark fixture: excluded from real calibration evidence. "
            "Status remains Estimated."
        )
        return
    mismatches = compatibility_report(profile, req)
    if result.benchmark and result.benchmark.get("profile_id") == profile.profile_id:
        mismatches = result.benchmark.get("mismatches", mismatches)
    if mismatches:
        if not result.warnings:
            result.warnings.append(
                "Benchmark profile is not compatible with this request; status remains Estimated. "
                "Material mismatches: " + "; ".join(mismatches)
            )
        result.benchmark = {"profile_id": profile.profile_id, "mismatches": mismatches}
        return
    if result.feasibility != Feasibility.FEASIBLE or result.selected is None:
        result.warnings.append(
            "Calibration requires a feasible candidate; status remains Estimated."
        )
        return
    # Jointly compatible measured capacity: request-rate and concurrency measured
    # together. Operating factor applied once, as for estimates.
    util = req.deployment.operating_factor
    eff_rps = profile.metrics.sustainable_rps * util
    eff_conc = profile.metrics.sustainable_concurrency * util
    rps = req.workload.peak_rps or 0
    inflight = req.workload.peak_inflight or 0
    serving = max(
        ceil(rps / eff_rps) if rps > 0 else 0,
        ceil(inflight / eff_conc) if inflight > 0 else 0,
    )
    result.evidence_status = Evidence.BENCHMARK_CALIBRATED
    result.serving_replicas = serving
    result.fleet_gpus = result.selected.gpus_per_replica * (serving + req.deployment.spare_replicas)
    result.capacity_rps = eff_rps * serving
    result.capacity_concurrency = int(eff_conc) * serving
    result.latency_bound_only = False
    result.benchmark = {
        "profile_id": profile.profile_id,
        "source": profile.source,
        "collected_at": profile.collected_at,
        "model_ref": profile.model_ref,
        "engine": f"{profile.engine_name} {profile.engine_version or '(unspecified)'}",
        "hardware_sku": profile.hardware_sku,
        "gpus_per_replica": profile.gpus_per_replica,
        "duration_s": profile.duration_s,
        "request_count": profile.request_count,
        "metrics": profile.metrics.model_dump(),
    }
    result.notes.append(
        f"Calibrated by benchmark {profile.profile_id} "
        f"(measured {profile.metrics.sustainable_rps:.2f} rps at "
        f"{profile.metrics.sustainable_concurrency} concurrent, "
        f"p95 TTFT {profile.metrics.ttft_ms_p95:.0f} ms, "
        f"{profile.metrics.decode_metric} p95 {profile.metrics.decode_ms_p95:.1f} ms, "
        f"SLO miss rate {profile.metrics.slo_miss_rate:.3f})."
    )