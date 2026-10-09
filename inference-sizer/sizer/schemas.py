"""Versioned request/result schemas for the Inference Sizer.

All sizes are normalized to BYTES internally. Values are stored with their
source-published units and converted explicitly at the boundary.

schema_version: 2 (adds MLA / MoE / hybrid-attention KV paths; v1 payloads
are accepted on import and migrated — see migrate_request_payload)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 2

# Bytes per element by format name. Weight storage format, compute precision and
# KV dtype are tracked independently; nothing infers compute capability from
# these byte counts.
FORMAT_BYTES: dict[str, float] = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "fp4": 0.5}

WeightFormat = Literal["bf16", "fp16", "fp8", "fp4"]
KVDtype = Literal["bf16", "fp16", "fp8", "fp4"]
AttentionKind = Literal["mha", "gqa", "mla", "mqa", "hybrid"]
KVModel = Literal["mha_gqa", "mla", "override"]


class NonFiniteAwareModel(BaseModel):
    """Rejects NaN/Inf anywhere (Pydantic floats allow them by default)."""

    model_config = ConfigDict(allow_inf_nan=False)


class ModelSpec(NonFiniteAwareModel):
    """Normalized model spec covering dense MHA/GQA, MLA, MoE and hybrid
    attention via three KV-cache paths (kv_model):

    - mha_gqa: 2 x kv_layers x kv_heads x head_dim x bytes
    - mla:     kv_layers x (kv_lora_rank + qk_rope_head_dim) x bytes, replicated
    - override: vendor/config-published per-token figure (hybrid CSA, DeltaNet,
      IndexPool-style caches), must cite its source
    """

    source: Literal["preset", "config_json", "manual"] = "manual"
    preset_id: str | None = None
    architecture: str | None = Field(
        None, description="HF architectures[0], e.g. LlamaForCausalLM"
    )
    attn: AttentionKind = "gqa"
    kv_model: KVModel = "mha_gqa"
    layers: int = Field(..., gt=0, description="num_hidden_layers")
    attention_kv_layers: int | None = Field(
        None, gt=0,
        description="Layers whose KV cache grows with context (full-attention / MLA "
        "layers). None = all layers (dense attention).",
    )
    kv_heads: int = Field(..., gt=0, description="num_key_value_heads (== heads for MHA)")
    head_dim: int = Field(..., gt=0)
    kv_lora_rank: int | None = Field(None, gt=0, description="MLA compressed KV latent rank")
    qk_rope_head_dim: int | None = Field(None, ge=0, description="MLA decoupled RoPE channel")
    kv_bytes_per_token_override: int | None = Field(
        None, gt=0,
        description="Published/derived KV bytes per token for cache layouts the formulas "
        "do not cover (hybrid sparse attention, DeltaNet hybrids). Requires a source.",
    )
    kv_override_source: str | None = Field(
        None, description="Where the KV override figure comes from; required with override"
    )
    kv_fixed_bytes_per_sequence: int | None = Field(
        None, ge=0,
        description="Per-sequence constant KV-side memory: sliding-window caps, linear/"
        "DeltaNet recurrent state, compressed sparse pools. Derived figures carry notes.",
    )
    num_experts: int | None = Field(None, ge=1, description="Routed experts (MoE)")
    active_experts: int | None = Field(None, ge=1, description="Experts active per token")
    attention_heads: int | None = Field(None, gt=0, description="num_attention_heads")
    hidden_size: int | None = Field(None, gt=0)
    total_params: float = Field(..., gt=0, description="Total resident parameters (not billions)")
    active_params: float | None = Field(
        None, gt=0, description="Active params per token; defaults to total for dense models"
    )
    max_context_tokens: int | None = Field(None, gt=0, description="Hard context limit")
    checkpoint_revision: str | None = None
    weight_format: WeightFormat = "bf16"
    quantization: str | None = Field(
        None, description="Quant scheme metadata, e.g. 'fp8-e5m2-weights-only'"
    )
    kv_dtype: KVDtype = "bf16"

    @model_validator(mode="after")
    def _check_heads(self) -> "ModelSpec":
        if self.attn in ("mha", "gqa") and (
            self.attention_heads is not None and self.kv_heads > self.attention_heads
        ):
            raise ValueError("kv_heads cannot exceed attention_heads for MHA/GQA")
        return self

    @model_validator(mode="after")
    def _check_kv_model(self) -> "ModelSpec":
        if self.kv_model == "mla":
            if self.kv_lora_rank is None:
                raise ValueError("kv_model=mla requires kv_lora_rank")
            if self.qk_rope_head_dim is None:
                object.__setattr__(self, "qk_rope_head_dim", 0)
        if self.kv_model == "override":
            if self.kv_bytes_per_token_override is None:
                raise ValueError("kv_model=override requires kv_bytes_per_token_override")
            if not self.kv_override_source:
                raise ValueError("kv_model=override requires kv_override_source (provenance)")
        if self.attention_kv_layers is not None and self.attention_kv_layers > self.layers:
            raise ValueError("attention_kv_layers cannot exceed layers")
        return self

    @model_validator(mode="after")
    def _check_active(self) -> "ModelSpec":
        if self.active_params is None:
            object.__setattr__(self, "active_params", self.total_params)
        return self

    def weight_bytes_each(self) -> float:
        return FORMAT_BYTES[self.weight_format]

    def effective_kv_layers(self) -> int:
        """Layers whose KV cache grows with context (full/MLA layers)."""
        return self.attention_kv_layers if self.attention_kv_layers is not None else self.layers


def migrate_request_payload(payload: dict) -> dict:
    """Accept v1 request payloads and upgrade them to the current schema.

    v1 -> v2: derive kv_model from attn (mla -> mla, else mha_gqa). Unknown
    versions still raise in the SizeRequest validator.
    """
    if not isinstance(payload, dict):
        return payload
    v = payload.get("schema_version", SCHEMA_VERSION)
    if v == SCHEMA_VERSION:
        return payload
    if v == 1:
        out = dict(payload)
        model = dict(out.get("model") or {})
        attn = model.get("attn", "gqa")
        model.setdefault("kv_model", "mla" if attn == "mla" else "mha_gqa")
        out["model"] = model
        out["schema_version"] = SCHEMA_VERSION
        return out
    return payload  # let the validator produce the real error


class WorkloadSpec(NonFiniteAwareModel):
    """Traffic and latency targets. Zero traffic is legitimate; zero capacity is not."""

    mean_input_tokens: int = Field(..., ge=0)
    p95_input_tokens: int = Field(..., ge=0)
    mean_output_tokens: int = Field(..., ge=0)
    p95_output_tokens: int = Field(..., ge=0)
    peak_rps: float = Field(
        0, ge=0, description="Peak requests per second. 0 is legitimate (no traffic)."
    )
    peak_inflight: int = Field(0, ge=0, description="Peak simultaneous generations. 0 = unspecified.")
    ttft_ms_p95: float = Field(..., gt=0)
    decode_ms_p95: float = Field(
        ..., gt=0, description="p95 decode latency objective (ITL or TPOT, stated by the caller)"
    )
    decode_metric: Literal["itl", "tpot"] = "itl"
    prefix_hit_rate: float = Field(0.0, ge=0, le=1)
    # Active-user cadence inputs (optional alternative to peak_rps). Explicit conversion:
    # peak_rps = active_users * requests_per_active_per_hour * calls_per_task / 3600
    active_users: int | None = Field(None, ge=0)
    requests_per_active_per_hour: float | None = Field(None, gt=0)
    calls_per_task: float | None = Field(None, gt=0, description="Model requests per user task")

    @model_validator(mode="after")
    def _tails(self) -> "WorkloadSpec":
        if self.p95_input_tokens < self.mean_input_tokens:
            raise ValueError("p95_input_tokens must be >= mean_input_tokens")
        if self.p95_output_tokens < self.mean_output_tokens:
            raise ValueError("p95_output_tokens must be >= mean_output_tokens")
        return self

    def derived_rps(self) -> float | None:
        """RPS from active-user cadence, or None when cadence inputs are absent."""
        if self.active_users is None or self.requests_per_active_per_hour is None:
            return None
        calls = self.calls_per_task if self.calls_per_task is not None else 1.0
        return self.active_users * self.requests_per_active_per_hour * calls / 3600.0


class EngineSpec(NonFiniteAwareModel):
    name: str = "vllm"
    version: str | None = None


class DeploymentSpec(NonFiniteAwareModel):
    hardware_profile_id: str
    engine: EngineSpec = EngineSpec()
    max_gpus_per_replica: int = Field(..., ge=1, description="1+; engine/deployment limit")
    spare_replicas: int = Field(0, ge=0)
    operating_factor: float = Field(
        0.65, gt=0, le=1,
        description="Fraction of measured/estimated capacity counted for sizing. "
        "Applied once to capacity, before replica rounding.",
    )
    observed_memory_gb: float | None = Field(
        None, gt=0,
        description="Measured per-device allocatable memory; overrides the published nominal value.",
    )
    allocation_size_gpus: int | None = Field(
        None, gt=0,
        description="Provider allocation granularity; kept distinct from GPUs per replica.",
    )
    bw_eff: float = Field(0.65, gt=0, le=1, description="Bandwidth efficiency (optimistic bound)")
    flops_eff: float = Field(0.45, gt=0, le=1, description="Compute efficiency (optimistic bound)")
    runtime_overhead_gb: float = Field(2.0, ge=0, description="Engine runtime, per device")
    cuda_graphs_gb: float = Field(1.5, ge=0, description="CUDA graph capture, per device")
    activations_workspace_gb: float = Field(
        2.0, ge=0, description="Activation/workspace estimate, per device (user-adjustable)"
    )
    reserve_fraction: float = Field(0.05, ge=0, le=0.5, description="Safety reserve of device memory")


class SizeRequest(NonFiniteAwareModel):
    """Versioned sizing request."""

    schema_version: int = SCHEMA_VERSION
    customer: str = Field("Customer", max_length=200)
    tier: str = Field("Interactive chat", max_length=200)
    model: ModelSpec
    workload: WorkloadSpec
    deployment: DeploymentSpec
    benchmark_profile_id: str | None = Field(
        None, description="Optional benchmark profile to calibrate this request"
    )
    # Optional inline benchmark_profile/v1 payload for API transport. The engine
    # falls back to this when no separate benchmark argument is passed.
    benchmark: dict | None = None

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {v}; this build speaks {SCHEMA_VERSION}")
        return v


class SourceRef(BaseModel):
    url: str | None = None
    file: str | None = None
    page: str | None = None
    publication: str | None = None
    verified_on: str = "2026-10-07"
    access: Literal["public", "internal"] = "public"


class HardwareStatus(BaseModel):
    """Four independent statuses; unknown inventory never defaults to available."""

    spec_confidence: Literal["published", "inferred", "prototype_anchor"] = "published"
    inventory: Literal["available", "unknown", "unavailable"] = "unknown"
    engine_qualified: bool = False
    benchmark_evidence: Literal["none", "pending", "available"] = "none"
    notes: list[str] = []


class HardwareProfile(BaseModel):
    """Catalog hardware profile. Published values are anchors, not observations."""

    id: str
    name: str
    gpu_sku: str
    system_family: str = Field(
        ..., description="DGX = NVIDIA server family; HGX = OEM-integrated platform"
    )
    device_count: int | None = Field(..., ge=1, description="Deployable devices (dies are separate)")
    device_count_note: str | None = None
    die_count_per_device: int | None = None
    die_count_note: str | None = None
    memory_per_device_gb: float = Field(..., gt=0)
    memory_qualifier: Literal["nominal", "up_to"] = "nominal"
    observed_memory_bytes: int | None = None
    hbm_bandwidth_tb_s: float = Field(..., gt=0)
    bandwidth_qualifier: Literal["nominal", "up_to"] = "nominal"
    # Bandwidths stored separately with direction and aggregation scope. Host
    # memory is NOT GPU KV capacity without a supported offload model.
    scale_up_bandwidth_tb_s: float | None = Field(
        None, description="NVLink-class scale-up, aggregated domain scope"
    )
    scale_out_bandwidth_tb_s: float | None = None
    cpu_gpu_bandwidth_tb_s: float | None = None
    topology: str | None = None
    allocation_sizes_gpus: list[int] | None = Field(
        None, description="Provider allocation granularity; unknown stays null"
    )
    compute_tflops: dict[str, float] | None = Field(
        None,
        description="Dense TFLOP/s by compute precision (bf16/fp8/fp4). Sparse values are NOT stored here.",
    )
    compute_note: str | None = None
    recommendation_mode: Literal["recommended", "comparison_only"] = "recommended"
    source_refs: dict[str, SourceRef]
    status: HardwareStatus


class ModelPreset(BaseModel):
    id: str
    name: str
    family: str = Field(..., description="Family label for UI grouping, e.g. 'Qwen'")
    architecture: str
    attn: AttentionKind = "gqa"
    kv_model: KVModel = "mha_gqa"
    layers: int
    attention_kv_layers: int | None = None
    kv_heads: int
    head_dim: int
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    kv_bytes_per_token_override: int | None = None
    kv_override_source: str | None = None
    kv_fixed_bytes_per_sequence: int | None = None
    num_experts: int | None = None
    active_experts: int | None = None
    attention_heads: int
    hidden_size: int
    total_params: float
    active_params: float | None = None
    max_context_tokens: int | None = None
    weight_format: WeightFormat = "bf16"
    spec_confidence: Literal["published", "unverified"] = "published"
    unsupported_reason: str | None = Field(
        None, description="Set when the preset is outside supported recommendations"
    )
    source_ref: SourceRef | None = None
    notes: list[str] = []


class Catalog(BaseModel):
    catalog_version: int = 3  # matches sizer/data/hardware.json; load_catalog always supplies it
    presets: list[ModelPreset]
    hardware_profiles: list[HardwareProfile]
    engines: list[dict]


# ---------------------------------------------------------------- result side

class Evidence(StrEnum):
    ESTIMATED = "estimated"
    BENCHMARK_CALIBRATED = "benchmark_calibrated"


class Feasibility(StrEnum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"


class MemoryComponents(BaseModel):
    """Separate visible terms, per replica, in bytes."""

    weights_bytes: float
    kv_p95_bytes_per_replica: float
    kv_fixed_bytes_per_replica: float = Field(
        0.0, description="Per-sequence constant KV-side memory (sliding-window caps, "
        "linear-attention state), x batch, per replica"
    )
    runtime_overhead_bytes: float
    cuda_graphs_bytes: float
    activations_workspace_bytes: float
    reserve_bytes: float
    total_bytes_per_replica: float
    total_bytes_per_device: float
    device_memory_bytes: float
    device_memory_source: Literal["observed", "published"]


class CandidateRecord(BaseModel):
    gpus_per_replica: int
    feasible: bool
    rejection_reasons: list[str] = []
    memory: MemoryComponents | None = None
    kv_per_token_bytes: int
    kv_sharding: Literal["sharded", "replicated"] | None = None
    kv_heads_per_device: int | None = None
    max_concurrency: int | None = Field(None, description="Batch inside the ITL bound and memory fit")
    prefill_bound_ms: float | None = None
    itl_bound_ms: float | None = None
    rps_capacity: float | None = None
    serving_replicas: int | None = None
    limit: Literal["memory", "itl", "ttft", "prefill_throughput", "concurrency", "request_rate", "none"] | None = None


class SizingResult(BaseModel):
    """Versioned, explainable result. Evidence and feasibility are independent."""

    schema_version: int = SCHEMA_VERSION
    calculator_version: str = "1.1.0"
    customer: str
    tier: str
    request: SizeRequest
    evidence_status: Evidence = Evidence.ESTIMATED
    architecture_status: Literal["supported", "unsupported"] = "supported"
    feasibility: Feasibility = Feasibility.INFEASIBLE
    benchmark_profile_id: str | None = None
    benchmark_synthetic: bool | None = None
    hardware_profile: HardwareProfile | None = None
    model_provenance: dict = {}
    memory_components: MemoryComponents | None = None
    selected: CandidateRecord | None = None
    serving_replicas: int | None = None
    spare_replicas: int = 0
    fleet_gpus: int | None = None
    allocation_size_gpus: int | None = None
    capacity_rps: float | None = None
    capacity_concurrency: int | None = None
    latency_bound_only: bool = True
    limiting_factors: list[str] = []
    rejection_reasons: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    assumptions: list[str] = []
    traffic_derived: dict = {}
    benchmark: dict | None = None