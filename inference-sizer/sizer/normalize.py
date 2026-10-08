"""Model normalization with provenance.

Fields carry their source: preset, config.json, manual, or inferred. Missing
values are surfaced, never silently defaulted where they would change results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .hardware import get_preset, load_catalog
from .schemas import ModelSpec, SourceRef


class ProvenanceEntry(BaseModel):
    field: str
    value: str | int | float | None
    source: str  # preset | config_json | manual | inferred
    note: str | None = None


class NormalizedModel(BaseModel):
    spec: ModelSpec
    provenance: list[ProvenanceEntry]
    missing: list[str]
    unsupported_reason: str | None = None


# Dense MHA/GQA architectures: fully first-principles (kv_model=mha_gqa).
SUPPORTED_ARCH_PREFIXES = (
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "MistralForCausalLM",
    "Phi3ForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Glm4ForCausalLM",
)

# MLA architectures: one compressed latent + RoPE channel per token per layer
# (kv_model=mla). Cache is replicated across TP, matching serving-engine layouts.
MLA_ARCHITECTURES = (
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV2ForCausalMoeModel",
    "DeepseekV3ForCausalMoeModel",
    "GlmMoeDsaForCausalLM",
    "KimiK2ForCausalLM",
    "KimiK25ForCausalLM",
    "KimiK2_5ForCausalLM",
)

# Hybrid MLA: MLA on a subset of layers + linear/KDA state layers. Supported
# only when the caller supplies the per-sequence constant state term.
MLA_HYBRID_ARCHITECTURES = (
    "KimiK3ForConditionalGeneration",
    "Glm5NextForConditionalGeneration",
)

# Hybrid GQA/sparse: full-attention subset + sliding-window/linear/sparse-pool
# layers. Supported only with explicit attention_kv_layers plus a per-sequence
# fixed term or a KV override (deepseek_v4 CSA/HCA lands here too).
HYBRID_ARCHITECTURES = (
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "InklingForConditionalGeneration",
    "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForConditionalGeneration",
)

MLA_STATE_KEYS = ("kv_lora_rank", "qk_rope_head_dim")
HYBRID_DOC = "docs/model-catalog-notes.md"


def _classify(architecture: str | None) -> str:
    """Returns one of: dense | mla | mla_hybrid | hybrid | unknown."""
    if not architecture:
        return "unknown"
    for p in MLA_HYBRID_ARCHITECTURES:
        if architecture.startswith(p):
            return "mla_hybrid"
    for p in HYBRID_ARCHITECTURES:
        if architecture.startswith(p):
            return "hybrid"
    for p in MLA_ARCHITECTURES:
        if architecture.startswith(p):
            return "mla"
    if any(architecture.startswith(p) for p in SUPPORTED_ARCH_PREFIXES):
        return "dense"
    return "unknown"


def from_preset(preset_id: str) -> NormalizedModel:
    preset = get_preset(preset_id)
    unsupported = getattr(preset, "unsupported_reason", None)
    spec = ModelSpec(
        source="preset",
        preset_id=preset.id,
        architecture=preset.architecture,
        attn=preset.attn,
        kv_model=preset.kv_model,
        layers=preset.layers,
        attention_kv_layers=preset.attention_kv_layers,
        kv_heads=preset.kv_heads,
        head_dim=preset.head_dim,
        kv_lora_rank=preset.kv_lora_rank,
        qk_rope_head_dim=preset.qk_rope_head_dim,
        kv_bytes_per_token_override=preset.kv_bytes_per_token_override,
        kv_override_source=preset.kv_override_source,
        kv_fixed_bytes_per_sequence=preset.kv_fixed_bytes_per_sequence,
        num_experts=preset.num_experts,
        active_experts=preset.active_experts,
        attention_heads=preset.attention_heads,
        hidden_size=preset.hidden_size,
        total_params=preset.total_params,
        active_params=preset.active_params,
        max_context_tokens=preset.max_context_tokens,
        weight_format=preset.weight_format,
        checkpoint_revision=None,
    )
    prov = [
        ProvenanceEntry(field=f, value=getattr(spec, f), source="preset")
        for f in (
            "architecture", "attn", "kv_model", "layers", "attention_kv_layers",
            "kv_heads", "head_dim", "kv_lora_rank", "qk_rope_head_dim",
            "kv_bytes_per_token_override", "kv_fixed_bytes_per_sequence",
            "num_experts", "active_experts", "attention_heads", "hidden_size",
            "total_params", "active_params", "max_context_tokens", "weight_format",
        )
        if getattr(spec, f) is not None
    ]
    missing = [] if spec.total_params else ["total_params"]
    return NormalizedModel(
        spec=spec, provenance=prov, missing=missing, unsupported_reason=unsupported
    )


def from_config_json(config: dict, overrides: dict | None = None) -> NormalizedModel:
    """Normalize a HuggingFace config.json. Safe: unknown/missing fields surface.

    A config alone cannot establish total parameters or the quantized weight
    footprint; total_params must come from an override or stays surfaced as missing.
    Nested `text_config` (multimodal wrappers: GLM-5.x, Kimi, Inkling, Gemma 4,
    Qwen3.5+) is searched after top-level keys.
    """
    overrides = overrides or {}
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    arch_list = config.get("architectures") or []
    architecture = arch_list[0] if arch_list else overrides.get("architecture")
    provenance: list[ProvenanceEntry] = []

    def take(field: str, *keys: str, cast=None):
        """Take first present key from overrides, then config, then text_config."""
        if field in overrides and overrides[field] is not None:
            v = overrides[field]
            provenance.append(ProvenanceEntry(field=field, value=v, source="manual"))
            return cast(v) if cast else v
        for k in keys:
            for src in (config, text):
                if k in src and src[k] is not None:
                    v = src[k]
                    provenance.append(ProvenanceEntry(field=field, value=v, source="config_json"))
                    return cast(v) if cast else v
        provenance.append(ProvenanceEntry(field=field, value=None, source="inferred", note="absent"))
        return None

    layers = take("layers", "num_hidden_layers", cast=int)
    kv_heads = take("kv_heads", "num_key_value_heads", cast=int)
    head_dim = take("head_dim", "head_dim", cast=int)
    attention_heads = take("attention_heads", "num_attention_heads", cast=int)
    hidden_size = take("hidden_size", "hidden_size", cast=int)
    max_context = take(
        "max_context_tokens", "max_position_embeddings", "model_max_length", cast=int
    )

    missing = []
    if head_dim is None and hidden_size and attention_heads:
        head_dim = hidden_size // attention_heads
        provenance.append(
            ProvenanceEntry(field="head_dim", value=head_dim, source="inferred",
                            note="hidden_size // num_attention_heads")
        )
    if attention_heads is None and hidden_size and head_dim:
        attention_heads = hidden_size // head_dim
        provenance.append(
            ProvenanceEntry(field="attention_heads", value=attention_heads, source="inferred",
                            note="hidden_size // head_dim")
        )

    klass = _classify(architecture)
    kv_lora_rank = take("kv_lora_rank", "kv_lora_rank", cast=int) if klass in ("mla", "mla_hybrid") else None
    qk_rope_head_dim = (
        take("qk_rope_head_dim", "qk_rope_head_dim", cast=int)
        if klass in ("mla", "mla_hybrid") else None
    )

    # Layers carrying a GROWING KV cache: count full-attention entries in
    # layer_types when the config publishes the schedule. Sliding-window /
    # linear layers contribute only the per-sequence constant term.
    layer_types = None
    for src in (overrides, config, text):
        if isinstance(src.get("layer_types"), list):
            layer_types = src["layer_types"]
            break
    attention_kv_layers = None
    if layer_types is not None and klass in ("hybrid", "mla_hybrid"):
        attention_kv_layers = sum(1 for t in layer_types if str(t).startswith("full"))
        provenance.append(
            ProvenanceEntry(field="attention_kv_layers", value=attention_kv_layers,
                            source="config_json", note="count of full-attention layers in layer_types")
        )

    # MoE fields are extracted, never unsupported by themselves.
    num_experts = take("num_experts", "num_experts", "n_routed_experts", cast=int)
    active_experts = take("active_experts", "num_experts_per_tok", cast=int)

    attn_label = {"mla": "mla", "mla_hybrid": "hybrid", "hybrid": "hybrid"}.get(klass)
    attn = attn_label or ("gqa" if (kv_heads and attention_heads and kv_heads < attention_heads) else "mha")
    if kv_heads is None:
        missing.append("kv_heads")
    if layers is None:
        missing.append("layers")
    if head_dim is None:
        missing.append("head_dim")

    total_params = overrides.get("total_params")
    if total_params is not None:
        provenance.append(
            ProvenanceEntry(field="total_params", value=total_params, source="manual",
                            note="config.json alone cannot establish total parameters")
        )
    else:
        missing.append("total_params (config.json cannot establish this; manual override required)")
    active_params = overrides.get("active_params")

    weight_format = overrides.get("weight_format", "bf16")
    kv_dtype = overrides.get("kv_dtype", "bf16")
    quantization = config.get("quantization_config", {}).get("quant_method") if isinstance(
        config.get("quantization_config"), dict
    ) else overrides.get("quantization")
    if quantization:
        provenance.append(ProvenanceEntry(field="quantization", value=quantization, source="config_json"))
    if weight_format in ("fp8", "fp4") and not quantization:
        quantization = quantization or "unspecified (weight_format override)"

    kv_bytes_per_token_override = overrides.get("kv_bytes_per_token_override")
    kv_override_source = overrides.get("kv_override_source")
    kv_fixed = overrides.get("kv_fixed_bytes_per_sequence")

    # ---- support classification (MoE is NOT unsupported; missing hybrid
    # constants are, because guessing them would silently change results)
    unsupported = None
    if klass == "mla":
        if kv_lora_rank is None:
            unsupported = (
                f"MLA architecture {architecture}: config carries no kv_lora_rank; "
                "the compressed-cache formula needs it"
            )
    elif klass in ("hybrid", "mla_hybrid"):
        if kv_bytes_per_token_override is None and (attention_kv_layers is None or kv_fixed is None):
            unsupported = (
                f"Hybrid architecture {architecture}: only the full-attention layers carry a "
                "growing KV cache and the remaining layers hold constant state/sliding-window "
                f"caches. Supply attention_kv_layers + kv_fixed_bytes_per_sequence (or a "
                f"kv_bytes_per_token_override with its source); derivations and per-model "
                f"figures are documented in {HYBRID_DOC}."
            )
    elif klass == "unknown":
        unsupported = f"Architecture {architecture} is not in the supported set; see {HYBRID_DOC}"

    kv_model = "mha_gqa"
    if klass == "mla" and kv_lora_rank is not None:
        kv_model = "mla"
    elif kv_bytes_per_token_override is not None:
        kv_model = "override"

    spec_kwargs = dict(
        source="config_json",
        preset_id=None,
        architecture=architecture,
        attn=attn,
        kv_model=kv_model,
        kv_heads=kv_heads or 1,
        head_dim=head_dim or 1,
        layers=layers or 1,
        attention_kv_layers=attention_kv_layers,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_bytes_per_token_override=kv_bytes_per_token_override,
        kv_override_source=kv_override_source,
        kv_fixed_bytes_per_sequence=kv_fixed,
        num_experts=num_experts,
        active_experts=active_experts,
        attention_heads=attention_heads,
        hidden_size=hidden_size,
        total_params=total_params or 1,  # placeholder; invalid until overridden (gt=0)
        active_params=active_params,
        max_context_tokens=max_context,
        weight_format=weight_format,
        kv_dtype=kv_dtype,
        quantization=quantization,
        checkpoint_revision=overrides.get("checkpoint_revision"),
    )
    # If still missing core fields, do not fabricate; raise so the caller surfaces them.
    problems = [m for m in missing if m.startswith(("layers", "kv_heads", "head_dim"))]
    if problems:
        raise ValueError("config.json missing required architecture fields: " + ", ".join(problems))
    if total_params is None:
        raise ValueError("total_params is required: config.json cannot establish total parameters; "
                         "supply a manual override")
    return NormalizedModel(spec=ModelSpec(**spec_kwargs), provenance=provenance, missing=missing,
                           unsupported_reason=unsupported)


def from_manual(**kwargs) -> NormalizedModel:
    spec = ModelSpec(source="manual", **kwargs)
    prov = [
        ProvenanceEntry(field=f, value=getattr(spec, f), source="manual")
        for f in ("architecture", "attn", "kv_model", "layers", "attention_kv_layers",
                  "kv_heads", "head_dim", "kv_lora_rank", "qk_rope_head_dim",
                  "kv_bytes_per_token_override", "kv_fixed_bytes_per_sequence",
                  "num_experts", "active_experts", "attention_heads", "hidden_size",
                  "total_params", "active_params")
        if getattr(spec, f) is not None
    ]
    return NormalizedModel(spec=spec, provenance=prov, missing=[])


def preset_list() -> list[dict]:
    """Presets for the catalog, with support status."""
    out = []
    for p in load_catalog().presets:
        d = p.model_dump()
        d["unsupported_reason"] = getattr(p, "unsupported_reason", None)
        out.append(d)
    return out