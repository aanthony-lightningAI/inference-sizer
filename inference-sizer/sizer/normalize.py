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


# Architectures the supported MHA/GQA path covers. Anything else (MLA, MoE,
# multimodal, hybrid) is Unsupported, not silently force-fitted.
SUPPORTED_ARCH_PREFIXES = (
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "MistralForCausalLM",
    "Phi3ForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
)

MLA_ARCHITECTURES = (
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV2ForCausalMoeModel",
    "DeepseekV3ForCausalMoeModel",
)


def from_preset(preset_id: str) -> NormalizedModel:
    preset = get_preset(preset_id)
    unsupported = getattr(preset, "unsupported_reason", None)
    spec = ModelSpec(
        source="preset",
        preset_id=preset.id,
        architecture=preset.architecture,
        attn=preset.attn,
        layers=preset.layers,
        kv_heads=preset.kv_heads,
        head_dim=preset.head_dim,
        attention_heads=preset.attention_heads,
        hidden_size=preset.hidden_size,
        total_params=preset.total_params,
        max_context_tokens=preset.max_context_tokens,
        checkpoint_revision=None,
    )
    prov = [
        ProvenanceEntry(field=f, value=getattr(spec, f), source="preset")
        for f in (
            "architecture", "attn", "layers", "kv_heads", "head_dim",
            "attention_heads", "hidden_size", "total_params", "max_context_tokens",
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
    """
    overrides = overrides or {}
    arch_list = config.get("architectures") or []
    architecture = arch_list[0] if arch_list else overrides.get("architecture")
    provenance: list[ProvenanceEntry] = []

    def take(field: str, *keys: str, cast=None):
        """Take first present key from config, else override, else None."""
        for k in keys:
            if k in config and config[k] is not None:
                v = config[k]
                provenance.append(ProvenanceEntry(field=field, value=v, source="config_json"))
                return cast(v) if cast else v
        if field in overrides and overrides[field] is not None:
            v = overrides[field]
            provenance.append(ProvenanceEntry(field=field, value=v, source="manual"))
            return cast(v) if cast else v
        provenance.append(ProvenanceEntry(field=field, value=None, source="inferred", note="absent"))
        return None

    layers = take("layers", "num_hidden_layers", cast=int)
    kv_heads = take("kv_heads", "num_key_value_heads", cast=int)
    head_dim = take("head_dim", "head_dim", cast=int)
    attention_heads = take("attention_heads", "num_attention_heads", cast=int)
    hidden_size = take("hidden_size", "hidden_size", cast=int)
    max_context = take("max_context_tokens", "max_position_embeddings", cast=int)

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
    attn = "gqa" if (kv_heads and attention_heads and kv_heads < attention_heads) else "mha"
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

    weight_format = overrides.get("weight_format", "bf16")
    kv_dtype = overrides.get("kv_dtype", "bf16")
    quantization = config.get("quantization_config", {}).get("quant_method") if isinstance(
        config.get("quantization_config"), dict
    ) else overrides.get("quantization")
    if quantization:
        provenance.append(ProvenanceEntry(field="quantization", value=quantization, source="config_json"))
    if weight_format in ("fp8", "fp4") and not quantization:
        quantization = quantization or "unspecified (weight_format override)"

    unsupported = None
    if architecture and any(architecture.startswith(a) for a in MLA_ARCHITECTURES):
        unsupported = "MLA architecture: KV memory formula for MHA/GQA does not apply; unsupported in v1"
    elif architecture and not any(architecture.startswith(a) for a in SUPPORTED_ARCH_PREFIXES):
        unsupported = f"Architecture {architecture} is not in the supported dense MHA/GQA set for v1"
    if "moe" in str(config.get("model_type", "")).lower() or any(
        k in config for k in ("num_experts", "n_routed_experts", "num_experts_per_tok")
    ):
        unsupported = unsupported or "MoE model: deferred in v1"

    spec_kwargs = dict(
        source="config_json",
        preset_id=None,
        architecture=architecture,
        attn=attn,
        kv_heads=kv_heads or 1,
        head_dim=head_dim or 1,
        layers=layers or 1,
        attention_heads=attention_heads,
        hidden_size=hidden_size,
        total_params=total_params or 1,  # placeholder; invalid until overridden (gt=0)
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
        for f in ("architecture", "attn", "layers", "kv_heads", "head_dim",
                  "attention_heads", "hidden_size", "total_params")
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