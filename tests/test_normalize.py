"""config.json normalization (server-side path used by tests; the UI mirrors it)."""

import pytest

from sizer.normalize import from_config_json, from_preset
from sizer.schemas import ModelSpec


def llama_config(**over):
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_attention_heads": 32,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position_embeddings": 131072,
    }
    cfg.update(over)
    return cfg


def test_config_json_normalize_with_overrides():
    nm = from_config_json(llama_config(), overrides={"total_params": 8.03e9})
    spec = nm.spec
    assert spec.source == "config_json"
    assert spec.architecture == "LlamaForCausalLM"
    assert spec.layers == 32 and spec.kv_heads == 8 and spec.head_dim == 128
    assert spec.attn == "gqa"  # 8 kv heads < 32 attention heads
    assert spec.total_params == 8.03e9
    sources = {p.field: p.source for p in nm.provenance}
    assert sources["layers"] == "config_json"
    assert sources["total_params"] == "manual"
    assert not any("total_params" in m for m in nm.missing)


def test_config_json_cannot_establish_params():
    with pytest.raises(ValueError, match="total_params"):
        from_config_json(llama_config())


def test_config_json_infers_head_dim():
    cfg = llama_config()
    del cfg["head_dim"]
    nm = from_config_json(cfg, overrides={"total_params": 1e9})
    assert nm.spec.head_dim == 4096 // 32
    inferred = [p for p in nm.provenance if p.source == "inferred"]
    assert any(p.field == "head_dim" for p in inferred)


def test_config_json_mla_unsupported():
    cfg = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "num_attention_heads": 128,
        "head_dim": 128,
    }
    nm = from_config_json(cfg, overrides={"total_params": 6.71e11})
    assert nm.unsupported_reason and "MLA" in nm.unsupported_reason


def test_config_json_moe_unsupported():
    cfg = llama_config(num_experts=8, num_experts_per_tok=2)
    nm = from_config_json(cfg, overrides={"total_params": 1e9})
    assert nm.unsupported_reason and "MoE" in nm.unsupported_reason


def test_config_json_missing_fields_surface():
    cfg = llama_config()
    del cfg["num_key_value_heads"]
    with pytest.raises(ValueError, match="kv_heads"):
        from_config_json(cfg, overrides={"total_params": 1e9})


def test_preset_mla_flagged_unsupported():
    nm = from_preset("deepseek_v3_mla")
    assert nm.unsupported_reason and "MLA" in nm.unsupported_reason


def test_quantization_config_detected():
    cfg = llama_config(quantization_config={"quant_method": "fp8"})
    nm = from_config_json(cfg, overrides={"total_params": 1e9})
    assert nm.spec.quantization == "fp8"
    assert nm.spec.weight_format == "bf16"  # only explicit fp8 strings switch format


def test_spec_validates_after_normalize():
    nm = from_config_json(llama_config(), overrides={"total_params": 8.03e9})
    ModelSpec.model_validate(nm.spec.model_dump())  # round-trips cleanly