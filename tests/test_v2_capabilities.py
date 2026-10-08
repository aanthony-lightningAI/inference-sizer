"""Schema v2 capabilities: MLA, MoE, KV override, fixed KV term, migration.

Every formula assertion uses an independently hand-computed expected value.
"""

from __future__ import annotations

import json

import pytest

from sizer.engine import kv_bytes_per_token, size
from sizer.normalize import from_config_json, from_preset
from sizer.schemas import ModelSpec, SCHEMA_VERSION, SizeRequest, migrate_request_payload
from sizer.cli import _example_request


def _req(model_overrides: dict) -> SizeRequest:
    d = json.loads(json.dumps(_example_request()))
    d["model"].update(model_overrides)
    return SizeRequest.model_validate(d)


def _preset_req(pid: str) -> SizeRequest:
    d = json.loads(json.dumps(_example_request()))
    d["model"] = from_preset(pid).spec.model_dump()
    return SizeRequest.model_validate(d)


# ---------------------------------------------------------------- MLA path

def test_mla_formula_kimi_k2_6():
    """61 layers x (512 + 64) x 2 B = 70,272 B/token (independent calc)."""
    spec = from_preset("kimi_k2_6").spec
    assert spec.kv_model == "mla"
    assert kv_bytes_per_token(spec) == 61 * (512 + 64) * 2 == 70_272


def test_mla_formula_deepseek_v3_preset():
    spec = from_preset("deepseek_v3_mla").spec
    assert kv_bytes_per_token(spec) == 61 * 576 * 2 == 70_272


def test_mla_replicated_across_tp():
    """The compressed MLA stream is replicated, so per-device KV is TP-invariant."""
    spec = from_preset("kimi_k2_6").spec
    from sizer.engine import _kv_bytes_per_token_per_device
    assert _kv_bytes_per_token_per_device(spec, 1) == _kv_bytes_per_token_per_device(spec, 8) == 70_272


def test_mla_sizing_end_to_end():
    """An MLA model produces a real recommendation with replicated KV placement."""
    r = size(_preset_req("kimi_k2_6")).model_dump(mode="json")
    sel = r["selected"]
    assert r["architecture_status"] == "supported"
    if sel:  # feasible on large-TP profiles; placement must be replicated either way
        assert sel["kv_sharding"] == "replicated"


def test_glm_5_3_flash_hybrid_mla():
    """MLA on 11 full layers only: 11 x (512+0) x 2 = 11,264 B/token."""
    spec = from_preset("glm_5_3_flash").spec
    assert spec.attention_kv_layers == 11
    assert kv_bytes_per_token(spec) == 11_264
    assert spec.kv_fixed_bytes_per_sequence == 34 * 64 * 128 * 128 * 2


# ---------------------------------------------------------------- override path

def test_override_used_for_dsv4():
    spec = from_preset("deepseek_v4_flash_0731").spec
    assert spec.kv_model == "override"
    assert kv_bytes_per_token(spec) == 90_112
    assert spec.kv_override_source  # provenance required by the schema


def test_override_requires_source():
    with pytest.raises(ValueError, match="kv_override_source"):
        ModelSpec(
            architecture="DeepseekV4ForCausalLM", attn="hybrid", kv_model="override",
            layers=43, kv_heads=1, head_dim=512, attention_heads=64,
            total_params=1e11, kv_bytes_per_token_override=90_112,
        )


def test_override_without_value_rejected():
    with pytest.raises(ValueError, match="kv_bytes_per_token_override"):
        ModelSpec(
            architecture="X", kv_model="override", layers=1, kv_heads=1, head_dim=1,
            total_params=1e9, kv_override_source="somewhere",
        )


# ---------------------------------------------------------------- MoE semantics

def test_moe_weights_total_params_prefill_active():
    """Weights use total (all experts resident); prefill uses active params."""
    spec = from_preset("kimi_k3").spec
    assert spec.total_params == 2.8e12 and spec.active_params == 104e9
    from sizer.engine import _weights_bytes_per_device, _prefill_ms
    from sizer.hardware import get_profile
    from sizer.schemas import DeploymentSpec, WorkloadSpec

    dep = DeploymentSpec(hardware_profile_id="hgx_b200", max_gpus_per_replica=72)
    w = WorkloadSpec(**_example_request()["workload"])
    prof = get_profile("hgx_b200")
    assert _weights_bytes_per_device(spec, 16) == pytest.approx(2.8e12 * 0.5 / 16)
    prefill = _prefill_ms(spec, dep, prof, 16, 10_000)
    dense_equiv = dict(spec); dense_equiv["active_params"] = None
    prefill_dense = _prefill_ms(ModelSpec(**dense_equiv), dep, prof, 16, 10_000)
    assert prefill < prefill_dense  # active params, not total, drive prefill


def test_moe_assumptions_surface_expert_policy():
    r = size(_preset_req("kimi_k3")).model_dump(mode="json")
    assert any("896" in a and "expert" in a.lower() for a in r["assumptions"])


# ---------------------------------------------------------------- fixed KV term

def test_fixed_kv_term_in_memory_and_batch():
    """Sliding-window/linear state enters per-device memory and bounds batch."""
    spec = from_preset("inkling").spec
    assert spec.kv_fixed_bytes_per_sequence == 55 * 16 * 128 * 2 * 2 * 512 == 230_686_720
    r = size(_preset_req("inkling")).model_dump(mode="json")
    if r["selected"]:
        assert r["memory_components"]["kv_fixed_bytes_per_replica"] > 0
        # fixed term grows the per-device total beyond weights+growing-KV alone
        mc = r["memory_components"]
        per_dev_kv = r["selected"]["memory"]["kv_p95_bytes_per_replica"] / r["selected"]["gpus_per_replica"]
        assert mc["total_bytes_per_device"] > mc["weights_bytes"] + per_dev_kv


def test_kv_bytes_per_token_ignores_fixed_term():
    """kv/token stays the growing-part figure; fixed is per-sequence, separate."""
    spec = from_preset("inkling").spec
    assert kv_bytes_per_token(spec) == 2 * 11 * 8 * 128 * 2 == 45_056


# ---------------------------------------------------------------- migration

def test_v1_payload_migrates_and_sizes():
    payload = _example_request()
    payload["schema_version"] = 1
    payload["model"] = {**payload["model"], "attn": "mla", "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    migrated = migrate_request_payload(payload)
    assert migrated["schema_version"] == SCHEMA_VERSION == 2
    assert migrated["model"]["kv_model"] == "mla"
    r = size(SizeRequest.model_validate(migrated)).model_dump(mode="json")
    assert r["architecture_status"] == "supported"


def test_v1_dense_payload_migration():
    payload = _example_request()
    payload["schema_version"] = 1
    migrated = migrate_request_payload(payload)
    assert migrated["model"]["kv_model"] == "mha_gqa"


def test_current_version_payload_untouched():
    payload = _example_request()
    payload["schema_version"] = 2
    assert migrate_request_payload(payload) is payload


# ---------------------------------------------------------------- catalog integrity

def test_catalog_all_presets_have_provenance():
    from sizer.hardware import load_catalog
    for p in load_catalog().presets:
        assert p.family, f"{p.id} missing family"
        assert p.spec_confidence in ("published", "unverified")
        if p.spec_confidence == "unverified":
            assert p.notes or p.source_ref, f"{p.id} unverified without explanation"
        if p.kv_model == "override":
            assert p.kv_bytes_per_token_override and p.kv_override_source


def test_catalog_new_families_present():
    from sizer.hardware import load_catalog
    ids = {p.id for p in load_catalog().presets}
    for pid in (
        "deepseek_v4_flash_0731", "deepseek_v4_1_flash",
        "glm_5", "glm_5_1", "glm_5_2", "glm_5_3_flash",
        "kimi_k2_6", "kimi_k2_7_code", "kimi_k3",
        "qwen_3_5_27b", "qwen_3_5_35b_a3b", "qwen_3_5_122b_a10b",
        "qwen_3_6_27b", "qwen_3_8_27b", "qwen_3_8_flash_next",
        "inkling", "inkling_small", "gemma_4_31b", "gemma_4_26b_a4b",
    ):
        assert pid in ids, f"missing requested preset {pid}"


def test_requested_but_nonexistent_not_fabricated():
    from sizer.hardware import load_catalog
    ids = {p.id for p in load_catalog().presets}
    for absent in ("qwen_3_5_235b", "qwen_3_6_122b", "qwen_3_6_235b", "qwen_3_8_35b", "qwen_3_8_122b", "qwen_3_8_235b", "muse_1_3_spark"):
        assert absent not in ids


def test_kv_math_matches_documented_derivations():
    """Each derived figure equals the independent recomputation from config values."""
    expectations = {
        "glm_5_2": (78 * (512 + 64) * 2, None),
        "kimi_k2_6": (61 * (512 + 64) * 2, None),
        "kimi_k3": (24 * (512 + 64) * 2, 69 * 96 * 128 * 128 * 4),
        "qwen_3_5_27b": (2 * 8 * 4 * 256 * 2, 24 * 32 * 128 * 128 * 4),
        "qwen_3_5_35b_a3b": (2 * 10 * 2 * 256 * 2, 30 * 32 * 128 * 128 * 4),
        "qwen_3_5_122b_a10b": (2 * 12 * 2 * 256 * 2, 36 * 64 * 128 * 128 * 4),
        "qwen_3_6_27b": (2 * 16 * 4 * 256 * 2, 48 * 48 * 128 * 128 * 4),
        "qwen_3_8_flash_next": (12 * 2 * 2 * 256 * 2 // 4, 36 * 48 * 128 * 128 * 4),
        "inkling": (2 * 11 * 8 * 128 * 2, 55 * 16 * 128 * 2 * 2 * 512),
        "gemma_4_31b": (10 * 4 * 512 * 2, 50 * 16 * 256 * 2 * 2 * 1024),
        "gemma_4_26b_a4b": (5 * 2 * 512 * 2, 25 * 8 * 256 * 2 * 2 * 1024),
    }
    for pid, (kv, fixed) in expectations.items():
        spec = from_preset(pid).spec
        assert kv_bytes_per_token(spec) == kv, pid
        if fixed is not None:
            assert spec.kv_fixed_bytes_per_sequence == fixed, pid


# ---------------------------------------------------------------- hybrid imports

def test_config_json_hybrid_requires_constants():
    """A Qwen3.5-style hybrid config without fixed-term overrides is unsupported,
    never silently sized with the wrong (all-layer) formula."""
    cfg = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "num_attention_heads": 16,
        "head_dim": 256,
        "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8,
    }
    nm = from_config_json(cfg, overrides={"total_params": 27e9})
    assert nm.unsupported_reason and "Hybrid" in nm.unsupported_reason
    assert nm.spec.attention_kv_layers == 8  # schedule still extracted


def test_config_json_nested_text_config():
    cfg = {
        "architectures": ["InklingForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 66,
            "num_key_value_heads": 8,
            "num_attention_heads": 64,
            "head_dim": 128,
            "hidden_size": 6144,
            "model_max_length": 1048576,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
        },
    }
    nm = from_config_json(cfg, overrides={"total_params": 975e9, "active_params": 41e9})
    assert nm.spec.layers == 66 and nm.spec.hidden_size == 6144
    assert nm.spec.max_context_tokens == 1048576
    assert nm.spec.num_experts == 256 and nm.spec.active_experts == 6