"""Independent memory/weight calculations, units, tail envelope, KV sharding."""

import json

from sizer.engine import kv_bytes_per_token
from sizer.hardware import device_memory_bytes, get_profile
from sizer.schemas import SizeRequest
from sizer.cli import _example_request


def _req(**over):
    d = json.loads(json.dumps(_example_request()))
    d.update(over)
    return SizeRequest.model_validate(d)


def test_kv_formula_independent():
    """2 x layers x kv_heads x head_dim x kv_element_bytes, checked by hand."""
    m = _req().model
    # 2 * 80 layers * 8 heads * 128 dim * 2 bytes (bf16)
    assert kv_bytes_per_token(m) == 2 * 80 * 8 * 128 * 2 == 327680

    m8 = m.model_copy(update={"kv_dtype": "fp8"})
    assert kv_bytes_per_token(m8) == 2 * 80 * 8 * 128 * 1

    # KV sharding halves per-device bytes at TP=4 (8 heads / 4)
    from sizer.engine import _kv_bytes_per_token_per_device
    assert _kv_bytes_per_token_per_device(m, 4) == 2 * 80 * 2 * 128 * 2
    # kv_heads=2 < TP=4 -> replicated (full per device)
    m2 = m.model_copy(update={"kv_heads": 2})
    assert _kv_bytes_per_token_per_device(m2, 4) == 2 * 80 * 2 * 128 * 2
    # kv_heads=6 not divisible by TP=4 and not < TP -> invalid
    m6 = m.model_copy(update={"kv_heads": 6})
    try:
        _kv_bytes_per_token_per_device(m6, 4)
        raise AssertionError("expected ValueError for non-divisible kv_heads")
    except ValueError:
        pass


def test_weight_bytes_independent():
    """total_params x weight_format bytes, verified against a hand calculation."""
    req = _req()
    m = req.model
    weights = m.total_params * 2.0  # bf16
    r = None
    from sizer.engine import size
    result = size(req).model_dump(mode="json")
    tp = result["selected"]["gpus_per_replica"]
    per_dev = result["memory_components"]["weights_bytes"]
    assert abs(per_dev - weights / tp) < 1.0


def test_units_gb_vs_gib_displayed():
    """Published GB anchors convert to bytes via 1e9 (decimal); GiB displayed too."""
    profile = get_profile("hgx_b200")
    mem_bytes, source = device_memory_bytes(profile, None)
    assert source == "published"
    assert mem_bytes == int(180 * 1e9)  # decimal GB anchor -> bytes
    observed, source2 = device_memory_bytes(profile, 170.0)
    assert source2 == "observed" and observed == int(170 * 1024**3)  # GiB input


def test_memory_components_are_separate_terms():
    r = None
    from sizer.engine import size
    result = size(_req()).model_dump(mode="json")
    mc = result["memory_components"]
    total = (
        mc["weights_bytes"] + mc["runtime_overhead_bytes"] + mc["cuda_graphs_bytes"]
        + mc["activations_workspace_bytes"] + mc["reserve_bytes"]
        + result["selected"]["memory"]["kv_p95_bytes_per_replica"] / result["selected"]["gpus_per_replica"]
    )
    assert abs(total - mc["total_bytes_per_device"]) < 2.0
    assert mc["device_memory_source"] == "published"


def test_tail_envelope_is_labeled_not_p95_total():
    from sizer.engine import size
    result = size(_req()).model_dump(mode="json")
    assert any("conservative tail envelope" in n for n in result["notes"])

    # Hard context limit caps memory and warns about admission.
    base = json.loads(json.dumps(_example_request()))
    base["model"]["max_context_tokens"] = 6144
    r = size(SizeRequest.model_validate(base)).model_dump(mode="json")
    assert any("hard context limit" in w for w in r["warnings"])


def test_per_device_not_pooled():
    """Weights must fit per device: 405B bf16 (=810 GB) cannot fit TP=4 x 180 GB."""
    from sizer.engine import size
    base = json.loads(json.dumps(_example_request()))
    base["model"] = {
        "source": "preset", "preset_id": "llama_3_1_405b", "architecture": "LlamaForCausalLM",
        "attn": "gqa", "layers": 126, "kv_heads": 8, "head_dim": 128,
        "attention_heads": 128, "hidden_size": 16384, "total_params": 405000000000,
        "weight_format": "bf16", "kv_dtype": "bf16",
    }
    r = size(SizeRequest.model_validate(base)).model_dump(mode="json")
    for c in r["rejection_reasons"]:
        assert isinstance(c, str)
    assert r["selected"] is None or r["selected"]["gpus_per_replica"] >= 8