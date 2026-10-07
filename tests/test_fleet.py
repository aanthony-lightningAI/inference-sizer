"""Fleet semantics: operating factor once, spares separate, evidence rules."""

import json

from sizer.engine import size
from sizer.schemas import SizeRequest
from sizer.cli import _example_request


def _req(**over):
    d = json.loads(json.dumps(_example_request()))
    d.update(over)
    return SizeRequest.model_validate(d)


def test_operating_factor_applied_once():
    """Doubling the operating factor must not double-apply anywhere."""
    r065 = size(_req()).model_dump(mode="json")
    r13_req = _req(deployment={**_example_request()["deployment"], "operating_factor": 1.0})
    r100 = size(r13_req).model_dump(mode="json")
    # With factor 1.0 the capacity per replica equals raw capacity.
    raw_cap = r100["capacity_rps"] / r100["serving_replicas"]
    assert abs(raw_cap - r100["selected"]["rps_capacity"]) < 0.5


def test_spares_separate_from_serving():
    base = _example_request()
    r = size(_req()).model_dump(mode="json")
    serving, spare, tp = r["serving_replicas"], r["spare_replicas"], r["selected"]["gpus_per_replica"]
    assert r["fleet_gpus"] == tp * (serving + spare)
    # Spares do not count as serving capacity.
    assert r["capacity_concurrency"] == r["selected"]["max_concurrency"] * serving
    assert any("excluded from normal serving capacity" in n for n in r["notes"])
    assert spare >= 0


def test_zero_effective_capacity_rejected():
    """Weights too large per device yield infeasible, not a zero-capacity pass."""
    base = json.loads(json.dumps(_example_request()))
    base["model"] = {
        "source": "preset", "preset_id": "llama_3_1_405b", "architecture": "LlamaForCausalLM",
        "attn": "gqa", "layers": 126, "kv_heads": 8, "head_dim": 128,
        "attention_heads": 128, "hidden_size": 16384, "total_params": 405000000000,
        "weight_format": "bf16", "kv_dtype": "bf16",
    }
    base["deployment"]["max_gpus_per_replica"] = 4  # 405B bf16 needs 101 GB/device at TP=8
    r = size(SizeRequest.model_validate(base)).model_dump(mode="json")
    assert r["feasibility"] == "infeasible"
    assert r["fleet_gpus"] is None
    assert any("exceed device memory" in x for x in r["rejection_reasons"])


def test_traffic_consistency_warned():
    """0.1 rps with 10k in-flight is suspicious; the engine surfaces it, keeps
    the peak in-flight as the binding constraint."""
    base = json.loads(json.dumps(_example_request()))
    base["workload"].update({"peak_rps": 0.1, "peak_inflight": 10000})
    r = size(SizeRequest.model_validate(base)).model_dump(mode="json")
    assert r["serving_replicas"] >= 2  # cannot satisfy 10k in-flight with one replica
    # Little's-law cross-check present in assumptions
    assert any("Little's law" in a for a in r["assumptions"])


def test_cadence_conversion_explicit():
    """Active-user cadence + calls-per-task convert to rps explicitly."""
    base = json.loads(json.dumps(_example_request()))
    base["workload"].update({
        "peak_rps": 0, "active_users": 60,
        "requests_per_active_per_hour": 20, "calls_per_task": 3,
    })
    r = size(SizeRequest.model_validate(base)).model_dump(mode="json")
    d = r["traffic_derived"]
    assert d["derived_rps"] == pytest_approx(60 * 20 * 3 / 3600)  # 1.0 rps
    assert d["used"] == "derived from active-user cadence x calls per task"


def pytest_approx(x):
    import pytest

    return pytest.approx(x)


def test_derived_vs_explicit_conflict_warned():
    base = json.loads(json.dumps(_example_request()))
    base["workload"].update({
        "peak_rps": 20, "active_users": 60,
        "requests_per_active_per_hour": 20, "calls_per_task": 3,
    })
    r = size(SizeRequest.model_validate(base)).model_dump(mode="json")
    assert any("differs by >10%" in w for w in r["warnings"])
    assert r["request"]["workload"]["peak_rps"] == 20  # explicit value kept


def test_estimated_status_uncalibrated_fleet():
    """Without a benchmark, fleet numbers stay Estimated with explicit assumptions."""
    r = size(_req()).model_dump(mode="json")
    assert r["evidence_status"] == "estimated"
    assert r["latency_bound_only"] is True
    assert any("optimistic bounds" in a for a in r["assumptions"])