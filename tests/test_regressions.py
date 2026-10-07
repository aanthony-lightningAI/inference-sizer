"""The four named regressions from the plan (audit probes on the OLD engine)."""

import json

import pytest

from sizer.engine import kv_bytes_per_token, size
from sizer.schemas import SizeRequest
from sizer.cli import _example_request


def _base() -> dict:
    return json.loads(json.dumps(_example_request()))


def _req(**over) -> SizeRequest:
    d = _base()
    d.update(over)
    return SizeRequest.model_validate(d)


def test_ttft_bound_rejects_failing_candidate():
    """Old engine: ttft_ms=1 -> fleet_gpus=4 despite ~85 ms prefill estimate.
    New engine must reject the candidate and cannot claim SLO compliance."""
    base = _base()
    r = size(_req(workload={**base["workload"], "ttft_ms_p95": 1.0})).model_dump(mode="json")
    assert r["feasibility"] == "infeasible"
    assert r["fleet_gpus"] is None
    assert "TTFT" in " ".join(r["rejection_reasons"])

    # A passing bound never claims p95 compliance: bounds-only flag stays on.
    r_pass = size(_req()).model_dump(mode="json")
    assert r_pass["selected"]["prefill_bound_ms"] is not None
    assert r_pass["latency_bound_only"] is True


def test_fleet_concurrency_enforced():
    """Old engine: rps=0.1, inflight=10000 -> 1 serving replica with capacity 103.
    New engine must size serving replicas for the concurrent demand."""
    base = _base()
    r = size(_req(workload={**base["workload"], "peak_rps": 0.1, "peak_inflight": 10000})).model_dump(mode="json")
    assert r["feasibility"] == "feasible"
    assert r["capacity_concurrency"] >= 10000
    util = r["request"]["deployment"]["operating_factor"]
    assert r["serving_replicas"] * r["selected"]["max_concurrency"] >= 10000 / util


def test_prefix_hit_does_not_erase_input_memory():
    """Old engine: prefix_hit=1.0 dropped p95 KV 1.56 -> 0.31 GiB.
    New engine: per-sequence KV is identical for any hit rate; monotonic in context."""
    base = _base()
    req0 = _req(workload={**base["workload"], "prefix_hit_rate": 0.0})
    req1 = _req(workload={**base["workload"], "prefix_hit_rate": 1.0})
    assert kv_bytes_per_token(req0.model) == kv_bytes_per_token(req1.model)

    # Per-sequence memory at equal batch is identical (no sharing credit).
    r0 = size(_req(workload={**base["workload"], "prefix_hit_rate": 0.0, "peak_inflight": 100})).model_dump(mode="json")
    r1 = size(_req(workload={**base["workload"], "prefix_hit_rate": 1.0, "peak_inflight": 100})).model_dump(mode="json")
    seq0 = r0["selected"]["memory"]["kv_p95_bytes_per_replica"] / r0["selected"]["max_concurrency"]
    seq1 = r1["selected"]["memory"]["kv_p95_bytes_per_replica"] / r1["selected"]["max_concurrency"]
    assert seq0 == seq1

    # Longer retained context can never reduce per-sequence memory.
    def seq_at(in_p95):
        r = size(_req(workload={**base["workload"], "p95_input_tokens": in_p95, "peak_inflight": 100})).model_dump(mode="json")
        return r["selected"]["memory"]["kv_p95_bytes_per_replica"] / r["selected"]["max_concurrency"]

    assert seq_at(8192) > seq_at(2048)


def test_invalid_inputs_field_errors():
    """max_gpus=0 and negative layers raise field-specific errors, plus
    non-finite, out-of-range fractions, and zero-capacity inputs."""
    base = _base()

    def expect_error(patch, field_frag):
        with pytest.raises(Exception) as ei:
            SizeRequest.model_validate({**base, **patch})
        locs = [".".join(str(p) for p in e["loc"]) for e in ei.value.errors()]
        assert any(field_frag in loc for loc in locs), f"expected {field_frag} in {locs}"

    expect_error({"deployment": {**base["deployment"], "max_gpus_per_replica": 0}}, "max_gpus_per_replica")
    expect_error({"deployment": {**base["deployment"], "max_gpus_per_replica": -3}}, "max_gpus_per_replica")
    expect_error({"model": {**base["model"], "layers": -5}}, "layers")
    expect_error({"model": {**base["model"], "head_dim": 0}}, "head_dim")
    expect_error({"model": {**base["model"], "total_params": 0}}, "total_params")
    expect_error({"deployment": {**base["deployment"], "observed_memory_gb": -1}}, "observed_memory_gb")
    expect_error({"workload": {**base["workload"], "peak_rps": float("nan")}}, "peak_rps")
    expect_error({"workload": {**base["workload"], "peak_rps": float("inf")}}, "peak_rps")
    expect_error({"workload": {**base["workload"], "ttft_ms_p95": 0}}, "ttft_ms_p95")
    expect_error({"workload": {**base["workload"], "prefix_hit_rate": 1.5}}, "prefix_hit_rate")
    expect_error({"deployment": {**base["deployment"], "operating_factor": 0}}, "operating_factor")
    expect_error({"deployment": {**base["deployment"], "spare_replicas": -1}}, "spare_replicas")
    expect_error({"model": {**base["model"], "kv_heads": 64, "attention_heads": 32}}, "model")

    # Legitimate zero traffic is NOT an error; distinct from zero capacity.
    r = size(_req(workload={**base["workload"], "peak_rps": 0, "peak_inflight": 0})).model_dump(mode="json")
    assert r["feasibility"] == "feasible"
    assert r["serving_replicas"] == 0
    assert "Zero traffic" in " ".join(r["notes"])


def test_unsupported_mla():
    base = _base()
    r = size(_req(model={**base["model"], "attn": "mla"})).model_dump(mode="json")
    assert r["architecture_status"] == "unsupported"
    assert r["feasibility"] == "unsupported"
    assert r["selected"] is None