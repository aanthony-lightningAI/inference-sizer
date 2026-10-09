"""Benchmark adapter: synthetic exclusion, compatibility, calibration."""

import json

from sizer.benchmarks import compatibility_report, parse_profile
from sizer.engine import size
from sizer.schemas import SizeRequest
from sizer.cli import _example_request


def _req(**over):
    d = json.loads(json.dumps(_example_request()))
    d.update(over)
    return SizeRequest.model_validate(d)


def test_synthetic_never_confers_calibration(synthetic_profile):
    r = size(_req(benchmark_profile_id=synthetic_profile["profile_id"]),
             benchmark=synthetic_profile).model_dump(mode="json")
    assert r["evidence_status"] == "estimated"
    assert r["benchmark_synthetic"] is True
    assert any("Synthetic" in w for w in r["warnings"])


def test_material_mismatch_strips_calibration(synthetic_profile):
    real = {**synthetic_profile, "synthetic": False}
    # Different weight format -> material mismatch.
    req = _req(
        benchmark_profile_id=synthetic_profile["profile_id"],
        model={**_example_request()["model"], "weight_format": "fp8"},
    )
    r = size(req, benchmark=real).model_dump(mode="json")
    assert r["evidence_status"] == "estimated"
    assert any("weight_format" in m for m in r["benchmark"]["mismatches"])

    # Different engine -> material mismatch.
    req2 = _req(
        benchmark_profile_id=synthetic_profile["profile_id"],
        deployment={**_example_request()["deployment"],
                    "engine": {"name": "sglang", "version": None}},
    )
    r2 = size(req2, benchmark=real).model_dump(mode="json")
    assert r2["evidence_status"] == "estimated"
    assert any("engine" in m for m in r2["benchmark"]["mismatches"])


def test_calibration_applied_when_compatible(synthetic_profile):
    """Make the request match the profile exactly (non-synthetic)."""
    real = {
        **synthetic_profile,
        "synthetic": False,
        "source": "measured run (example import)",
        "prefix_behavior": "cold",
    }
    base = json.loads(json.dumps(_example_request()))
    base["model"]["quantization"] = None
    base["workload"]["prefix_hit_rate"] = 0.0  # profile measured cold prefixes
    base["deployment"]["max_gpus_per_replica"] = 4
    base["benchmark_profile_id"] = real["profile_id"]
    req = SizeRequest.model_validate(base)
    assert compatibility_report(parse_profile(real), req) == []
    r = size(req, benchmark=real).model_dump(mode="json")
    assert r["evidence_status"] == "benchmark_calibrated"
    assert r["latency_bound_only"] is False
    # Serving from measured capacity: rps 20 / (12 * 0.65) = 3 (ceil), conc
    # 160 / (120*0.65)=3 (ceil) -> 3
    assert r["serving_replicas"] == 3
    assert r["fleet_gpus"] == 4 * (3 + 1)
    assert r["benchmark"]["metrics"]["ttft_ms_p95"] == 640.0


def test_profile_id_mismatch_refused(synthetic_profile):
    r = size(_req(benchmark_profile_id="some_other_profile"),
             benchmark=synthetic_profile).model_dump(mode="json")
    assert r["evidence_status"] == "estimated"
    assert any("does not match" in w for w in r["warnings"])


def test_bad_schema_version_refused(synthetic_profile):
    bad = {**synthetic_profile, "schema_version": 99}
    r = size(_req(benchmark_profile_id=synthetic_profile["profile_id"]), benchmark=bad).model_dump(mode="json")
    assert r["evidence_status"] == "estimated"
    assert any("rejected" in w for w in r["warnings"])


def test_tpot_vs_itl_distinction_preserved():
    p = parse_profile({
        **json.loads(json.dumps(_example_request())),  # dummy to satisfy import
    }) if False else None
    from sizer.benchmarks import BenchmarkMetrics
    m = BenchmarkMetrics(
        ttft_ms_p95=100, decode_ms_p95=20, decode_metric="tpot",
        sustainable_rps=1, sustainable_concurrency=1, completion_rate=1.0,
        error_rate=0, slo_miss_rate=0, avg_time_per_output_token_ms=15,
    )
    assert m.decode_metric == "tpot"  # preserved as stated, never relabeled itl