"""CLI: python -m sizer.cli [request.json] [--export out.json]

Uses the exact same size() path as the API.
"""

from __future__ import annotations

import argparse
import json
import sys

from sizer.engine import size
from sizer.schemas import SizeRequest


def _result_from_payload(payload: dict) -> tuple[SizeRequest, dict | None]:
    """Accept a raw request or a full scenario export (request + benchmark)."""
    if "request" in payload and isinstance(payload["request"], dict):
        benchmark = payload.get("benchmark_export")
        req = SizeRequest.model_validate(payload["request"])
        return req, benchmark.get("profile") if isinstance(benchmark, dict) else None
    return SizeRequest.model_validate(payload), None


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightning AI Inference Sizer (CLI)")
    ap.add_argument("request", nargs="?", help="Request JSON file (or full scenario export)")
    ap.add_argument("--export", metavar="OUT", help="Write the complete scenario export")
    args = ap.parse_args()

    if args.request:
        with open(args.request) as f:
            payload = json.load(f)
        req, benchmark = _result_from_payload(payload)
    else:
        req = SizeRequest.model_validate(_example_request())
        benchmark = None

    result = size(req, benchmark=benchmark)
    out = result.model_dump(mode="json")
    if args.export:
        export = {
            "export_schema_version": 1,
            "calculator_version": result.calculator_version,
            "request": json.loads(req.model_dump_json()),
            "result": out,
            "benchmark_export": (
                {"profile": benchmark, "profile_id": result.benchmark_profile_id}
                if benchmark
                else None
            ),
        }
        with open(args.export, "w") as f:
            json.dump(export, f, indent=2)
        sys.stdout.write(f"Wrote full scenario export to {args.export}\n")
    else:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")


def _example_request() -> dict:
    return {
        "customer": "Example Co",
        "tier": "Interactive chat",
        "model": {
            "source": "preset",
            "preset_id": "llama_3_1_70b",
            "attn": "gqa",
            "layers": 80,
            "kv_heads": 8,
            "head_dim": 128,
            "attention_heads": 64,
            "hidden_size": 8192,
            "total_params": 70600000000,
            "weight_format": "bf16",
            "kv_dtype": "bf16",
        },
        "workload": {
            "mean_input_tokens": 2048,
            "p95_input_tokens": 8192,
            "mean_output_tokens": 512,
            "p95_output_tokens": 2048,
            "peak_rps": 20,
            "peak_inflight": 160,
            "ttft_ms_p95": 800,
            "decode_ms_p95": 40,
            "prefix_hit_rate": 0.4,
        },
        "deployment": {
            "hardware_profile_id": "hgx_b200",
            "max_gpus_per_replica": 8,
            "spare_replicas": 1,
            "operating_factor": 0.65,
        },
    }


if __name__ == "__main__":
    main()