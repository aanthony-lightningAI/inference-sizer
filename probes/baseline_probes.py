# Reproduce the four baseline failure probes against the ORIGINAL engine.
# Requires the unmodified baseline (git show 08bd435:sizer/engine.py).
# Usage: PYTHONPATH=<repo-with-original-engine> .venv/bin/python probes/baseline_probes.py

from sizer.engine import SizeRequest, size

base = SizeRequest()

print("=== Probe A: ttft_ms=1 ===")
r = size(base.model_copy(update={"ttft_ms": 1.0}))
c = r["colocated"]["replica"]
print("fleet_gpus:", r["fleet_gpus"], "| prefill_ms:", round(c["prefill_ms"], 1), "| ttft_ok:", c["ttft_ok"])

print("=== Probe B: rps=0.1 inflight=10000 ===")
r = size(base.model_copy(update={"rps": 0.1, "inflight": 10000}))
c = r["colocated"]["replica"]
print("fleet_gpus:", r["fleet_gpus"], "| replicas:", c["replicas"], "| concurrency capacity:", c["concurrency"])

print("=== Probe C: prefix_hit=1.0 ===")
r0 = size(base.model_copy(update={"prefix_hit": 0.0}))
r1 = size(base.model_copy(update={"prefix_hit": 1.0}))
print("prefix_hit=0.0 p95 KV per seq GiB:", round(r0["colocated"]["replica"]["kv_per_seq_gib"], 2))
print("prefix_hit=1.0 p95 KV per seq GiB:", round(r1["colocated"]["replica"]["kv_per_seq_gib"], 2))

print("=== Probe D: max_gpus_per_replica=0 ===")
try:
    size(base.model_copy(update={"max_gpus_per_replica": 0}))
except Exception as e:
    print(type(e).__name__, ":", str(e).splitlines()[-1])

print("=== Probe D2: layers=-5 ===")
try:
    r = size(base.model_copy(update={"layers": -5}))
except Exception as e:
    print("raised:", type(e).__name__)
else:
    print("accepted; kv_bytes_per_token:", r["kv_bytes_per_token"])