export const FIELDS = [
  ["customer", "Customer", "text", "Customer", true],
  ["tier", "Workload tier", "text", "Interactive chat", true],
  ["params_b", "Total params (B)", "number", 70],
  ["active_b", "Active params (B)", "number", 70],
  ["layers", "Layers", "number", 80],
  ["kv_heads", "KV heads", "number", 8],
  ["head_dim", "Head dim", "number", 128],
  ["attn", "Attention", "select", "gqa", false, [["gqa", "GQA / MHA"], ["mla", "MLA"]]],
  ["mla_dim", "MLA latent + RoPE", "number", 576],
  ["weight_bytes", "Weight dtype", "select", 1, false, [["2", "BF16 (2 B)"], ["1", "FP8 (1 B)"], ["0.5", "FP4 (0.5 B)"]]],
  ["kv_bytes", "KV dtype", "select", 1, false, [["2", "BF16 (2 B)"], ["1", "FP8 (1 B)"], ["0.5", "FP4 (0.5 B)"]]],
  ["rps", "Peak requests / sec", "number", 20],
  ["inflight", "Peak in-flight", "number", 160],
  ["in_mean", "Input tokens, mean", "number", 2048],
  ["in_p95", "Input tokens, p95", "number", 8192],
  ["out_mean", "Output tokens, mean", "number", 512],
  ["out_p95", "Output tokens, p95", "number", 2048],
  ["prefix_hit", "Prefix-cache hit rate", "number", 0.4],
  ["ttft_ms", "TTFT SLO p95 (ms)", "number", 800],
  ["itl_ms", "Inter-token SLO p95 (ms)", "number", 40],
  ["util", "Utilization cap", "number", 0.65],
  ["margin_replicas", "Failure margin (replicas)", "number", 1],
  ["hbm_gb", "HBM (GB)", "number", 192],
  ["bw_tb_s", "HBM bandwidth (TB/s)", "number", 8],
  ["flops_tflops", "Dense TFLOP/s", "number", 4500],
  ["bw_eff", "Bandwidth efficiency", "number", 0.65],
  ["flops_eff", "Compute efficiency", "number", 0.45],
  ["max_gpus_per_replica", "Max GPUs per replica", "number", 8],
  ["overhead_gb", "Runtime overhead (GB)", "number", 2],
  ["link_gb_s", "KV transfer link (GB/s)", "number", 50],
  ["link_eff", "Transfer efficiency", "number", 0.5],
];

export function formToRequest(form) {
  const req = { disagg: form.elements.disagg.checked, gpu_name: form.elements.gpu_name.value };
  for (const [name, , type] of FIELDS) {
    const raw = form.elements[name].value;
    req[name] = type === "number" ? Number(raw) : raw;
  }
  return req;
}

export function applyPreset(form, preset) {
  for (const [k, v] of Object.entries(preset)) {
    if (k === "name" || !form.elements[k]) continue;
    form.elements[k].value = v;
  }
}

export function applyGpu(form, gpu) {
  form.elements.gpu_name.value = gpu.name;
  form.elements.hbm_gb.value = gpu.hbm_gb;
  form.elements.bw_tb_s.value = gpu.bw_tb_s;
  const wb = String(form.elements.weight_bytes.value);
  const flops = gpu.flops[wb] ?? gpu.flops[1] ?? gpu.flops["1"];
  if (flops) form.elements.flops_tflops.value = flops;
}
