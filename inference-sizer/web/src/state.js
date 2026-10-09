// Application state: mirrors SizeRequest schema v2. All edits flow through
// this module so export/import and the form stay in sync.

export const state = {
  customer: "Example Co",
  tier: "Interactive chat",
  model: {
    source: "preset",
    preset_id: "llama_3_1_70b",
    architecture: "LlamaForCausalLM",
    attn: "gqa",
    kv_model: "mha_gqa",
    layers: 80,
    attention_kv_layers: null,
    kv_heads: 8,
    head_dim: 128,
    kv_lora_rank: null,
    qk_rope_head_dim: null,
    kv_bytes_per_token_override: null,
    kv_override_source: null,
    kv_fixed_bytes_per_sequence: null,
    num_experts: null,
    active_experts: null,
    attention_heads: 64,
    hidden_size: 8192,
    total_params_b: 70.6, // UI works in billions; converted to absolute on send
    active_params_b: null,
    max_context_tokens: 131072,
    checkpoint_revision: "",
    weight_format: "bf16",
    kv_dtype: "bf16",
    quantization: "",
  },
  workload: {
    mean_input_tokens: 2048,
    p95_input_tokens: 8192,
    mean_output_tokens: 512,
    p95_output_tokens: 2048,
    peak_rps: 20,
    peak_inflight: 160,
    ttft_ms_p95: 800,
    decode_ms_p95: 40,
    decode_metric: "itl",
    prefix_hit_rate: 0.4,
    active_users: null,
    requests_per_active_per_hour: null,
    calls_per_task: null,
  },
  deployment: {
    hardware_profile_id: "hgx_b200",
    engine: { name: "vllm", version: "0.11.x" },
    max_gpus_per_replica: 8,
    spare_replicas: 1,
    operating_factor: 0.65,
    observed_memory_gb: null,
    allocation_size_gpus: null,
    bw_eff: 0.65,
    flops_eff: 0.45,
    runtime_overhead_gb: 2,
    cuda_graphs_gb: 1.5,
    activations_workspace_gb: 2,
    reserve_fraction: 0.05,
  },
  benchmark: null, // imported benchmark_profile/v1 payload (optional)
  // provenance: field -> source label for display
  provenance: {},
};

export function buildRequest() {
  const m = state.model;
  const model = {
    source: m.source,
    preset_id: m.source === "preset" ? m.preset_id : null,
    architecture: m.architecture || null,
    attn: m.attn,
    kv_model: m.kv_model || "mha_gqa",
    layers: m.layers,
    attention_kv_layers: m.attention_kv_layers ?? null,
    kv_heads: m.kv_heads,
    head_dim: m.head_dim,
    kv_lora_rank: m.kv_lora_rank ?? null,
    qk_rope_head_dim: m.qk_rope_head_dim ?? null,
    kv_bytes_per_token_override: m.kv_bytes_per_token_override ?? null,
    kv_override_source: m.kv_override_source || null,
    kv_fixed_bytes_per_sequence: m.kv_fixed_bytes_per_sequence ?? null,
    num_experts: m.num_experts ?? null,
    active_experts: m.active_experts ?? null,
    attention_heads: m.attention_heads || null,
    hidden_size: m.hidden_size || null,
    total_params: Math.round(m.total_params_b * 1e9),
    active_params: m.active_params_b ? Math.round(m.active_params_b * 1e9) : null,
    max_context_tokens: m.max_context_tokens || null,
    checkpoint_revision: m.checkpoint_revision || null,
    weight_format: m.weight_format,
    quantization: m.quantization || null,
    kv_dtype: m.kv_dtype,
  };
  const w = state.workload;
  const workload = {
    mean_input_tokens: w.mean_input_tokens,
    p95_input_tokens: w.p95_input_tokens,
    mean_output_tokens: w.mean_output_tokens,
    p95_output_tokens: w.p95_output_tokens,
    peak_rps: w.peak_rps,
    peak_inflight: w.peak_inflight,
    ttft_ms_p95: w.ttft_ms_p95,
    decode_ms_p95: w.decode_ms_p95,
    decode_metric: w.decode_metric,
    prefix_hit_rate: w.prefix_hit_rate,
    active_users: w.active_users ?? null,
    requests_per_active_per_hour: w.requests_per_active_per_hour ?? null,
    calls_per_task: w.calls_per_task ?? null,
  };
  const d = state.deployment;
  const deployment = {
    hardware_profile_id: d.hardware_profile_id,
    engine: { ...d.engine, version: d.engine.version || null },
    max_gpus_per_replica: d.max_gpus_per_replica,
    spare_replicas: d.spare_replicas,
    operating_factor: d.operating_factor,
    observed_memory_gb: d.observed_memory_gb ?? null,
    allocation_size_gpus: d.allocation_size_gpus ?? null,
    bw_eff: d.bw_eff,
    flops_eff: d.flops_eff,
    runtime_overhead_gb: d.runtime_overhead_gb,
    cuda_graphs_gb: d.cuda_graphs_gb,
    activations_workspace_gb: d.activations_workspace_gb,
    reserve_fraction: d.reserve_fraction,
  };
  const req = {
    schema_version: 2,
    customer: state.customer,
    tier: state.tier,
    model,
    workload,
    deployment,
  };
  if (state.benchmark) {
    req.benchmark = state.benchmark;
    req.benchmark_profile_id = state.benchmark.profile_id;
  }
  return req;
}

export function loadRequest(req) {
  // Accepts a request or a full scenario export.
  const r = req.request && req.result ? req.request : req;
  state.customer = r.customer ?? state.customer;
  state.tier = r.tier ?? state.tier;
  const m = r.model;
  state.model = {
    ...state.model,
    source: m.source ?? "manual",
    preset_id: m.preset_id ?? null,
    architecture: m.architecture ?? "",
    attn: m.attn,
    kv_model: m.kv_model ?? "mha_gqa",
    layers: m.layers,
    attention_kv_layers: m.attention_kv_layers ?? null,
    kv_heads: m.kv_heads,
    head_dim: m.head_dim,
    kv_lora_rank: m.kv_lora_rank ?? null,
    qk_rope_head_dim: m.qk_rope_head_dim ?? null,
    kv_bytes_per_token_override: m.kv_bytes_per_token_override ?? null,
    kv_override_source: m.kv_override_source ?? null,
    kv_fixed_bytes_per_sequence: m.kv_fixed_bytes_per_sequence ?? null,
    num_experts: m.num_experts ?? null,
    active_experts: m.active_experts ?? null,
    attention_heads: m.attention_heads,
    hidden_size: m.hidden_size,
    total_params_b: m.total_params / 1e9,
    active_params_b: m.active_params ? m.active_params / 1e9 : null,
    max_context_tokens: m.max_context_tokens,
    checkpoint_revision: m.checkpoint_revision ?? "",
    weight_format: m.weight_format,
    kv_dtype: m.kv_dtype,
    quantization: m.quantization ?? "",
  };
  state.workload = { ...state.workload, ...r.workload };
  state.deployment = {
    ...state.deployment,
    ...r.deployment,
    engine: { ...state.deployment.engine, ...(r.deployment?.engine ?? {}) },
  };
  state.benchmark = r.benchmark ?? (req.benchmark_export?.profile ?? null);
  state.provenance = {};
}

export function applyPreset(preset) {
  state.model = {
    ...state.model,
    source: "preset",
    preset_id: preset.id,
    architecture: preset.architecture,
    attn: preset.attn,
    kv_model: preset.kv_model || "mha_gqa",
    layers: preset.layers,
    attention_kv_layers: preset.attention_kv_layers ?? null,
    kv_heads: preset.kv_heads,
    head_dim: preset.head_dim,
    kv_lora_rank: preset.kv_lora_rank ?? null,
    qk_rope_head_dim: preset.qk_rope_head_dim ?? null,
    kv_bytes_per_token_override: preset.kv_bytes_per_token_override ?? null,
    kv_override_source: preset.kv_override_source ?? null,
    kv_fixed_bytes_per_sequence: preset.kv_fixed_bytes_per_sequence ?? null,
    num_experts: preset.num_experts ?? null,
    active_experts: preset.active_experts ?? null,
    attention_heads: preset.attention_heads,
    hidden_size: preset.hidden_size,
    total_params_b: preset.total_params / 1e9,
    active_params_b: preset.active_params ? preset.active_params / 1e9 : null,
    max_context_tokens: preset.max_context_tokens,
    weight_format: preset.weight_format || "bf16",
  };
  for (const f of [
    "architecture", "attn", "kv_model", "layers", "attention_kv_layers", "kv_heads",
    "head_dim", "kv_lora_rank", "qk_rope_head_dim", "kv_bytes_per_token_override",
    "kv_fixed_bytes_per_sequence", "num_experts", "active_experts", "attention_heads",
    "hidden_size", "total_params_b", "active_params_b", "max_context_tokens", "weight_format",
  ]) {
    if (state.model[f] !== null) state.provenance[f] = "preset";
  }
}

export function applyHardwareProfile(profile) {
  state.deployment.hardware_profile_id = profile.id;
  state.provenance["hardware_profile_id"] = "catalog";
}

export function applyConfigJson(config) {
  // HuggingFace config.json -> model fields. A config alone cannot establish
  // total parameters; that stays a manual override. Multimodal wrappers carry
  // architecture fields inside text_config; search it after top-level keys.
  const m = state.model;
  const text = config.text_config && typeof config.text_config === "object" ? config.text_config : {};
  const pick = (...keys) => {
    for (const k of keys) {
      if (config[k] !== undefined && config[k] !== null) return config[k];
      if (text[k] !== undefined && text[k] !== null) return text[k];
    }
    return undefined;
  };
  const archList = config.architectures || [];
  m.source = "config_json";
  m.preset_id = null;
  m.architecture = archList[0] ?? m.architecture;
  m.layers = pick("num_hidden_layers") ?? m.layers;
  m.kv_heads = pick("num_key_value_heads") ?? m.attention_heads ?? m.kv_heads;
  m.head_dim = pick("head_dim") ?? (pick("hidden_size") && pick("num_attention_heads") ? Math.floor(pick("hidden_size") / pick("num_attention_heads")) : m.head_dim);
  m.attention_heads = pick("num_attention_heads") ?? m.attention_heads;
  m.hidden_size = pick("hidden_size") ?? m.hidden_size;
  m.max_context_tokens = pick("max_position_embeddings", "model_max_length") ?? m.max_context_tokens;
  m.num_experts = pick("num_experts", "n_routed_experts") ?? m.num_experts;
  m.active_experts = pick("num_experts_per_tok") ?? m.active_experts;
  if (config.quantization_config?.quant_method) {
    m.quantization = config.quantization_config.quant_method;
    if (m.quantization.includes("fp8")) m.weight_format = "fp8";
    if (m.quantization.includes("fp4") || m.quantization.includes("nvfp4")) m.weight_format = "fp4";
  }
  m.attn = m.kv_heads < m.attention_heads ? "gqa" : "mha";
  m.kv_model = "mha_gqa";
  m.attention_kv_layers = null;
  m.kv_lora_rank = null;
  m.qk_rope_head_dim = null;
  m.kv_bytes_per_token_override = null;
  m.kv_override_source = null;
  m.kv_fixed_bytes_per_sequence = null;

  const arch = m.architecture || "";
  const MLA_ARCHS = ["DeepseekV2", "DeepseekV3", "GlmMoeDsa", "KimiK2", "KimiK25"];
  // Hybrid families: only some layers carry a growing KV cache; the rest hold
  // sliding-window or constant-state caches the config does not quantify.
  const HYBRID_ARCHS = ["DeepseekV4", "Glm5Next", "KimiK3", "Qwen3_5", "Qwen4Exp", "Inkling", "Gemma4"];
  state.configWarning = null;
  if (MLA_ARCHS.some((a) => arch.startsWith(a))) {
    m.attn = "mla";
    m.kv_lora_rank = pick("kv_lora_rank") ?? null;
    m.qk_rope_head_dim = pick("qk_rope_head_dim") ?? null;
    if (m.kv_lora_rank != null) {
      m.kv_model = "mla";
    } else {
      state.configWarning = "MLA architecture: config carries no kv_lora_rank — supply the MLA fields manually (Advanced).";
    }
  } else if (HYBRID_ARCHS.some((a) => arch.startsWith(a))) {
    m.attn = "hybrid";
    const layerTypes = pick("layer_types");
    if (Array.isArray(layerTypes)) {
      m.attention_kv_layers = layerTypes.filter((t) => String(t).startsWith("full")).length || null;
    }
    state.configWarning =
      "Hybrid architecture: only full-attention layers carry a growing KV cache. " +
      "Set attention KV layers and the per-sequence fixed KV term (or a KV override with its source) in Advanced — see docs/model-catalog-notes.md.";
  }
  // MoE needs no warning: supported since schema v2 (fields extracted, engine handles it).
  for (const f of ["architecture", "layers", "kv_heads", "head_dim", "attention_heads", "hidden_size", "max_context_tokens"]) {
    state.provenance[f] = "config.json";
  }
}