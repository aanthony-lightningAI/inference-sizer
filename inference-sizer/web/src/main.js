// Lightning AI Inference Sizer — guided 4-step workflow over the versioned API.
// The form is fully remounted whenever state is replaced (preset, config.json,
// scenario import) so every input rebinding reads from state.

import "./theme/tokens.css";
import "./style.css";
import { el, labelWrap, createSequencedFetcher, fmt } from "./dom.js";
import { state, buildRequest, loadRequest, applyPreset, applyHardwareProfile, applyConfigJson } from "./state.js";
import { renderResult, renderComparison, badge as badgeEl } from "./render.js";

const fetchLatest = createSequencedFetcher();

let catalog = { presets: [], hardware_profiles: [], engines: [] };
const scenarios = []; // up to 3 for comparison: {name, request, result}
let lastResult = null;
let timer;
let seq = 0;

const app = document.querySelector("#app");
const fileInput = el("input", { type: "file", accept: "application/json", class: "visually-hidden" });
app.append(fileInput);

// ---------------------------------------------------------------- shell
app.append(
  el(
    "header",
    { class: "app-header" },
    el("img", {
      class: "wordmark",
      src: new URL("./theme/assets/lightning-wordmark-dark.svg", import.meta.url).href,
      alt: "Lightning AI",
    }),
    el(
      "div",
      { class: "header-text" },
      el("h1", { text: "Inference Sizer" }),
      el("p", {
        class: "muted",
        text: "Internal MLSE sizing tool — Estimated mode. Latency claims carry optimistic bounds until a compatible benchmark is imported.",
      })
    ),
    el(
      "div",
      { class: "header-actions" },
      el("button", { id: "btn-export", type: "button", text: "Export scenario" }),
      el("button", { id: "btn-import", type: "button", text: "Import scenario" }),
      el("button", { id: "btn-print", type: "button", text: "Print" })
    )
  )
);

const main = el("main", { class: "layout" });
const formCol = el("div", { class: "form-col" });
const resultCol = el("section", { class: "result-col", "aria-live": "polite" });
main.append(formCol, resultCol);
app.append(main);

const resultSlot = el("div", { id: "result" });
resultCol.append(el("h2", { text: "Results" }), resultSlot);

// ---------------------------------------------------------------- helpers
function showError(msg) {
  resultSlot.replaceChildren(
    el(
      "div",
      { class: "callout callout-err" },
      el("strong", { text: "Could not size this scenario." }),
      el("p", { text: msg })
    )
  );
}

function numInput(get, set, attrs = {}) {
  const input = el("input", { type: "number", step: "any", ...attrs });
  input.value = get() ?? "";
  input.addEventListener("input", () => {
    const v = input.value === "" ? null : Number(input.value);
    set(v);
    schedule();
  });
  return input;
}
function textInput(get, set) {
  const input = el("input", { type: "text", maxlength: 200 });
  input.value = get() ?? "";
  input.addEventListener("input", () => {
    set(input.value);
    schedule();
  });
  return input;
}
function select(options, get, set) {
  const sel = el("select");
  for (const [v, text] of options) sel.add(new Option(text, String(v)));
  sel.value = String(get() ?? "");
  sel.addEventListener("change", () => {
    set(sel.value);
    schedule();
  });
  return sel;
}

// ---------------------------------------------------------------- form
let presetSel, hwSel, benchStatus, derivedLine, provenanceLine, configWarnEl;

function mountForm() {
  formCol.replaceChildren();

  function step(num, title, subtitle) {
    const sec = el("section", { class: "step" });
    sec.append(
      el("h2", {}, el("span", { class: "step-num", text: String(num) }), el("span", { text: title })),
      el("p", { class: "muted step-sub", text: subtitle })
    );
    formCol.append(sec);
    return sec;
  }

  const m = state.model;
  const wl = state.workload;
  const dp = state.deployment;

  // --- Step 1: Model
  const stepModel = step(1, "Model", "Choose a preset, import a config.json, or enter architecture details.");
  presetSel = el("select", { id: "preset" });
  {
    const families = [];
    for (const p of catalog.presets) {
      const fam = p.family || "Other";
      let grp = families.find((f) => f.name === fam);
      if (!grp) { grp = { name: fam, presets: [] }; families.push(grp); }
      grp.presets.push(p);
    }
    const empty = document.createElement("option");
    empty.value = ""; empty.textContent = "Custom (manual entry)";
    presetSel.append(empty);
    for (const fam of families) {
      const og = document.createElement("optgroup");
      og.label = fam.name;
      for (const p of fam.presets) {
        const tag = p.spec_confidence === "unverified" ? " — specs unverified" : "";
        const bad = p.unsupported_reason ? " (unsupported)" : "";
        og.append(new Option(`${p.name}${bad}${tag}`, p.id));
      }
      presetSel.append(og);
    }
  }
  presetSel.value = m.preset_id ?? "";
  presetSel.addEventListener("change", () => {
    const p = catalog.presets.find((x) => x.id === presetSel.value);
    if (p) {
      state.configWarning = p.unsupported_reason || null;
      applyPreset(p);
    } else {
      state.model.source = "manual";
      state.model.preset_id = null;
    }
    refreshAll();
  });
  // Preset fact line: architecture badges (MoE / MLA / hybrid / override),
  // spec-confidence chip, preset notes.
  const presetInfo = el("div", { class: "preset-info" });
  function updatePresetInfo() {
    const p = catalog.presets.find((x) => x.id === m.preset_id);
    presetInfo.replaceChildren();
    if (!p) return;
    const badges = [];
    if (m.num_experts != null) badges.push(badgeEl("MoE", "info"));
    if (m.kv_model === "mla") badges.push(badgeEl("MLA cache", "info"));
    if (m.attn === "hybrid") badges.push(badgeEl("Hybrid attention", "info"));
    if (m.kv_model === "override") badges.push(badgeEl("KV override", "warn"));
    if (p.spec_confidence === "unverified") badges.push(badgeEl("Specs unverified", "warn"));
    presetInfo.append(...badges);
    for (const n of p.notes ?? []) {
      presetInfo.append(el("p", { class: "muted preset-note", text: n }));
    }
    if (m.kv_bytes_per_token_override != null) {
      presetInfo.append(
        el("p", { class: "muted preset-note", text: `KV cache: ${m.kv_bytes_per_token_override.toLocaleString()} B/token (override)` })
      );
    }
  }
  const importConfigBtn = el("button", { type: "button", text: "Import config.json…" });
  importConfigBtn.addEventListener("click", () => {
    fileInput.onchange = () => {
      const f = fileInput.files[0];
      if (!f) return;
      f.text().then((t) => {
        try {
          applyConfigJson(JSON.parse(t));
          refreshAll();
        } catch (e) {
          showError(`config.json could not be parsed: ${e.message}`);
        }
      });
    };
    fileInput.click();
  });

  provenanceLine = el("p", { class: "muted provenance", text: "" });
  configWarnEl = el("p", { class: "callout callout-warn", text: "" });
  if (!state.configWarning) configWarnEl.classList.add("visually-hidden");

  const modelFields = el("div", { class: "grid" });
  modelFields.append(
    labelWrap("Preset", presetSel),
    labelWrap(
      "Attention",
      select(
        [["gqa", "GQA"], ["mha", "MHA"], ["mla", "MLA"], ["mqa", "MQA"], ["hybrid", "Hybrid (subset of layers)"]],
        () => m.attn,
        (v) => (m.attn = v)
      )
    ),
    labelWrap(
      "KV cache path",
      select(
        [["mha_gqa", "MHA/GQA formula (2 x layers x heads x dim)"], ["mla", "MLA latent (rank + rope)"], ["override", "Published/derived figure"]],
        () => m.kv_model,
        (v) => (m.kv_model = v)
      ),
      "how KV bytes/token is computed"
    ),
    labelWrap("Layers", numInput(() => m.layers, (v) => (m.layers = v), { min: 1 })),
    labelWrap("KV heads", numInput(() => m.kv_heads, (v) => (m.kv_heads = v), { min: 1 })),
    labelWrap("Head dim", numInput(() => m.head_dim, (v) => (m.head_dim = v), { min: 1 })),
    labelWrap("Attention heads", numInput(() => m.attention_heads, (v) => (m.attention_heads = v), { min: 1 })),
    labelWrap("Hidden size", numInput(() => m.hidden_size, (v) => (m.hidden_size = v), { min: 1 })),
    labelWrap(
      "Total params (B)",
      numInput(() => m.total_params_b, (v) => (m.total_params_b = v), { min: 0.0001 }),
      "config.json cannot provide this"
    ),
    labelWrap("Max context tokens", numInput(() => m.max_context_tokens, (v) => (m.max_context_tokens = v), { min: 1 })),
    labelWrap("Checkpoint revision", textInput(() => m.checkpoint_revision, (v) => (m.checkpoint_revision = v))),
    labelWrap(
      "Weight format",
      select(
        [["bf16", "BF16 (2 B/weight)"], ["fp8", "FP8 (1 B/weight)"], ["fp4", "FP4 (0.5 B/weight)"]],
        () => m.weight_format,
        (v) => (m.weight_format = v)
      )
    ),
    labelWrap(
      "KV dtype",
      select(
        [["bf16", "BF16 (2 B)"], ["fp8", "FP8 (1 B)"], ["fp4", "FP4 (0.5 B)"]],
        () => m.kv_dtype,
        (v) => (m.kv_dtype = v)
      )
    ),
    labelWrap("Quantization", textInput(() => m.quantization, (v) => (m.quantization = v)))
  );
  const modelAdvanced = el("details", { class: "advanced" });
  modelAdvanced.append(
    el("summary", { text: "Advanced: MLA / MoE / hybrid-attention fields" }),
    el(
      "div",
      { class: "grid" },
      labelWrap("Active params (B, optional)", numInput(() => m.active_params_b, (v) => (m.active_params_b = v), { min: 0.0001 }), "prefill compute uses this; MoE models activate a subset"),
      labelWrap("Routed experts", numInput(() => m.num_experts, (v) => (m.num_experts = v), { min: 1 })),
      labelWrap("Active experts / token", numInput(() => m.active_experts, (v) => (m.active_experts = v), { min: 1 })),
      labelWrap("KV-carrying layers", numInput(() => m.attention_kv_layers, (v) => (m.attention_kv_layers = v), { min: 1 }), "full-attention/MLA layers; blank = all layers"),
      labelWrap("MLA kv_lora_rank", numInput(() => m.kv_lora_rank, (v) => (m.kv_lora_rank = v), { min: 1 })),
      labelWrap("MLA qk_rope_head_dim", numInput(() => m.qk_rope_head_dim, (v) => (m.qk_rope_head_dim = v), { min: 0 })),
      labelWrap("KV override (B/token)", numInput(() => m.kv_bytes_per_token_override, (v) => (m.kv_bytes_per_token_override = v), { min: 1 }), "requires a source below"),
      labelWrap("KV override source", textInput(() => m.kv_override_source, (v) => (m.kv_override_source = v)), "where the figure comes from"),
      labelWrap("Fixed KV per sequence (B)", numInput(() => m.kv_fixed_bytes_per_sequence, (v) => (m.kv_fixed_bytes_per_sequence = v), { min: 0 }), "sliding-window caps / linear-attention state")
    )
  );
  stepModel.append(
    el("div", { class: "row" }, importConfigBtn),
    provenanceLine,
    modelFields,
    modelAdvanced,
    presetInfo,
    configWarnEl
  );

  // --- Step 2: Workload
  const stepWorkload = step(2, "Workload", "Token lengths, demand and latency targets.");
  derivedLine = el("p", { class: "muted provenance", text: "" });
  const workloadFields = el("div", { class: "grid" });
  workloadFields.append(
    labelWrap("Input tokens, mean", numInput(() => wl.mean_input_tokens, (v) => (wl.mean_input_tokens = v), { min: 0 })),
    labelWrap("Input tokens, p95", numInput(() => wl.p95_input_tokens, (v) => (wl.p95_input_tokens = v), { min: 0 })),
    labelWrap("Output tokens, mean", numInput(() => wl.mean_output_tokens, (v) => (wl.mean_output_tokens = v), { min: 0 })),
    labelWrap("Output tokens, p95", numInput(() => wl.p95_output_tokens, (v) => (wl.p95_output_tokens = v), { min: 0 })),
    labelWrap("Peak requests / sec (0 = none)", numInput(() => wl.peak_rps, (v) => (wl.peak_rps = v), { min: 0 })),
    labelWrap("Peak simultaneous generations (0 = unspecified)", numInput(() => wl.peak_inflight, (v) => (wl.peak_inflight = v), { min: 0 })),
    labelWrap("TTFT SLO p95 (ms)", numInput(() => wl.ttft_ms_p95, (v) => (wl.ttft_ms_p95 = v), { min: 0.1 })),
    labelWrap("Decode SLO p95 (ms)", numInput(() => wl.decode_ms_p95, (v) => (wl.decode_ms_p95 = v), { min: 0.1 })),
    labelWrap(
      "Decode metric",
      select(
        [["itl", "ITL (inter-token latency)"], ["tpot", "TPOT (time per output token)"]],
        () => wl.decode_metric,
        (v) => (wl.decode_metric = v)
      )
    ),
    labelWrap("Prefix-cache hit rate (0–1)", numInput(() => wl.prefix_hit_rate, (v) => (wl.prefix_hit_rate = v), { min: 0, max: 1 })),
    labelWrap("Active users (optional)", numInput(() => wl.active_users, (v) => (wl.active_users = v), { min: 0 })),
    labelWrap("Requests per active user / hour (optional)", numInput(() => wl.requests_per_active_per_hour, (v) => (wl.requests_per_active_per_hour = v), { min: 0 })),
    labelWrap("Model calls per task (optional)", numInput(() => wl.calls_per_task, (v) => (wl.calls_per_task = v), { min: 0 }))
  );
  stepWorkload.append(derivedLine, workloadFields);

  // --- Step 3: Deployment
  const stepDeploy = step(3, "Deployment", "Hardware profile, engine, formats, parallelism and spare policy.");
  hwSel = el("select", { id: "hardware" });
  catalog.hardware_profiles.forEach((p) => {
    const tag = p.recommendation_mode === "comparison_only" ? " (comparison only)" : "";
    hwSel.add(new Option(`${p.name}${tag}`, p.id));
  });
  hwSel.value = dp.hardware_profile_id;
  hwSel.addEventListener("change", () => {
    const p = catalog.hardware_profiles.find((x) => x.id === hwSel.value);
    if (p) applyHardwareProfile(p);
    refreshAll();
  });
  const hwStatus = el("p", { class: "muted provenance", text: "" });
  const engineSel = select(
    [["vllm", "vLLM (benchmark-eligible)"], ["other", "Other engine (not benchmark-eligible)"]],
    () => dp.engine.name,
    (v) => (dp.engine.name = v)
  );
  const deployFields = el("div", { class: "grid" });
  deployFields.append(
    labelWrap("Hardware profile", hwSel),
    labelWrap("Engine", engineSel),
    labelWrap("Engine version", textInput(() => dp.engine.version, (v) => (dp.engine.version = v))),
    labelWrap("Max GPUs per replica", numInput(() => dp.max_gpus_per_replica, (v) => (dp.max_gpus_per_replica = v), { min: 1 })),
    labelWrap("Spare replicas", numInput(() => dp.spare_replicas, (v) => (dp.spare_replicas = v), { min: 0 })),
    labelWrap(
      "Operating factor (0–1]",
      numInput(() => dp.operating_factor, (v) => (dp.operating_factor = v), { min: 0.0001, max: 1 }),
      "fraction of capacity counted for sizing"
    ),
    labelWrap(
      "Observed device memory (GB, optional)",
      numInput(() => dp.observed_memory_gb, (v) => (dp.observed_memory_gb = v), { min: 0.1 }),
      "overrides the published nominal value"
    ),
    labelWrap(
      "Provider allocation size (GPUs, optional)",
      numInput(() => dp.allocation_size_gpus, (v) => (dp.allocation_size_gpus = v), { min: 1 }),
      "kept distinct from GPUs per replica"
    )
  );
  const advanced = el("details", { class: "advanced" });
  advanced.append(
    el("summary", { text: "Advanced: efficiencies and overheads (optimistic-bound controls)" }),
    el(
      "div",
      { class: "grid" },
      labelWrap("Bandwidth efficiency (0–1]", numInput(() => dp.bw_eff, (v) => (dp.bw_eff = v), { min: 0.0001, max: 1 })),
      labelWrap("Compute efficiency (0–1]", numInput(() => dp.flops_eff, (v) => (dp.flops_eff = v), { min: 0.0001, max: 1 })),
      labelWrap("Runtime overhead (GB/device)", numInput(() => dp.runtime_overhead_gb, (v) => (dp.runtime_overhead_gb = v), { min: 0 })),
      labelWrap("CUDA graphs (GB/device)", numInput(() => dp.cuda_graphs_gb, (v) => (dp.cuda_graphs_gb = v), { min: 0 })),
      labelWrap("Activations/workspace (GB/device)", numInput(() => dp.activations_workspace_gb, (v) => (dp.activations_workspace_gb = v), { min: 0 })),
      labelWrap("Reserve fraction (0–0.5)", numInput(() => dp.reserve_fraction, (v) => (dp.reserve_fraction = v), { min: 0, max: 0.5 }))
    )
  );
  const metaFields = el("div", { class: "grid" });
  metaFields.append(
    labelWrap("Customer", textInput(() => state.customer, (v) => (state.customer = v))),
    labelWrap("Workload tier", textInput(() => state.tier, (v) => (state.tier = v)))
  );
  stepDeploy.append(hwStatus, deployFields, advanced, metaFields);

  // --- Step 4: Results controls
  const stepResults = step(4, "Results", "Calculate, compare up to three scenarios, export or print.");
  const benchBtn = el("button", { type: "button", text: "Import benchmark JSON…" });
  benchStatus = el("p", {
    class: "muted provenance",
    text: state.benchmark
      ? `Benchmark ${state.benchmark.profile_id} imported (${state.benchmark.source}).`
      : "No benchmark imported — results stay Estimated.",
  });
  benchBtn.addEventListener("click", () => {
    fileInput.onchange = () => {
      const f = fileInput.files[0];
      if (!f) return;
      f.text().then((t) => {
        try {
          const prof = JSON.parse(t);
          state.benchmark = prof;
          refreshAll();
        } catch (e) {
          showError(`Benchmark JSON could not be parsed: ${e.message}`);
        }
      });
    };
    fileInput.click();
  });
  const compareAddBtn = el("button", { type: "button", text: "Add current result to comparison" });
  compareAddBtn.addEventListener("click", () => {
    if (!lastResult) return;
    if (scenarios.length >= 3) {
      showError("Comparison holds up to three scenarios; clear one first.");
      return;
    }
    scenarios.push({
      name: `${state.customer} · ${catalog.hardware_profiles.find((p) => p.id === dp.hardware_profile_id)?.name ?? ""}`,
      result: lastResult,
    });
    compareSlot.replaceChildren(renderComparison(scenarios));
  });
  const compareClearBtn = el("button", { type: "button", text: "Clear comparison" });
  compareClearBtn.addEventListener("click", () => {
    scenarios.length = 0;
    compareSlot.replaceChildren(renderComparison(scenarios));
  });
  const compareSlot = el("div", { id: "compare" });
  stepResults.append(
    el("div", { class: "row" }, benchBtn),
    benchStatus,
    el("div", { class: "row" }, compareAddBtn, compareClearBtn),
    compareSlot
  );

  function updateProvenance() {
    updatePresetInfo();
    const src = { preset: "preset", config_json: "config.json", manual: "manual" }[m.source] ?? m.source;
    const fromConfig = Object.entries(state.provenance)
      .filter(([, v]) => v === "config.json")
      .map(([k]) => k)
      .join(", ");
    const missing = [];
    if (!m.total_params_b) missing.push("total parameters (config.json cannot establish these — enter manually)");
    provenanceLine.textContent =
      `Model source: ${src}.` +
      (fromConfig ? ` Fields from config.json: ${fromConfig}.` : "") +
      (missing.length ? ` Missing: ${missing.join("; ")}.` : "");
    if (state.configWarning) {
      configWarnEl.textContent = state.configWarning;
      configWarnEl.classList.remove("visually-hidden");
    } else {
      configWarnEl.classList.add("visually-hidden");
    }
    const p = catalog.hardware_profiles.find((x) => x.id === dp.hardware_profile_id);
    if (p) {
      const s = p.status;
      const mode = p.recommendation_mode === "comparison_only" ? " · comparison only" : "";
      hwStatus.textContent =
        `${p.name}: ${p.device_count ?? "system-configured"} devices × ${p.memory_per_device_gb} GB (${p.memory_qualifier}), ` +
        `${p.hbm_bandwidth_tb_s} TB/s (${p.bandwidth_qualifier}). Inventory: ${s.inventory}. Engine qualified: ${s.engine_qualified ? "yes" : "no"}. ` +
        `Benchmark evidence: ${s.benchmark_evidence}.${mode}` +
        (s.notes?.length ? ` ${s.notes[0]}` : "");
    }
    if (wl.active_users && wl.requests_per_active_per_hour) {
      const calls = wl.calls_per_task ?? 1;
      const rps = (wl.active_users * wl.requests_per_active_per_hour * calls) / 3600;
      derivedLine.textContent =
        `Derived demand: ${fmt(rps, 3)} rps = ${wl.active_users} active users × ${wl.requests_per_active_per_hour} requests/h × ${calls} calls/task ÷ 3600. ` +
        `Leave peak rps at 0 to use the derived rate.`;
    } else {
      derivedLine.textContent = "Optional: enter active users and cadence to derive request rate.";
    }
    if (state.benchmark) {
      benchStatus.textContent = state.benchmark.synthetic
        ? "Synthetic fixture imported: excluded from real calibration evidence; results stay Estimated."
        : `Benchmark ${state.benchmark.profile_id} imported (${state.benchmark.source}, collected ${state.benchmark.collected_at}). Compatible profiles calibrate this scenario.`;
    }
  }
  updateProvenance();
}

// ---------------------------------------------------------------- pipeline
async function calculate() {
  const my = ++seq;
  resultSlot.replaceChildren(el("p", { class: "muted", text: "Calculating…" }));
  try {
    const result = await fetchLatest("/api/size", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(buildRequest()),
    });
    if (my !== seq) return; // stale response: never overwrite newer results
    lastResult = result;
    resultSlot.replaceChildren(renderResult(result));
  } catch (e) {
    if (e.name === "AbortError" || my !== seq) return;
    showError(
      String(e.message).includes("Failed to fetch")
        ? "The sizing service is not reachable. Your inputs are preserved; results will appear when the service is back."
        : e.message
    );
  }
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(calculate, 250);
}

function refreshAll() {
  mountForm();
  calculate();
}

// ---------------------------------------------------------------- header actions
document.querySelector("#btn-import").addEventListener("click", () => {
  fileInput.onchange = () => {
    const f = fileInput.files[0];
    if (!f) return;
    f.text().then((t) => {
      try {
        loadRequest(JSON.parse(t));
        refreshAll();
      } catch (e) {
        showError(`Scenario JSON could not be imported: ${e.message}`);
      }
    });
  };
  fileInput.click();
});

document.querySelector("#btn-export").addEventListener("click", () => {
  if (!lastResult) {
    showError("Calculate a scenario before exporting.");
    return;
  }
  const exportObj = {
    export_schema_version: 1,
    calculator_version: lastResult.calculator_version,
    request: buildRequest(),
    result: lastResult,
    benchmark_export: state.benchmark
      ? { profile: state.benchmark, profile_id: state.benchmark.profile_id }
      : null,
  };
  const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: "inference-sizing-scenario.json" });
  a.click();
  URL.revokeObjectURL(a.href);
});

document.querySelector("#btn-print").addEventListener("click", () => window.print());

// ---------------------------------------------------------------- boot
(async function boot() {
  try {
    catalog = await fetch("/api/catalog").then((r) => {
      if (!r.ok) throw new Error(`catalog HTTP ${r.status}`);
      return r.json();
    });
    mountForm();
    presetSel.focus();
    calculate();
  } catch (e) {
    showError(`The catalog service is not reachable (${e.message}). Your inputs are preserved; results will appear when the service is back.`);
  }
})();