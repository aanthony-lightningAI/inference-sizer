// Results rendering. Built exclusively with DOM nodes + textContent: user
// strings (customer, tier, imported text) are never interpolated into HTML.

import { el, fmt, bytesToGB, bytesToGiB } from "./dom.js";

export function badge(text, kind) {
  return el("span", { class: `badge badge-${kind}`, text });
}

export function statusRow(result) {
  const ev = result.evidence_status === "benchmark_calibrated"
    ? badge("Benchmark calibrated", "ok")
    : badge("Estimated", "info");
  const feas =
    result.feasibility === "feasible"
      ? badge("Feasible", "ok")
      : result.feasibility === "unsupported"
        ? badge("Unsupported", "muted")
        : badge("Infeasible", "err");
  const latency = result.latency_bound_only
    ? badge("Latency: optimistic bounds only", "warn")
    : badge("Latency: benchmark measured", "ok");
  const arch = result.architecture_status === "unsupported" ? badge("Architecture unsupported", "muted") : null;
  return el("div", { class: "status-row" }, ev, feas, latency, arch);
}

function kv(label, value, sub) {
  return el(
    "div",
    { class: "kpi" },
    el("span", { class: "kpi-label", text: label }),
    el("strong", { class: "kpi-value", text: value }),
    sub ? el("em", { class: "kpi-sub", text: sub }) : null
  );
}

function table(rows) {
  const t = el("table", { class: "result-table" }, el("tbody"));
  for (const [k, v, cls] of rows) {
    const tr = el("tr", {}, el("th", { text: k }));
    const td = el("td", { text: v });
    if (cls) td.className = cls;
    tr.append(td);
    t.firstChild.append(tr);
  }
  return t;
}

function memoryTable(result) {
  const mc = result.memory_components;
  if (!mc) return null;
  const sel = result.selected;
  return [
    el("h3", { text: "Memory per device (tail-envelope context)" }),
    table([
      ["Weights (sharded)", `${bytesToGB(mc.weights_bytes)} GB · ${bytesToGiB(mc.weights_bytes)} GiB`],
      [
        `KV @ p95 envelope (${fmt(sel.max_concurrency)} concurrent)`,
        `${bytesToGB(sel.memory.kv_p95_bytes_per_replica / sel.gpus_per_replica)} GB · ${bytesToGiB(sel.memory.kv_p95_bytes_per_replica / sel.gpus_per_replica)} GiB`,
      ],
      ...(mc.kv_fixed_bytes_per_replica > 0
        ? [[
            `KV fixed / sequence (window caps, state)`,
            `${bytesToGB(sel.memory.kv_fixed_bytes_per_replica / sel.gpus_per_replica)} GB · ${bytesToGiB(sel.memory.kv_fixed_bytes_per_replica / sel.gpus_per_replica)} GiB`,
          ]]
        : []),
      ["Runtime overhead", `${bytesToGB(mc.runtime_overhead_bytes)} GB`],
      ["CUDA graphs", `${bytesToGB(mc.cuda_graphs_bytes)} GB`],
      ["Activations/workspace", `${bytesToGB(mc.activations_workspace_bytes)} GB`],
      ["Reserve", `${bytesToGB(mc.reserve_bytes)} GB`],
      ["Total per device", `${bytesToGB(mc.total_bytes_per_device)} GB · ${bytesToGiB(mc.total_bytes_per_device)} GiB`, "strong"],
      [
        `Device memory (${mc.device_memory_source})`,
        `${bytesToGB(mc.device_memory_bytes)} GB · ${bytesToGiB(mc.device_memory_bytes)} GiB`,
      ],
      ["KV bytes / token / sequence", `${fmt(sel.kv_per_token_bytes)} B`],
      [
        "KV placement",
        sel.kv_sharding === "replicated"
          ? "replicated across TP (MLA/override single stream)"
          : `sharded (${sel.kv_heads_per_device} heads/device)`,
      ],
    ]),
  ];
}

function fleetTable(result) {
  return [
    el("h3", { text: "Fleet" }),
    table([
      ["GPUs per replica", fmt(result.selected.gpus_per_replica)],
      ["Serving replicas", fmt(result.serving_replicas)],
      ["Spare replicas (excluded from serving)", fmt(result.spare_replicas)],
      ["Fleet GPUs", `${fmt(result.fleet_gpus)}`, "strong"],
      ["Allocation size", result.allocation_size_gpus ? `${fmt(result.allocation_size_gpus)} GPUs` : "provider default / unspecified"],
      ["Capacity (rps, with operating factor)", result.capacity_rps === null ? "—" : fmt(result.capacity_rps, 2)],
      ["Capacity (concurrent generations)", result.capacity_concurrency === null ? "—" : fmt(result.capacity_concurrency)],
      ["Limiting factors", (result.limiting_factors || []).join("; ") || "—"],
      ["Optimistic TTFT bound", result.selected.prefill_bound_ms === null ? "unverified (no compute data)" : `${fmt(result.selected.prefill_bound_ms, 1)} ms`],
      ["Optimistic decode bound", `${fmt(result.selected.itl_bound_ms, 2)} ms`],
      ["Latency claim", result.latency_bound_only ? "bounds only — p95 SLO NOT verified" : "measured (benchmark)"],
    ]),
  ];
}

function listSection(title, items, cls) {
  if (!items?.length) return null;
  return el(
    "div",
    { class: `section ${cls || ""}` },
    el("h3", { text: title }),
    ...items.map((t) => el("p", { class: cls ? `${cls}-item` : "note-item", text: t }))
  );
}

export function renderResult(result) {
  const out = el("div", { class: "result" });
  out.append(statusRow(result));

  const summary = el("div", { class: "kpis" });
  if (result.selected) {
    summary.append(
      kv("Fleet GPUs", fmt(result.fleet_gpus), `${result.selected.gpus_per_replica} per replica`),
      kv("Serving replicas", fmt(result.serving_replicas), `${fmt(result.spare_replicas)} spare`),
      kv("Concurrency / replica", fmt(result.selected.max_concurrency), `ITL+memory bound`),
      kv("Capacity", `${fmt(result.capacity_rps, 1)} rps`, `${fmt(result.capacity_concurrency)} concurrent`)
    );
  } else {
    summary.append(kv("Fleet GPUs", "—", result.feasibility));
  }
  out.append(summary);
  out.append(el("p", { class: "customer-line" }, el("strong", { text: result.customer }), ` · ${result.tier} · ${result.hardware_profile?.name ?? ""}`));

  if (!result.selected) {
    out.append(el("div", { class: "callout callout-err" }, el("strong", { text: "No feasible candidate." })));
    (result.rejection_reasons || []).forEach((r) => out.append(el("p", { class: "reason", text: r })));
  } else {
    out.append(...memoryTable(result));
    out.append(...fleetTable(result));
  }

  out.append(listSection("Warnings", result.warnings, "warn"));
  out.append(listSection("Rejection reasons", result.selected ? null : null)); // rendered above when infeasible
  if (result.selected) out.append(listSection("Candidate rejections", (result.rejection_reasons || []).slice(0, 8), "reason"));
  out.append(listSection("Notes", result.notes, "note"));
  out.append(listSection("Assumptions", result.assumptions, "assumption"));

  if (result.benchmark) {
    const b = result.benchmark;
    out.append(
      el("div", { class: "section benchmark" },
        el("h3", { text: "Benchmark evidence" }),
        el("p", { text: `${b.profile_id} · ${b.source} · collected ${b.collected_at} · ${b.engine} on ${b.hardware_sku} (${b.gpus_per_replica} GPUs/replica)` }),
        el("p", { text: `Measured: ${fmt(b.metrics.sustainable_rps, 2)} rps at ${fmt(b.metrics.sustainable_concurrency)} concurrent; p95 TTFT ${fmt(b.metrics.ttft_ms_p95, 1)} ms; ${b.metrics.decode_metric} p95 ${fmt(b.metrics.decode_ms_p95, 1)} ms; SLO miss ${fmt(b.metrics.slo_miss_rate * 100, 2)}%; ${b.metrics.completion_rate * 100}% completed over ${fmt(b.duration_s)} s / ${fmt(b.request_count)} requests.` })
      )
    );
  }
  return out;
}

export function renderComparison(results) {
  if (!results.length) return el("p", { class: "muted", text: "No scenarios compared yet." });
  const cols = results.map((s) => s.result);
  const row = (label, fn) => el("tr", {}, el("th", { text: label }), ...cols.map((c) => el("td", { text: fn(c) })));
  return el(
    "table",
    { class: "result-table compare" },
    el(
      "thead",
      {},
      el("tr", {}, el("th", { text: "Metric" }), ...results.map((s) => el("th", { text: s.name })))
    ),
    el(
      "tbody",
      {},
      row("Evidence", (c) => c.evidence_status === "benchmark_calibrated" ? "calibrated" : "estimated"),
      row("Feasibility", (c) => c.feasibility),
      row("Hardware", (c) => c.hardware_profile?.name ?? "—"),
      row("GPUs per replica", (c) => c.selected ? fmt(c.selected.gpus_per_replica) : "—"),
      row("Serving replicas", (c) => fmt(c.serving_replicas)),
      row("Spare replicas", (c) => fmt(c.spare_replicas)),
      row("Fleet GPUs", (c) => fmt(c.fleet_gpus)),
      row("Capacity rps", (c) => c.capacity_rps === null ? "—" : fmt(c.capacity_rps, 1)),
      row("Concurrency", (c) => c.capacity_concurrency === null ? "—" : fmt(c.capacity_concurrency)),
      row("Limiting factor", (c) => (c.limiting_factors || []).join("; ") || "—")
    )
  );
}