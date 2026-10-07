function n(v, d = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
}

function table(rows) {
  return `<table><tbody>${rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("")}</tbody></table>`;
}

export function renderResult(data) {
  const notes = (data.notes || []).map((w) => `<div class="warn">${w}</div>`).join("");
  const coloc = data.colocated?.replica;
  const dis = data.disaggregated;
  let body = `<div class="kpis">
    <div class="kpi"><span>Fleet GPUs</span><strong>${n(data.fleet_gpus)}</strong><em>${data.mode} · ${data.gpu}</em></div>
    <div class="kpi"><span>Weight footprint</span><strong>${n(data.weight_gib, 1)}</strong><em>GiB</em></div>
    <div class="kpi"><span>KV per token</span><strong>${n(data.kv_bytes_per_token / 1024, 1)}</strong><em>KiB</em></div>
    <div class="kpi"><span>Peak output</span><strong>${n(data.peak_output_tok_s)}</strong><em>tokens / sec</em></div>
  </div>
  <p>${data.customer} · ${data.tier}</p>${notes}`;
  body += `<h2>Colocated replica</h2>`;
  body += coloc
    ? table([
        ["GPUs per replica", n(coloc.gpus_per_replica)],
        ["Concurrency inside ITL and p95 KV", n(coloc.concurrency)],
        ["Goodput at mean context", `${n(coloc.goodput_tok_s)} tok/s`],
        ["Prefill time, mean uncached input", `${n(coloc.prefill_ms)} ms${coloc.ttft_ok ? "" : " · over TTFT SLO"}`],
        ["Replicas", n(coloc.replicas)],
        ["Fleet including margin", n(coloc.fleet_gpus)],
        ["p95 KV per sequence", `${n(coloc.kv_per_seq_gib, 2)} GiB`],
      ])
    : `<div class="warn">No feasible colocated replica.</div>`;
  if (dis) {
    body += `<h2>Disaggregated pools</h2>`;
    if (dis.decode && dis.prefill) {
      body += table([
        ["Decode GPUs per replica", n(dis.decode.gpus_per_replica)],
        ["Decode concurrency", n(dis.decode.concurrency)],
        ["Decode goodput", `${n(dis.decode.goodput_tok_s)} tok/s`],
        ["Decode fleet GPUs", n(dis.decode.fleet_gpus)],
        ["Prefill GPUs per replica", n(dis.prefill.gpus_per_replica)],
        ["Prefill goodput", `${n(dis.prefill.goodput_input_tok_s)} input tok/s`],
        ["Prefill fleet GPUs", n(dis.prefill.fleet_gpus)],
        ["KV transfer, p95", `${n(dis.prefill.kv_transfer_ms)} ms`],
        ["TTFT estimate", `${n(dis.prefill.ttft_ms)} ms`],
        ["Fleet total", n(dis.fleet_gpus)],
      ]);
    }
  }
  body += `<h2>Assumptions</h2><p>${(data.assumptions || []).join(" ")}</p>`;
  return body;
}
