import "./style.css";
import { FIELDS, applyGpu, applyPreset, formToRequest } from "./form.js";
import { renderResult } from "./render.js";

const app = document.querySelector("#app");
app.innerHTML = `
  <header>
    <div>
      <h1>Inference GPU sizer</h1>
      <p>Vite client over a Python sizing engine. Add a field in the form module and on SizeRequest, then use it in sizer/engine.py.</p>
    </div>
    <button id="export" type="button">Export JSON</button>
  </header>
  <main>
    <form id="f">
      <fieldset><legend>Model preset and GPU</legend>
        <div class="grid">
          <label class="wide">Preset<select name="preset" id="preset"></select></label>
          <label class="wide">GPU<select name="gpu" id="gpu"></select></label>
          <input type="hidden" name="gpu_name" value="B200 192GB" />
        </div>
      </fieldset>
      <fieldset><legend>Inputs</legend><div class="grid" id="fields"></div></fieldset>
      <fieldset><legend>Mode</legend>
        <label class="wide" style="flex-direction:row;align-items:center;gap:8px;color:inherit">
          <input name="disagg" type="checkbox" /> Disaggregated prefill / decode
        </label>
      </fieldset>
    </form>
    <section id="out"><p class="err">Start the API, then edit any field.</p></section>
  </main>`;

const form = document.querySelector("#f");
const fields = document.querySelector("#fields");
for (const [name, label, type, value, wide, options] of FIELDS) {
  const wrap = document.createElement("label");
  if (wide || name === "customer" || name === "tier") wrap.className = "wide";
  wrap.append(document.createTextNode(label));
  if (type === "select") {
    const sel = document.createElement("select");
    sel.name = name;
    for (const [v, text] of options) sel.add(new Option(text, v));
    sel.value = String(value);
    wrap.append(sel);
  } else {
    const input = document.createElement("input");
    input.name = name;
    input.type = type === "number" ? "number" : "text";
    input.step = "any";
    input.value = value;
    wrap.append(input);
  }
  fields.append(wrap);
}

let catalog = { presets: [], gpus: [] };
const presetSel = document.querySelector("#preset");
const gpuSel = document.querySelector("#gpu");

async function loadCatalog() {
  const res = await fetch("/api/catalog");
  if (!res.ok) throw new Error("catalog failed");
  catalog = await res.json();
  presetSel.innerHTML = "";
  gpuSel.innerHTML = "";
  catalog.presets.forEach((p, i) => presetSel.add(new Option(p.name, i)));
  catalog.gpus.forEach((g, i) => gpuSel.add(new Option(g.name, i)));
  presetSel.value = "1";
  gpuSel.value = "2";
  applyPreset(form, catalog.presets[1]);
  applyGpu(form, catalog.gpus[2]);
}

let timer;
async function run() {
  const out = document.querySelector("#out");
  try {
    const res = await fetch("/api/size", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(formToRequest(form)),
    });
    if (!res.ok) throw new Error(await res.text());
    out.innerHTML = renderResult(await res.json());
  } catch (err) {
    out.innerHTML = `<p class="err">API not reachable. From inference-sizer/: python -m uvicorn server:app --reload --port 8000. ${err.message}</p>`;
  }
}

form.addEventListener("input", (ev) => {
  if (ev.target.id === "preset") applyPreset(form, catalog.presets[presetSel.value]);
  if (ev.target.id === "gpu" || ev.target.name === "weight_bytes") {
    const gpu = catalog.gpus[gpuSel.value];
    if (gpu) applyGpu(form, gpu);
  }
  clearTimeout(timer);
  timer = setTimeout(run, 150);
});

document.querySelector("#export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(formToRequest(form), null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "inference-sizing.json";
  a.click();
});

loadCatalog().then(run).catch(run);
