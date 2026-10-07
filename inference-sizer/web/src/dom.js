// DOM + fetch helpers. All user-entered strings are rendered with textContent
// (never interpolated into HTML) so injected customer text stays inert.

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = String(v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function labelWrap(labelText, input, hint) {
  const wrap = el("label", { class: "field" });
  wrap.append(el("span", { class: "field-label", text: labelText }), input);
  if (hint) wrap.append(el("span", { class: "field-hint", text: hint }));
  return wrap;
}

// Stale-response guard: only the latest request may paint results.
export function createSequencedFetcher() {
  let seq = 0;
  return async function fetchLatest(url, options) {
    const my = ++seq;
    const res = await fetch(url, options);
    if (my !== seq) throw new DOMException("stale response", "AbortError");
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        if (body.errors?.length) {
          msg = body.errors.map((e) => `${e.field}: ${e.message}`).join("; ");
        } else if (typeof body.detail === "string") {
          msg = body.detail;
        }
      } catch {
        /* keep default */
      }
      throw new Error(msg);
    }
    return res.json();
  };
}

export function fmt(n, d = 0) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
}

// Bytes -> explicit GB (decimal, source-style) and GiB (binary) strings.
export function bytesToGB(b) {
  return fmt(b / 1e9, 1);
}
export function bytesToGiB(b) {
  return fmt(b / 2 ** 30, 1);
}