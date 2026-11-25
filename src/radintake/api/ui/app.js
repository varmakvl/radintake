"use strict";
// Plain JS, same-origin requests only. Boxes arrive in ORIGINAL page pixels; the masked page image has
// that exact size, so a box is scaled by (displayed width / natural width) and nothing else.
const $ = (id) => document.getElementById(id);
const state = {caseId: null, page: 1, doc: null, cited: []};

function show(view) {
  for (const v of ["upload", "case", "queue", "query"]) $("view-" + v).hidden = v !== view;
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  if (view === "queue") loadQueue();
}
window.addEventListener("hashchange", () => show((location.hash || "#upload").slice(1)));

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.title || ("HTTP " + r.status));
  return body;
}

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = $("file").files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  $("upload-msg").textContent = "Uploading...";
  try {
    const res = await api("/cases", {method: "POST", body: fd});
    $("upload-msg").textContent = "Accepted: " + res.case_id;
    openCase(res.case_id);
  } catch (err) { $("upload-msg").textContent = "Upload failed: " + err.message; }
});

async function openCase(id, page, cited) {
  state.caseId = id; state.page = page || 1; state.cited = cited || [];
  location.hash = "#case"; show("case");
  $("case-id").textContent = id;
  for (;;) {
    let c;
    try { c = await api("/cases/" + encodeURIComponent(id)); } catch (err) { $("case-status").textContent = err.message; return; }
    $("case-status").textContent = c.status;
    $("case-flags").replaceChildren(...(c.flags || []).map((f) => Object.assign(document.createElement("span"), {textContent: f})));
    if (!["pending", "running"].includes(c.status)) { state.doc = c.result; renderCase(); return; }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

function renderCase() {
  const d = state.doc || {};
  const pager = $("pager");
  pager.replaceChildren();
  for (let n = 1; n <= (d.n_pages || 1); n++) {
    const b = Object.assign(document.createElement("button"), {textContent: "Page " + n, type: "button"});
    b.onclick = () => { state.page = n; renderCase(); };
    pager.append(b);
  }
  const rep = d.report || {};
  $("case-fields").textContent = Object.entries(rep).filter(([, f]) => f && f.value)
    .map(([k, f]) => k + (f.needs_review ? " [needs review]" : "") + ": " + f.value).join("\n");
  const img = $("page-img");
  img.onload = drawBoxes;
  img.src = "/cases/" + encodeURIComponent(state.caseId) + "/pages/" + state.page + ".png";
  if (img.complete && img.naturalWidth) drawBoxes();
}

function box(cls, b, title) {
  const img = $("page-img");
  const s = img.clientWidth / img.naturalWidth;
  const el = document.createElement("div");
  el.className = "box " + cls;
  el.title = title || "";
  el.style.left = b[0] * s + "px"; el.style.top = b[1] * s + "px";
  el.style.width = (b[2] - b[0]) * s + "px"; el.style.height = (b[3] - b[1]) * s + "px";
  return el;
}

function drawBoxes() {
  const d = state.doc || {}, p = state.page, out = [];
  for (const [name, f] of Object.entries(d.report || {}))
    for (const pr of (f && f.provenance) || []) if (pr.page === p) out.push(box("field", pr.bbox, name));
  for (const det of (d.phi && d.phi.detections) || []) if (det.page === p && det.bbox) out.push(box("phi", det.bbox, det.type));
  for (const t of d.tables || []) if (t.page === p) out.push(box("table", t.bbox, "table"));
  for (const c of state.cited) if (c.page === p) out.push(box("cite", c.bbox, "cited: " + c.field));
  $("overlay").replaceChildren(...out);
}

async function loadQueue() {
  const body = $("queue-table").querySelector("tbody");
  body.replaceChildren();
  try {
    const q = await api("/review-queue");
    for (const it of q.items) {
      const tr = document.createElement("tr");
      const a = Object.assign(document.createElement("a"), {textContent: it.case_id, href: "#case"});
      a.onclick = () => openCase(it.case_id);
      const c0 = document.createElement("td"); c0.append(a);
      tr.append(c0);
      for (const t of [it.input, (it.reasons || []).join(", "), (it.fields || []).join(", ")])
        tr.append(Object.assign(document.createElement("td"), {textContent: t}));
      body.append(tr);
    }
  } catch (err) { body.append(Object.assign(document.createElement("tr"), {textContent: err.message})); }
}
$("queue-refresh").onclick = loadQueue;

$("query-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const box_ = $("answer");
  box_.textContent = "Searching...";
  try {
    const r = await api("/query", {method: "POST", headers: {"Content-Type": "application/json"},
                                   body: JSON.stringify({question: $("question").value})});
    box_.replaceChildren();
    const head = Object.assign(document.createElement("p"), {id: "answer-text",
      textContent: r.abstained ? "Abstained: " + (r.reason || "no supported answer") : (r.answer || "")});
    box_.append(head);
    for (const c of r.citations || []) {
      const div = document.createElement("div");
      div.className = "cite-item";
      div.append(Object.assign(document.createElement("div"), {className: "citation",
        textContent: c.case_id + " / " + c.field + " / page " + c.page + ": " + (c.quote || "")}));
      const b = Object.assign(document.createElement("button"), {type: "button", textContent: "Show cited box"});
      b.onclick = () => openCase(c.case_id, c.page, [c]);
      div.append(b);
      box_.append(div);
    }
  } catch (err) { box_.textContent = "Query failed: " + err.message; }
});

show((location.hash || "#upload").slice(1));
