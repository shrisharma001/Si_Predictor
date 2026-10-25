/* Hot-metal silicon interface: replay of held-out casts, prediction from uploaded files, and training a model for
   another furnace. No dependencies. Column names from uploaded files are shown with textContent only. */
"use strict";

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  measured: [],      // {t: Date, v}
  predicted: [],     // {t: Date, v, lo, hi, status, messages, actual}
  xDomain: null, yMax: null, typical: null, level: 0.9,
  playing: null, replay: null, focusIndex: null,
};

/* ---------------------------------------------------------------- formatting */
const fmtNum = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const fmtTime = (t) => t.toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
const fmtDay = (t) => t.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
const fmtHour = (t) => t.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
const toLocalInput = (iso) => (iso ? iso.slice(0, 16) : "");
const parseT = (iso) => new Date(iso);

function el(tag, attrs = {}, parent = null) {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function svgText(parent, x, y, text, cls, anchor = "start") {
  const t = el("text", { x, y, class: cls, "text-anchor": anchor }, parent);
  t.textContent = text;
  return t;
}

/* ---------------------------------------------------------------- errors */
function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }
async function readJson(resp) {
  let body = null;
  try { body = await resp.json(); } catch (_) { /* not JSON */ }
  if (!resp.ok && !(body && body.results)) {
    let msg = (body && body.error && body.error.message) || `The server answered ${resp.status}.`;
    const d = body && body.error && body.error.details;
    if (d && d.missing) msg += `: ${d.missing.join(", ")}`;
    throw new Error(msg);
  }
  return body;
}

/* ---------------------------------------------------------------- readout */
const STATUS_TEXT = { ok: "Inputs look normal", warning: "Prediction made, with warnings", no_prediction: "No prediction", idle: "Waiting for data" };
const STATUS_ICON = {
  ok: "M10 1.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17Zm-1.2 12.3-3.6-3.6 1.4-1.4 2.2 2.2 4.8-4.8 1.4 1.4-6.2 6.2Z",
  warning: "M10 1.5 .8 17.5h18.4L10 1.5Zm-1 5.5h2v5.2H9V7Zm0 6.6h2v2H9v-2Z",
  no_prediction: "M10 1.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17ZM5.5 9h9v2h-9V9Z",
  idle: "M10 1.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17Zm0 3a5.5 5.5 0 1 1 0 11 5.5 5.5 0 0 1 0-11Z",
};

function setStatus(status, messages) {
  const line = $("status-line");
  line.dataset.status = status;
  const icon = line.querySelector("svg");
  icon.replaceChildren();
  el("path", { d: STATUS_ICON[status] || STATUS_ICON.idle }, icon);
  $("status-text").textContent = STATUS_TEXT[status] || status;
  const ul = $("messages");
  ul.replaceChildren();
  for (const m of messages || []) { const li = document.createElement("li"); li.textContent = sentence(m); ul.appendChild(li); }
}
const sentence = (m) => (m ? m.charAt(0).toUpperCase() + m.slice(1) + (/[.!?]$/.test(m) ? "" : ".") : "");

function setReadout(p) {
  if (!p) {
    $("pred-value").textContent = "–";
    $("pred-when").textContent = "Run a replay or upload files to see a prediction.";
    drawGauge(null); setStatus("idle", []); return;
  }
  $("pred-value").textContent = p.v === null ? "–" : fmtNum(p.v);
  $("pred-when").textContent = `For the cast at ${fmtTime(p.t)}`;
  drawGauge(p);
  setStatus(p.status, p.messages);
}

function drawGauge(p) {
  const svg = $("gauge");
  svg.replaceChildren();
  const cap = $("gauge-caption");
  const W = svg.clientWidth || 320, H = 76, padL = 8, padR = 8, y0 = 32, h = 12;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!p || p.v === null) { cap.textContent = ""; return; }
  const hiVal = Math.max(1.0, (p.hi ?? p.v) + 0.1, (p.actual ?? 0) + 0.1);
  const max = Math.ceil(hiVal * 5) / 5;
  const x = (v) => padL + (Math.max(0, Math.min(v, max)) / max) * (W - padL - padR);
  el("rect", { x: padL, y: y0, width: W - padL - padR, height: h, fill: "var(--gauge-track)", rx: h / 2 }, svg);
  if (state.typical) el("rect", { x: x(state.typical[0]), y: y0, width: x(state.typical[1]) - x(state.typical[0]), height: h, fill: "var(--typical)", stroke: "var(--typical-edge)", rx: 2 }, svg);
  if (p.lo !== null && p.hi !== null) el("rect", { x: x(p.lo), y: y0 + 2, width: Math.max(2, x(p.hi) - x(p.lo)), height: h - 4, rx: 3, fill: "var(--band)", stroke: "var(--band-edge)" }, svg);
  el("line", { x1: x(p.v), x2: x(p.v), y1: y0 - 7, y2: y0 + h + 7, stroke: "var(--predicted)", "stroke-width": 2.5, "stroke-linecap": "round" }, svg);
  if (p.actual !== undefined && p.actual !== null) {
    const ax = x(p.actual);
    el("path", { d: `M${ax - 5},${y0 - 14} L${ax + 5},${y0 - 14} L${ax},${y0 - 6} Z`, fill: "var(--measured)", "stroke-linejoin": "round" }, svg);
  }
  const step = max > 1.6 ? 0.5 : 0.2;
  for (let v = 0; v <= max + 1e-9; v += step) {
    el("line", { x1: x(v), x2: x(v), y1: y0 + h + 3, y2: y0 + h + 7, stroke: "var(--gauge-tick)" }, svg);
    svgText(svg, x(v), y0 + h + 20, fmtNum(v, 1), "axis-text", v === 0 ? "start" : (v + step > max + 1e-9 ? "end" : "middle")).setAttribute("style", "fill:var(--ink-3);font:450 11.5px var(--font);font-variant-numeric:tabular-nums");
  }
  let txt = `${Math.round(state.level * 100)} % interval ${fmtNum(p.lo)} to ${fmtNum(p.hi)} wt%.`;
  if (state.typical) txt += ` Typical range ${fmtNum(state.typical[0])} to ${fmtNum(state.typical[1])} (shaded).`;
  if (p.actual !== undefined && p.actual !== null) txt += ` Measured ${fmtNum(p.actual)} (blue marker).`;
  cap.textContent = txt;
}

/* ---------------------------------------------------------------- chart */
function niceTimeStep(hours, width = 800) {
  const maxLabels = Math.max(3, Math.min(9, Math.floor(width / 75)));     // room for "25 Jan" at each label
  const steps = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1440];
  for (const s of steps) if (hours / s <= maxLabels) return s;
  return 2160;
}
function segments(points, maxGapH = 6) {
  const out = []; let cur = [];
  for (const p of points) {
    if (p.v === null || p.v === undefined) continue;
    if (cur.length && (p.t - cur[cur.length - 1].t) / 3.6e6 > maxGapH) { out.push(cur); cur = []; }
    cur.push(p);
  }
  if (cur.length) out.push(cur);
  return out;
}

function drawChart() {
  const svg = $("chart");
  svg.replaceChildren();
  const W = svg.clientWidth || 800, H = svg.clientHeight || 360;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const all = [...state.measured, ...state.predicted];
  $("chart-empty").hidden = all.length > 0;
  if (!all.length) return;
  const m = { l: 44, r: W < 560 ? 12 : 84, t: 26, b: 30 };
  let [t0, t1] = state.xDomain || [Math.min(...all.map((p) => p.t)), Math.max(...all.map((p) => p.t))];
  if (t1 - t0 < 3 * 3.6e6) { t0 -= 1.5 * 3.6e6; t1 += 1.5 * 3.6e6; }
  if (!state.xDomain) t1 += (t1 - t0) * 0.04;          // breathing room after a single predicted point
  const vals = all.flatMap((p) => [p.v, p.hi]).filter((v) => v !== null && v !== undefined);
  const yMax = state.yMax || Math.max(1.0, Math.ceil((Math.max(...vals) + 0.05) * 5) / 5);
  const X = (t) => m.l + ((t - t0) / (t1 - t0)) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - Math.min(v, yMax) / yMax) * (H - m.t - m.b);

  // chart-paper grid
  const g = el("g", {}, svg);
  const [major, minor] = yMax > 1.6 ? [0.5, 0.1] : [0.2, 0.05];
  for (let v = 0; v <= yMax + 1e-9; v += minor) {
    const isMajor = Math.abs(v / major - Math.round(v / major)) < 1e-6;
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: isMajor ? "grid-major" : "grid-minor" }, g);
    if (isMajor) svgText(g, m.l - 6, Y(v) + 4, fmtNum(v, 1), "axis-text", "end");
  }
  const hours = (t1 - t0) / 3.6e6, stepH = niceTimeStep(hours, W - m.l - m.r), minorH = stepH >= 24 ? stepH / 4 : Math.max(stepH / 3, 1);
  const start = new Date(t0); start.setMinutes(0, 0, 0);
  if (stepH >= 24) start.setHours(0);
  for (let t = start.getTime(); t <= t1; t += minorH * 3.6e6) {
    if (t < t0) continue;
    const d = new Date(t);
    const isMajor = stepH >= 24 ? d.getHours() === 0 && Math.round((t - start.getTime()) / 3.6e6) % stepH === 0 : d.getHours() % stepH === 0 && d.getMinutes() === 0;
    el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, class: isMajor ? "grid-major" : "grid-minor" }, g);
    if (isMajor) svgText(g, X(t), H - m.b + 18, stepH >= 24 || d.getHours() === 0 ? fmtDay(d) : fmtHour(d), "axis-text", "middle");
  }
  svgText(g, m.l - 6, m.t - 10, "wt% Si", "axis-text", "end");

  if (state.typical) {
    el("rect", { x: m.l, y: Y(state.typical[1]), width: W - m.l - m.r, height: Y(state.typical[0]) - Y(state.typical[1]), class: "typical", opacity: 0.55 }, svg);
  }

  // interval band and traces
  const pred = state.predicted.filter((p) => p.v !== null);
  for (const seg of segments(pred)) {
    if (seg.length === 1) {
      const p = seg[0];
      el("line", { x1: X(p.t), x2: X(p.t), y1: Y(p.hi), y2: Y(p.lo), class: "whisker" }, svg);
      continue;
    }
    const up = seg.map((p) => `${X(p.t)},${Y(p.hi)}`), dn = seg.slice().reverse().map((p) => `${X(p.t)},${Y(p.lo)}`);
    el("polygon", { points: [...up, ...dn].join(" "), class: "band" }, svg);
  }
  for (const seg of segments(state.measured)) {
    el("path", { d: seg.map((p, i) => `${i ? "L" : "M"}${X(p.t)},${Y(p.v)}`).join(""), class: "trace-measured" }, svg);
    if (seg.length <= 60) for (const p of seg) el("circle", { cx: X(p.t), cy: Y(p.v), r: 3, class: "dot-measured" }, svg);
  }
  for (const seg of segments(pred)) {
    if (seg.length === 1) { el("circle", { cx: X(seg[0].t), cy: Y(seg[0].v), r: 5, class: "dot-predicted" }, svg); continue; }
    el("path", { d: seg.map((p, i) => `${i ? "L" : "M"}${X(p.t)},${Y(p.v)}`).join(""), class: "trace-predicted" }, svg);
  }
  if (state.playing && pred.length) { const p = pred[pred.length - 1]; el("circle", { cx: X(p.t), cy: Y(p.v), r: 4, class: "pen-tip" }, svg); }

  // direct labels at the end of each trace
  if (m.r > 40) {
    const lastM = state.measured.filter((p) => p.v !== null).slice(-1)[0], lastP = pred.slice(-1)[0];
    let yM = lastM ? Y(lastM.v) + 4 : null, yP = lastP ? Y(lastP.v) + 4 : null;
    if (yM !== null && yP !== null && Math.abs(yM - yP) < 14) { const mid = (yM + yP) / 2; if (yM <= yP) { yM = mid - 7; yP = mid + 7; } else { yM = mid + 7; yP = mid - 7; } }
    if (lastM) svgText(svg, Math.min(X(lastM.t), W - m.r) + 8, yM, "Measured", "label-measured");
    if (lastP) svgText(svg, Math.min(X(lastP.t), W - m.r) + 8, yP, "Predicted", "label-predicted");
  }

  // hover / keyboard layer
  const times = [...new Set(all.map((p) => p.t.getTime()))].sort((a, b) => a - b);
  state.chartTimes = times;
  const cross = el("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  const show = (idx, px) => {
    if (idx === null) { cross.setAttribute("visibility", "hidden"); $("tooltip").hidden = true; return; }
    const t = times[idx];
    cross.setAttribute("x1", X(t)); cross.setAttribute("x2", X(t)); cross.setAttribute("visibility", "visible");
    showTooltip(t, px ?? X(t), W);
  };
  const nearest = (clientX) => {
    const r = svg.getBoundingClientRect(), px = ((clientX - r.left) / r.width) * W;
    const t = t0 + ((px - m.l) / (W - m.l - m.r)) * (t1 - t0);
    let best = 0;
    for (let i = 1; i < times.length; i++) if (Math.abs(times[i] - t) < Math.abs(times[best] - t)) best = i;
    return best;
  };
  hit.addEventListener("pointermove", (e) => { state.focusIndex = nearest(e.clientX); show(state.focusIndex); });
  hit.addEventListener("pointerleave", () => show(null));
  svg.onkeydown = (e) => {
    if (!times.length) return;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const i = state.focusIndex ?? times.length - 1;
      state.focusIndex = Math.max(0, Math.min(times.length - 1, i + (e.key === "ArrowRight" ? 1 : -1)));
      show(state.focusIndex);
    } else if (e.key === "Escape") show(null);
  };
  svg.onblur = () => show(null);
}

function showTooltip(t, px, W) {
  const tip = $("tooltip");
  tip.replaceChildren();
  const meas = state.measured.find((p) => p.t.getTime() === t);
  const pred = state.predicted.find((p) => p.t.getTime() === t);
  const head = document.createElement("div"); head.className = "tt-time"; head.textContent = fmtTime(new Date(t)); tip.appendChild(head);
  const row = (color, value, label) => {
    const r = document.createElement("div"); r.className = "tt-row";
    const k = document.createElement("span"); k.className = "tt-key"; k.style.background = color;
    const b = document.createElement("b"); b.textContent = value;
    const s = document.createElement("span"); s.textContent = label;
    r.append(k, b, s); tip.appendChild(r);
  };
  if (meas) row("var(--measured)", fmtNum(meas.v), "measured");
  if (pred && pred.v !== null) {
    row("var(--predicted)", fmtNum(pred.v), "predicted");
    row("var(--band-edge)", `${fmtNum(pred.lo)} to ${fmtNum(pred.hi)}`, `${Math.round(state.level * 100)} % interval`);
  }
  if (pred && pred.status !== "ok") { const s = document.createElement("div"); s.textContent = STATUS_TEXT[pred.status]; tip.appendChild(s); }
  tip.hidden = false;
  const paper = tip.parentElement.getBoundingClientRect(), svgW = $("chart").getBoundingClientRect().width;
  const left = (px / W) * svgW;
  tip.style.top = "12px";
  tip.style.left = `${left + 14 + tip.offsetWidth > paper.width ? left - tip.offsetWidth - 14 : left + 14}px`;
}

/* ---------------------------------------------------------------- table */
function drawTable() {
  const body = $("table-body");
  body.replaceChildren();
  const byT = new Map();
  for (const p of state.measured) byT.set(p.t.getTime(), { t: p.t, meas: p.v });
  for (const p of state.predicted) byT.set(p.t.getTime(), { ...(byT.get(p.t.getTime()) || { t: p.t }), pred: p });
  for (const row of [...byT.values()].sort((a, b) => a.t - b.t)) {
    const tr = document.createElement("tr");
    const p = row.pred;
    for (const v of [fmtTime(row.t), fmtNum(row.meas), p ? fmtNum(p.v) : "", p && p.v !== null ? `${fmtNum(p.lo)} to ${fmtNum(p.hi)}` : "", p ? STATUS_TEXT[p.status] : "lab result"]) {
      const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

function render() { drawChart(); drawTable(); }

/* ---------------------------------------------------------------- replay */
function toPred(r) {
  return { t: parseT(r.time), v: r.prediction, lo: r.lower, hi: r.upper, status: r.status, messages: r.messages, actual: r.actual };
}

function runningLine(k, rows, summary) {
  const done = rows.slice(0, k).filter((r) => r.prediction !== null);
  if (!done.length) return "";
  const mae = done.reduce((s, r) => s + Math.abs(r.actual - r.prediction), 0) / done.length;
  const pmae = done.reduce((s, r) => s + Math.abs(r.actual - r.persistence), 0) / done.length;
  const held = done.filter((r) => r.lower <= r.actual && r.actual <= r.upper).length;
  let txt = `Cast ${k} of ${rows.length}. Average error ${fmtNum(mae, 3)} wt%, against ${fmtNum(pmae, 3)} for simply repeating the last lab result. ` +
    `The interval held the measured value in ${held} of ${done.length} casts.`;
  if (k === rows.length && summary.upsets) txt += ` ${summary.upsets} of these casts were above 1.2 wt%, where the model is least reliable.`;
  const faulty = new Map();
  for (const r of rows.slice(0, k)) for (const msg of r.messages) {
    const hit = /^faulty or missing readings were replaced: (.*)$/.exec(msg);
    if (hit) for (const s of hit[1].replace(" ...", "").split(", ")) faulty.set(s, (faulty.get(s) || 0) + 1);
  }
  const worst = [...faulty.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3);
  if (worst.length) txt += ` Faulty sensor readings were replaced: ${worst.map(([s, n]) => `${s} in ${n} ${n === 1 ? "cast" : "casts"}`).join(", ")}.`;
  return txt;
}

function stopPlayback() {
  if (state.playing) { clearInterval(state.playing); state.playing = null; }
  $("replay-skip").disabled = true;
}

function revealTo(k) {
  const { rows, lead, summary } = state.replay;
  state.measured = [...lead, ...rows.slice(0, k).map((r) => ({ t: parseT(r.time), v: r.actual }))];
  state.predicted = rows.slice(0, k).map(toPred);
  setReadout(state.predicted[state.predicted.length - 1]);
  $("running").textContent = runningLine(k, rows, summary);
  render();
}

async function startReplay(ev) {
  ev.preventDefault();
  stopPlayback(); showError("");
  const start = $("replay-start").value, casts = Number($("replay-casts").value);
  const go = $("replay-go");
  go.disabled = true; go.textContent = "Preparing replay…";
  try {
    const resp = await fetch("/api/demo/replay", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ start, casts }) });
    const body = await readJson(resp);
    const rows = body.results, lead = body.lead_in.map((p) => ({ t: parseT(p.time), v: p.si }));
    applyModelMeta(body.model);
    state.replay = { rows, lead, summary: body.summary };
    const times = [...lead, ...rows.map((r) => ({ t: parseT(r.time) }))].map((p) => p.t.getTime());
    state.xDomain = [Math.min(...times), Math.max(...times)];
    const vals = rows.flatMap((r) => [r.actual, r.upper]).concat(lead.map((p) => p.v)).filter((v) => v !== null);
    state.yMax = Math.max(1.0, Math.ceil((Math.max(...vals) + 0.05) * 5) / 5);
    if (reduceMotion) { revealTo(rows.length); return; }
    let k = 0;
    const interval = Math.max(30, Math.min(160, 7000 / rows.length));
    $("replay-skip").disabled = false;
    state.playing = setInterval(() => {
      k += 1;
      if (k >= rows.length) { stopPlayback(); revealTo(rows.length); return; }
      revealTo(k);
    }, interval);
  } catch (e) {
    showError(e.message);
  } finally {
    go.disabled = false; go.textContent = "Play replay";
  }
}

/* ---------------------------------------------------------------- files */
async function predictFromFiles(ev) {
  ev.preventDefault();
  stopPlayback(); showError("");
  const fd = new FormData();
  fd.append("sensor_file", $("sensor-file").files[0]);
  fd.append("si_file", $("si-file").files[0]);
  fd.append("query_time", $("files-time").value);
  const fid = $("files-model").value;
  const url = fid ? `/api/furnaces/${encodeURIComponent(fid)}/predict/upload` : "/api/predict/upload";
  const btn = $("files-go");
  btn.disabled = true; btn.textContent = "Reading files…";
  try {
    const body = await readJson(await fetch(url, { method: "POST", body: fd }));
    applyModelMeta(body.model);
    const r = body.results[0];
    state.measured = body.context.si_history.map((p) => ({ t: parseT(p.time), v: p.si }));
    state.predicted = [toPred({ ...r, actual: undefined })];
    state.xDomain = null; state.yMax = null; state.replay = null;
    setReadout(state.predicted[0]);
    $("running").textContent = `Used ${body.context.sensor_rows_used} sensor rows from ${fmtTime(parseT(body.context.sensor_from))} to ${fmtTime(parseT(body.context.sensor_to))}, and the ${state.measured.length} lab results shown before the cast.`;
    render();
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Predict next cast";
  }
}

/* ---------------------------------------------------------------- another furnace */
const train = { upload: null, polling: null, furnaces: [], current: null };
const fmtPct = (v, d = 0) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}\u00a0%`);   // no line break before %
const fmtDate = (s) => (s ? new Date(s).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : "–");
function option(sel, value, text, selected = false) {
  const o = document.createElement("option"); o.value = value; o.textContent = text; o.selected = selected; sel.appendChild(o); return o;
}
function cell(tr, text) { const td = document.createElement("td"); td.textContent = text; tr.appendChild(td); return td; }

function sheetOf(role) {
  const sheets = role === "sensor" ? train.upload.sensor_sheets : train.upload.si_sheets;
  return sheets[$(role === "sensor" ? "map-sensor-sheet" : "map-si-sheet").value];
}
function timeColumns(sheet) { return sheet.columns.filter((c) => c.kind === "time").map((c) => c.name); }
function numericColumns(sheet) { return sheet.columns.filter((c) => c.kind === "numeric"); }

function fillSelect(sel, values, chosen, emptyLabel = null) {
  sel.replaceChildren();
  if (emptyLabel !== null) option(sel, "", emptyLabel, !chosen);
  for (const v of values) option(sel, v, v, v === chosen);
  if (!values.length && emptyLabel === null) option(sel, "", "No suitable column", true);
}

function renderSensorSide(useSuggestion) {
  const sh = sheetOf("sensor"), sug = train.upload.suggested.sensor;
  const times = timeColumns(sh);
  fillSelect($("map-sensor-time"), times, useSuggestion && times.includes(sug.time_col) ? sug.time_col : times[0]);
  const tcol = $("map-sensor-time").value;
  const meta = sh.columns.find((c) => c.name === tcol) || {};
  $("map-sensor-dayfirst").checked = useSuggestion ? sug.dayfirst : !!meta.dayfirst_suggested;
  $("map-sensor-info").textContent = `${sh.rows.toLocaleString("en-GB")} rows.` +
    (meta.dayfirst_ambiguous ? " The dates could be day-first or month-first; check the box if days come first." : "");
  const box = $("map-sensors"); box.replaceChildren();
  const chosen = new Set(useSuggestion ? sug.columns : []);
  for (const c of numericColumns(sh)) {
    if (c.name === tcol) continue;
    const lab = document.createElement("label"); lab.dataset.name = c.name;
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.value = c.name;
    cb.checked = useSuggestion ? chosen.has(c.name) : c.missing_share < 0.5;
    cb.addEventListener("change", updateSensorCount);
    const name = document.createElement("span"); name.textContent = c.name;
    const ex = document.createElement("span"); ex.className = "ex"; ex.textContent = c.example ?? "";
    lab.append(cb, name, ex);
    if (c.missing_share >= 0.5) { const f = document.createElement("span"); f.className = "flag"; f.textContent = `${fmtPct(c.missing_share)} empty`; lab.appendChild(f); }
    box.appendChild(lab);
  }
  updateSensorCount(useSuggestion ? sug.blast_off_col : null);
}

function selectedSensors() { return [...$("map-sensors").querySelectorAll("input:checked")].map((cb) => cb.value); }
function updateSensorCount(prefBlast) {
  const sel = selectedSensors();
  $("map-sensor-count").textContent = `Sensors: ${sel.length} selected`;
  const blast = $("map-blast"), keep = typeof prefBlast === "string" ? prefBlast : blast.value;
  fillSelect(blast, sel, sel.includes(keep) ? keep : "", "None");
}

function renderSiSide(useSuggestion) {
  const sh = sheetOf("si"), sug = train.upload.suggested.si;
  const times = timeColumns(sh), nums = numericColumns(sh).map((c) => c.name);
  fillSelect($("map-si-time"), times, useSuggestion && times.includes(sug.time_col) ? sug.time_col : times[0]);
  fillSelect($("map-si-value"), nums, useSuggestion && nums.includes(sug.value_col) ? sug.value_col : nums[0]);
  const meta = sh.columns.find((c) => c.name === $("map-si-time").value) || {};
  $("map-si-dayfirst").checked = useSuggestion ? sug.dayfirst : !!meta.dayfirst_suggested;
  $("map-si-info").textContent = `${sh.rows.toLocaleString("en-GB")} rows.` +
    (meta.dayfirst_ambiguous ? " The dates could be day-first or month-first; check the box if days come first." : "");
}

function renderMapping() {
  const up = train.upload;
  $("map-sensor-file").textContent = `(${up.files.sensor})`;
  $("map-si-file").textContent = `(${up.files.si})`;
  for (const [role, sheets, sug] of [["sensor", up.sensor_sheets, up.suggested.sensor], ["si", up.si_sheets, up.suggested.si]]) {
    const sel = $(`map-${role}-sheet`), names = Object.keys(sheets);
    fillSelect(sel, names, sug.sheet);
    $(`map-${role}-sheet-wrap`).hidden = names.length < 2;
  }
  renderSensorSide(true); renderSiSide(true);
  if (!$("map-name").value) $("map-name").value = up.files.sensor.replace(/\.[^.]+$/, "").replace(/[_-]?sensors?$/i, "").replace(/[_-]+/g, " ").trim().slice(0, 60);
  $("train-map").hidden = false;
}

async function readTrainingFiles(ev) {
  ev.preventDefault(); showError("");
  const fd = new FormData();
  fd.append("sensor_file", $("train-sensor").files[0]);
  fd.append("si_file", $("train-si").files[0]);
  const btn = $("train-read");
  btn.disabled = true; btn.textContent = "Reading files…";
  try {
    train.upload = await readJson(await fetch("/api/furnaces/inspect", { method: "POST", body: fd }));
    $("train-report").hidden = true;
    renderMapping();
    $("train-map").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  } catch (e) { showError(e.message); } finally { btn.disabled = false; btn.textContent = "Read files"; }
}

function collectMapping() {
  return {
    sensor: { sheet: $("map-sensor-sheet").value, time_col: $("map-sensor-time").value, columns: selectedSensors(),
      blast_off_col: $("map-blast").value || null, dayfirst: $("map-sensor-dayfirst").checked },
    si: { sheet: $("map-si-sheet").value, time_col: $("map-si-time").value, value_col: $("map-si-value").value,
      dayfirst: $("map-si-dayfirst").checked },
  };
}

async function startTraining(ev) {
  ev.preventDefault(); showError("");
  for (const l of $("map-sensors").querySelectorAll("label.problem")) l.classList.remove("problem");
  const btn = $("train-go");
  btn.disabled = true;
  try {
    const body = await readJson(await fetch("/api/furnaces/train", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: train.upload.upload_id, name: $("map-name").value, mapping: collectMapping() }) }));
    $("train-progress").hidden = false; $("train-report").hidden = true;
    $("train-bar").value = 0; $("train-step").textContent = "Waiting to start";
    pollJob(body.furnace_id);
  } catch (e) { showError(e.message); btn.disabled = false; }
}

function pollJob(fid) {
  clearInterval(train.polling);
  const tick = async () => {
    let job;
    try { job = await readJson(await fetch(`/api/furnaces/${encodeURIComponent(fid)}/job`)); } catch (e) { return; }
    $("train-bar").value = job.progress || 0;
    $("train-step").textContent = job.status === "running" ? `${job.step}…` : (job.step || "");
    if (job.status === "done" || job.status === "failed") {
      clearInterval(train.polling); train.polling = null;
      $("train-go").disabled = false; $("train-progress").hidden = true;
      if (job.status === "failed") {
        const err = job.error || {};
        let msg = err.message || "Training failed.";
        const bad = err.details && err.details.sensors;
        if (bad) {
          msg += ` ${bad.join(", ")}`;
          for (const l of $("map-sensors").querySelectorAll("label")) if (bad.includes(l.dataset.name)) l.classList.add("problem");
        }
        if (err.details && err.details.missing) msg += `: ${err.details.missing.join(", ")}`;
        showError(msg);
      } else {
        await refreshFurnaces();
        await showReport(fid);
      }
    }
  };
  train.polling = setInterval(tick, 1500); tick();
}

async function showReport(fid) {
  showError("");
  let card;
  try { card = await readJson(await fetch(`/api/furnaces/${encodeURIComponent(fid)}`)); } catch (e) { showError(e.message); return; }
  train.current = card;
  const ev = card.evaluation, t = ev.test;
  $("report-title").textContent = `Result for ${card.name}`;
  const v = $("report-verdict"); v.dataset.level = ev.verdict.level; v.textContent = ev.verdict.text;
  const cvMean = (key) => ev.folds.reduce((s, f) => s + f[key].mae, 0) / ev.folds.length;
  const body = $("report-body"); body.replaceChildren();
  for (const [label, key] of [["This model", "model"], ["Repeat the last lab result", "persistence"], ["Training average", "mean"]]) {
    const tr = document.createElement("tr");
    cell(tr, label); cell(tr, fmtNum(t[key].mae, 3)); cell(tr, fmtNum(t[key].r2, 2)); cell(tr, fmtPct(t[key].within_0_1)); cell(tr, fmtNum(cvMean(key), 3));
    body.appendChild(tr);
  }
  const facts = $("report-facts"); facts.replaceChildren();
  const add = (text, cls) => { const li = document.createElement("li"); li.textContent = text; if (cls) li.className = cls; facts.appendChild(li); };
  const boot = ev.test_vs_persistence;
  add(`Held-out period ${fmtDate(t.from)} to ${fmtDate(t.to)}: ${t.n.toLocaleString("en-GB")} casts the model never saw. ` +
      `Difference in MAE against repeating the last result: ${fmtNum(boot.diff, 4)} wt% (95 % interval ${fmtNum(boot.ci[0], 4)} to ${fmtNum(boot.ci[1], 4)}).`);
  const cvCov = ev.folds.reduce((s, f) => s + f.coverage, 0) / ev.folds.length;
  add(`90 % intervals held the measured value in ${fmtPct(t.coverage, 1)} of held-out casts (${fmtPct(cvCov, 1)} in cross-validation); ` +
      `they are ±${fmtNum(card.interval_half_width, 2)} wt% wide.`);
  add(`Trained on ${card.training_samples.toLocaleString("en-GB")} Si results from ${fmtDate(card.training_range[0])} to ${fmtDate(card.training_range[1])}, ` +
      `with ${card.mapping.sensor.columns.length} sensors (${card.n_features} inputs), in ${card.training_seconds} s.`);
  if (card.sensor_faults.length) add(`Readings replaced most often: ${card.sensor_faults.slice(0, 4).map((f) => `${f.sensor} (${fmtPct(f.share, 1)})`).join(", ")}.`);
  add(`Warnings are raised when the previous Si result is above ${fmtNum(card.upset_level, 2)} wt% (top 0.5 % of this furnace’s history) or older than ${card.stale_hours} h.`);
  for (const w of card.warnings) add(sentence(w), "warn");
  $("train-report").hidden = false;

  // the held-out period on the recorder chart
  const s = card.test_series;
  stopPlayback();
  state.typical = card.typical_si_range; state.level = card.interval_level;
  state.measured = s.time.map((x, i) => ({ t: parseT(x), v: s.actual[i] }));
  state.predicted = s.time.map((x, i) => ({ t: parseT(x), v: s.prediction[i], lo: s.lower[i], hi: s.upper[i], status: "ok", messages: [], actual: s.actual[i] }));
  state.xDomain = null; state.yMax = null; state.replay = null;
  setReadout(state.predicted[state.predicted.length - 1]);
  $("pred-when").textContent = `Last held-out cast, ${fmtTime(state.predicted[state.predicted.length - 1].t)}`;
  $("running").textContent = `${card.name}, held-out period: average error ${fmtNum(t.model.mae, 3)} wt%, against ${fmtNum(t.persistence.mae, 3)} for repeating the last lab result.`;
  render();
}

async function refreshFurnaces() {
  try { train.furnaces = (await readJson(await fetch("/api/furnaces"))).furnaces; } catch (_) { return; }
  const ready = train.furnaces.filter((f) => f.verdict);
  const hs = $("health-source"), keepH = hs.value;
  hs.replaceChildren(); option(hs, "demo", "Built-in model, held-out period");
  for (const f of ready) option(hs, f.furnace_id, `${f.name}, held-out period`, f.furnace_id === keepH);
  const sel = $("files-model"), keep = sel.value;
  sel.replaceChildren(); option(sel, "", "Built-in model (this furnace)");
  for (const f of ready) option(sel, f.furnace_id, f.name, f.furnace_id === keep);
  updateFilesLede();
  const body = $("furnace-body"); body.replaceChildren();
  $("furnace-list-wrap").hidden = train.furnaces.length === 0;
  for (const f of train.furnaces) {
    const tr = document.createElement("tr");
    cell(tr, f.name);
    if (f.verdict) {
      cell(tr, fmtDate(f.training_range[1]));
      cell(tr, `${fmtNum(f.test_mae, 3)} vs ${fmtNum(f.persistence_mae, 3)} wt%`);
      const tag = document.createElement("span"); tag.className = "tag"; tag.dataset.level = f.verdict;
      tag.textContent = { good: "Better than repeating", weak: "Not clearly better", poor: "Not better" }[f.verdict];
      cell(tr, "").appendChild(tag);
    } else { cell(tr, "–"); cell(tr, f.status === "failed" ? "Training failed" : "Training…"); cell(tr, ""); }
    const actions = cell(tr, "");
    if (f.verdict) {
      const view = document.createElement("button"); view.type = "button"; view.className = "small"; view.textContent = "Report";
      view.setAttribute("aria-label", `Report for ${f.name}`);
      view.addEventListener("click", () => showReport(f.furnace_id)); actions.appendChild(view);
    }
    const del = document.createElement("button"); del.type = "button"; del.className = "small"; del.textContent = "Delete";
    del.setAttribute("aria-label", `Delete ${f.name}`);
    del.addEventListener("click", async () => {
      if (!window.confirm(`Delete the model for ${f.name}? This cannot be undone.`)) return;
      try { await readJson(await fetch(`/api/furnaces/${encodeURIComponent(f.furnace_id)}`, { method: "DELETE" })); } catch (e) { showError(e.message); }
      if (train.current && train.current.furnace_id === f.furnace_id) { $("train-report").hidden = true; train.current = null; }
      refreshFurnaces();
    });
    actions.appendChild(del);
    body.appendChild(tr);
  }
}

function updateFilesLede() {
  const fid = $("files-model").value, f = train.furnaces.find((x) => x.furnace_id === fid);
  $("files-lede-builtin").hidden = !!f; $("files-lede-furnace").hidden = !f;
  if (!f) return;
  const list = (cols) => cols.join(", ");
  $("files-furnace-name").textContent = f.name;
  $("files-furnace-sensor").textContent = list(f.required_columns.sensor);
  $("files-furnace-si").textContent = list(f.required_columns.si);
}

function initTraining() {
  $("train-upload").addEventListener("submit", readTrainingFiles);
  $("train-map").addEventListener("submit", startTraining);
  $("train-restart").addEventListener("click", () => { $("train-map").hidden = true; train.upload = null; $("train-upload").reset(); $("train-sensor").focus(); });
  $("map-sensor-sheet").addEventListener("change", () => renderSensorSide(false));
  $("map-si-sheet").addEventListener("change", () => renderSiSide(false));
  $("map-sensor-time").addEventListener("change", () => {
    const keep = new Set(selectedSensors()); renderSensorSide(false);
    for (const cb of $("map-sensors").querySelectorAll("input")) cb.checked = keep.has(cb.value) || (!keep.size && cb.checked);
    updateSensorCount();
  });
  $("map-all").addEventListener("click", () => { for (const cb of $("map-sensors").querySelectorAll("input")) cb.checked = true; updateSensorCount(); });
  $("map-none").addEventListener("click", () => { for (const cb of $("map-sensors").querySelectorAll("input")) cb.checked = false; updateSensorCount(); });
  $("report-use").addEventListener("click", () => {
    if (!train.current) return;
    selectTab("tab-files"); $("files-model").value = train.current.furnace_id; updateFilesLede(); $("tab-files").focus();
  });
  $("files-model").addEventListener("change", updateFilesLede);
  $("health-source").addEventListener("change", loadHealth);
  new ResizeObserver(() => { if (health.report && !$("panel-health").hidden) drawHealth(health.report); }).observe($("health-chart"));
  refreshFurnaces();
}

/* ---------------------------------------------------------------- model health */
const health = { report: null, seq: 0 };
const KIND = { coverage: "Intervals too narrow", error: "Not better than repeating", inputs: "Sensor faults" };

async function loadHealth() {
  const src = $("health-source").value, seq = ++health.seq;
  const url = src === "demo" ? "/api/health/demo" : `/api/furnaces/${encodeURIComponent(src)}/health`;
  $("health-status").dataset.level = "none"; $("health-status").textContent = "Checking the predictions against the lab results…";
  try {
    const rep = await readJson(await fetch(url));
    if (seq !== health.seq) return;               // a newer selection is loading
    health.report = rep; showError(""); renderHealth(rep);
  } catch (e) { if (seq === health.seq) { $("health-status").textContent = ""; showError(e.message); } }
}

function renderHealth(rep) {
  const st = $("health-status"); st.dataset.level = rep.status.level;
  st.textContent = (rep.status.level === "ok" ? "Healthy now. " : "Needs attention now. ") + rep.status.text;
  const o = rep.overall;
  $("health-summary").textContent = `${rep.source.name}: ${rep.casts.toLocaleString("en-GB")} casts from ${fmtDate(rep.from)} to ${fmtDate(rep.to)}. ` +
    `Overall MAE ${fmtNum(o.mae, 3)} wt% against ${fmtNum(o.persistence_mae, 3)} for repeating the last result; intervals held ${fmtPct(o.coverage, 1)}. ` +
    `Each curve point averages the ${rep.window} casts around it; “now” uses the last ${rep.window}. ${rep.source.note}`;
  const ul = $("health-alerts"); ul.replaceChildren();
  if (!rep.alerts.length) { const li = document.createElement("li"); li.textContent = "No alerts in this period."; ul.appendChild(li); }
  for (const a of rep.alerts) {
    const li = document.createElement("li");
    const when = document.createElement("span"); when.className = "when";
    when.textContent = `${fmtDate(a.from)} to ${fmtDate(a.to)} · `;
    const kind = document.createElement("span"); kind.className = "kind"; kind.textContent = KIND[a.kind] || a.kind;
    when.appendChild(kind);
    li.append(when, document.createTextNode(a.text));
    ul.appendChild(li);
  }
  const hasFault = rep.rolling.fault_share !== undefined;
  $("health-fault-col").hidden = !hasFault; $("health-fault-legend").hidden = !hasFault;
  const body = $("health-months"); body.replaceChildren();
  for (const m of rep.monthly) {
    const tr = document.createElement("tr");
    const label = new Date(`${m.month}-01T00:00:00`).toLocaleDateString("en-GB", { month: "short", year: "numeric" });
    cell(tr, label); cell(tr, m.casts); cell(tr, fmtNum(m.mae, 3)); cell(tr, fmtNum(m.persistence_mae, 3)); cell(tr, fmtPct(m.coverage));
    if (hasFault) cell(tr, fmtPct(m.fault_share)); cell(tr, fmtNum(m.max_si, 2));
    body.appendChild(tr);
  }
  $("health-chart-desc").textContent = `Rolling error of the model and of repeating the last result, rolling interval coverage` +
    (hasFault ? " and the share of casts with replaced sensor readings" : "") + `, with ${rep.alerts.length} alert periods shaded.`;
  drawHealth(rep);
}

function drawHealth(rep) {
  const svg = $("health-chart"); svg.replaceChildren();
  const W = svg.clientWidth || 900, H = svg.clientHeight || 420;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const r = rep.rolling, times = r.time.map((x) => parseT(x).getTime());
  if (!times.length) return;
  const hasFault = r.fault_share !== undefined;
  const m = { l: 48, r: 14, t: 8, b: 26 }, gap = 26;
  const panels = hasFault ? [0.42, 0.33, 0.25] : [0.55, 0.45];
  const inner = H - m.t - m.b - gap * (panels.length - 1);
  let y0 = m.t;
  const boxes = panels.map((f) => { const b = { top: y0 + 16, h: f * inner - 16 }; y0 += f * inner + gap; return b; });
  const t0 = Math.min(times[0], parseT(rep.from).getTime()), t1 = Math.max(times[times.length - 1], parseT(rep.to).getTime());
  const X = (t) => m.l + ((t - t0) / Math.max(1, t1 - t0)) * (W - m.l - m.r);
  const line = (vals, Y, cls, g) => {
    let d = "", pen = false;
    vals.forEach((v, i) => { if (v === null) { pen = false; return; } d += `${pen ? "L" : "M"}${X(times[i])},${Y(v)}`; pen = true; });
    el("path", { d, class: cls }, g);
  };
  const panel = (box, title, lo, hi, ticks, fmt, refs, kind) => {
    const g = el("g", {}, svg);
    const Y = (v) => box.top + (1 - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * box.h;
    for (const a of rep.alerts.filter((x) => x.kind === kind)) {
      const a0 = X(parseT(a.from).getTime()), a1 = X(parseT(a.to).getTime());
      el("rect", { x: a0, y: box.top, width: Math.max(2, a1 - a0), height: box.h, class: "alert-band" }, g);
    }
    for (const v of ticks) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid-major" }, g); svgText(g, m.l - 6, Y(v) + 4, fmt(v), "axis-text", "end"); }
    for (const [v, label] of refs) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "ref" }, g); svgText(g, W - m.r - 4, Y(v) - 4, label, "axis-text", "end"); }
    svgText(g, m.l, box.top - 5, title, "panel-title");
    return { g, Y };
  };
  const errMax = Math.max(0.05, ...r.mae.filter((v) => v !== null), ...r.persistence_mae.filter((v) => v !== null));
  const top = Math.ceil(errMax * 1.1 * 20) / 20, step = top > 0.2 ? 0.1 : 0.05;
  const ticksE = []; for (let v = 0; v <= top + 1e-9; v += step) ticksE.push(+v.toFixed(2));
  const p1 = panel(boxes[0], "Average error around each point (wt% Si)", 0, top, ticksE, (v) => fmtNum(v, 2), [], "error");
  line(r.persistence_mae, p1.Y, "line-persist", p1.g); line(r.mae, p1.Y, "line-model", p1.g);
  const p2 = panel(boxes[1], "Measured values inside the 90 % interval", 0.4, 1.0, [0.4, 0.6, 0.8, 1.0], (v) => fmtPct(v),
    [[rep.level, "target"], [rep.thresholds.coverage, "alert below"]], "coverage");
  line(r.coverage, p2.Y, "line-cov", p2.g);
  if (hasFault) {
    const p3 = panel(boxes[2], "Casts with replaced sensor readings", 0, 1, [0, 0.5, 1], (v) => fmtPct(v), [[rep.thresholds.fault_share, "alert above"]], "inputs");
    line(r.fault_share, p3.Y, "line-fault", p3.g);
  }
  // month labels along the bottom
  const d = new Date(t0); d.setDate(1); d.setHours(0, 0, 0, 0); d.setMonth(d.getMonth() + 1);
  const months = []; for (let x = new Date(d); x.getTime() <= t1; x.setMonth(x.getMonth() + 1)) months.push(new Date(x));
  const every = Math.max(1, Math.ceil(months.length / Math.max(2, Math.floor((W - m.l - m.r) / 70))));
  months.forEach((x, i) => {
    if (i % every) return;
    el("line", { x1: X(x.getTime()), x2: X(x.getTime()), y1: m.t + 16, y2: H - m.b, class: "ref", opacity: 0.35 }, svg);
    svgText(svg, X(x.getTime()), H - m.b + 17, x.toLocaleDateString("en-GB", { month: "short", year: "2-digit" }), "axis-text", "middle");
  });
}

/* ---------------------------------------------------------------- set-up */
function applyModelMeta(meta) {
  if (!meta) return;
  if (meta.typical_si_range) state.typical = meta.typical_si_range;
  if (meta.interval_level) state.level = meta.interval_level;
}

function moveThumb(animate = true) {
  const sw = document.querySelector(".mode-switch"), thumb = sw && sw.querySelector(".tab-thumb");
  const on = sw && sw.querySelector('[aria-selected="true"]');
  if (!thumb || !on) return;
  if (!animate) thumb.style.transition = "none";
  thumb.style.width = `${on.offsetWidth}px`; thumb.style.height = `${on.offsetHeight}px`;
  thumb.style.transform = `translate(${on.offsetLeft}px, ${on.offsetTop}px)`;
  sw.classList.add("has-thumb");
  if (!animate) requestAnimationFrame(() => { thumb.style.transition = ""; });
}
const TABS = [["tab-replay", "panel-replay"], ["tab-files", "panel-files"], ["tab-train", "panel-train"], ["tab-health", "panel-health"]];
function selectTab(which) {
  for (const [tab, panel] of TABS) {
    const on = tab === which;
    $(tab).setAttribute("aria-selected", on); $(tab).tabIndex = on ? 0 : -1; $(panel).hidden = !on;
  }
  document.body.dataset.tab = which;
  moveThumb();
  stopPlayback(); showError("");
  state.measured = []; state.predicted = []; state.replay = null; state.xDomain = null; state.yMax = null;
  $("running").textContent = ""; setReadout(null); render();
  if (which === "tab-train" && train.current && !$("train-report").hidden) showReport(train.current.furnace_id);   // put its chart back
  if (which === "tab-health") loadHealth();
}

/* ---------------------------------------------------------------- light / dark */
function setIcon(t) {                // the icon and label offer the OTHER theme
  const next = t === "dark" ? "light" : "dark";
  $("theme-toggle").dataset.icon = next === "dark" ? "moon" : "sun";
  $("theme-label").textContent = next === "dark" ? "Dark mode" : "Light mode";
}
function applyTheme(t, remember, icon = true) {
  document.documentElement.dataset.theme = t;
  const next = t === "dark" ? "light" : "dark";
  if (icon) setIcon(t);
  $("theme-toggle").setAttribute("aria-pressed", String(t === "dark"));
  $("theme-toggle").setAttribute("aria-label", `Switch to ${next} mode`);
  if (remember) { try { localStorage.setItem("si-theme", t); } catch (_) { /* storage blocked: theme lasts for this visit */ } }
  // charts read colours through CSS variables, so a redraw is only needed for the gauge's tick text style
  if (state.predicted.length) drawGauge(state.predicted[state.predicted.length - 1]);
}
function initTheme() {
  const root = document.documentElement;
  root.classList.add("icon-instant");                     // no morph on page load
  applyTheme(root.dataset.theme === "dark" ? "dark" : "light", false);
  requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove("icon-instant")));
  $("theme-toggle").addEventListener("click", () => switchTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
}

let themeBusy = false;
const THEME_CANVAS = { light: "#f4f5f7", dark: "#0f1316" };      // must match --canvas in app.css
const GROW = { duration: 900, easing: "cubic-bezier(0.45, 0, 0.15, 1)" };   // slow start, long soft landing
const SHRINK = { duration: 780, easing: "cubic-bezier(0.55, 0, 0.25, 1)" };

/* keyframes for a circular mask centred on (x, y): its box is S px square, so the opaque part (84 % of the radius) just
   covers the farthest corner at full size and the rim stays soft (16 % of the radius) */
function circleFrames(x, y, far, grow) {
  const S = Math.ceil((2 * far) / 0.84) + 4;
  const small = { maskSize: "0px 0px", maskPosition: `${x}px ${y}px`, webkitMaskSize: "0px 0px", webkitMaskPosition: `${x}px ${y}px` };
  const big = { maskSize: `${S}px ${S}px`, maskPosition: `${x - S / 2}px ${y - S / 2}px`,
                webkitMaskSize: `${S}px ${S}px`, webkitMaskPosition: `${x - S / 2}px ${y - S / 2}px` };
  return grow ? [small, big] : [big, small];
}

async function switchTheme(next) {
  const root = document.documentElement, btn = $("theme-toggle");
  if (themeBusy) return;
  themeBusy = true;
  const grow = next === "dark";
  const r = btn.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
  const far = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
  try {
    if (reduceMotion) {                                   // no movement: a quick, plain fade
      const veil = document.createElement("div");
      veil.className = "theme-veil"; veil.style.cssText = `-webkit-mask:none;mask:none;background:${THEME_CANVAS[root.dataset.theme]}`;
      document.body.appendChild(veil);
      applyTheme(next, true);
      await veil.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 220, easing: "ease-out" }).finished;
      veil.remove();
    } else if (document.startViewTransition) {           // reveal the real page through the circle
      root.dataset.vt = grow ? "expand" : "contract";
      root.classList.add("icon-instant");               // the snapshot must show the finished icon; its layer animates the change
      const t = document.startViewTransition(() => applyTheme(next, true));
      await t.ready;
      root.animate(circleFrames(x, y, far, grow), { ...(grow ? GROW : SHRINK), fill: "both",
        pseudoElement: grow ? "::view-transition-new(root)" : "::view-transition-old(root)" });
      await t.finished;
    } else {                                              // older browsers: a veil in the other theme's colour
      const veil = document.createElement("div");
      veil.className = "theme-veil";
      veil.style.background = THEME_CANVAS[grow ? "dark" : root.dataset.theme];
      document.body.appendChild(veil);
      if (grow) {
        await veil.animate(circleFrames(x, y, far, true), { ...GROW, fill: "both" }).finished;
        applyTheme(next, true);
        await veil.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 260, easing: "ease-out", fill: "both" }).finished;
      } else {
        applyTheme(next, true, false);                    // the light page is ready underneath; the dark veil draws away
        await veil.animate(circleFrames(x, y, far, false), { ...SHRINK, fill: "both" }).finished;
        setIcon(next);                                    // the button is uncovered last, so its icon morphs now
      }
      veil.remove();
    }
  } catch (_) {
    applyTheme(next, true);                               // whatever happens, end on the chosen theme
  } finally {
    delete root.dataset.vt;
    root.classList.remove("icon-instant");
    themeBusy = false;
  }
}

async function init() {
  initTheme();
  moveThumb(false);
  new ResizeObserver(() => moveThumb(false)).observe(document.querySelector(".mode-switch"));
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => moveThumb(false));   // tab widths change once Geist loads
  TABS.forEach(([id], i) => {
    $(id).addEventListener("click", () => selectTab(id));
    $(id).addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const other = TABS[(i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length][0];
        selectTab(other); $(other).focus();
      }
    });
  });
  initTraining();
  $("replay-form").addEventListener("submit", startReplay);
  $("replay-skip").addEventListener("click", () => { if (state.replay) { stopPlayback(); revealTo(state.replay.rows.length); } });
  $("files-form").addEventListener("submit", predictFromFiles);
  $("replay-preset").addEventListener("change", (e) => { if (e.target.value) $("replay-start").value = e.target.value; });
  $("chart").setAttribute("tabindex", "0");
  new ResizeObserver(() => { drawChart(); if (state.predicted.length) drawGauge(state.predicted[state.predicted.length - 1]); }).observe($("chart"));
  setReadout(null); render();

  try {
    const h = await (await fetch("/health")).json();
    applyModelMeta(h.model);
    const d = parseT(h.trained_through);
    $("model-line").textContent = `Live model trained on data up to ${d.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" })}, with ${Math.round(state.level * 100)} % prediction intervals.`;
  } catch (_) { $("model-line").textContent = "The model server is not responding."; }

  try {
    const m = await (await fetch("/api/model")).json();
    if (m.required_sensor_columns) $("sensor-cols").textContent = m.required_sensor_columns.join(", ");
  } catch (_) { /* keep the short list */ }

  $("replay-go").disabled = true; $("replay-go").textContent = "Loading replay data…";
  const info = await (await fetch("/api/demo/info")).json();
  $("replay-go").disabled = false; $("replay-go").textContent = "Play replay";
  if (!info.available) {
    $("replay-form").hidden = true;
    const n = $("replay-unavailable"); n.hidden = false; n.textContent = `Replay is not available: ${info.reason}`;
    return;
  }
  applyModelMeta(info.model);
  $("held-from").textContent = parseT(info.model.trained_through).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  const sel = $("replay-preset");
  for (const s of info.suggested_starts) { const o = document.createElement("option"); o.value = toLocalInput(s.start); o.textContent = s.label; sel.appendChild(o); }
  const custom = document.createElement("option"); custom.value = ""; custom.textContent = "Custom start"; sel.appendChild(custom);
  const inp = $("replay-start");
  inp.min = toLocalInput(info.held_out_from); inp.max = toLocalInput(info.held_out_to); inp.value = sel.options[0].value;
  inp.addEventListener("input", () => { sel.value = ""; });
  $("replay-casts").max = info.max_casts_per_replay;
}

init();