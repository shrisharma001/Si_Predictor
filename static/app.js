/* Hot-metal silicon interface: replay of held-out casts and prediction from uploaded files. No dependencies. */
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
  const W = svg.clientWidth || 320, H = 76, padL = 8, padR = 8, y0 = 30, h = 16;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!p || p.v === null) { cap.textContent = ""; return; }
  const hiVal = Math.max(1.0, (p.hi ?? p.v) + 0.1, (p.actual ?? 0) + 0.1);
  const max = Math.ceil(hiVal * 5) / 5;
  const x = (v) => padL + (Math.max(0, Math.min(v, max)) / max) * (W - padL - padR);
  el("rect", { x: padL, y: y0, width: W - padL - padR, height: h, fill: "#e9eeea", rx: 2 }, svg);
  if (state.typical) el("rect", { x: x(state.typical[0]), y: y0, width: x(state.typical[1]) - x(state.typical[0]), height: h, fill: "var(--typical)", stroke: "#a9bfa6" }, svg);
  if (p.lo !== null && p.hi !== null) el("rect", { x: x(p.lo), y: y0 + 3, width: Math.max(2, x(p.hi) - x(p.lo)), height: h - 6, fill: "var(--band)", stroke: "var(--band-edge)" }, svg);
  el("line", { x1: x(p.v), x2: x(p.v), y1: y0 - 6, y2: y0 + h + 6, stroke: "var(--predicted)", "stroke-width": 3 }, svg);
  if (p.actual !== undefined && p.actual !== null) {
    const ax = x(p.actual);
    el("path", { d: `M${ax - 6},${y0 - 14} L${ax + 6},${y0 - 14} L${ax},${y0 - 5} Z`, fill: "var(--measured)" }, svg);
  }
  const step = max > 1.6 ? 0.5 : 0.2;
  for (let v = 0; v <= max + 1e-9; v += step) {
    el("line", { x1: x(v), x2: x(v), y1: y0 + h, y2: y0 + h + 4, stroke: "#8a9690" }, svg);
    svgText(svg, x(v), y0 + h + 17, fmtNum(v, 1), "axis-text", v === 0 ? "start" : (v + step > max + 1e-9 ? "end" : "middle")).setAttribute("style", "fill:var(--ink-2);font:12px var(--font)");
  }
  let txt = `${Math.round(state.level * 100)} % interval ${fmtNum(p.lo)} to ${fmtNum(p.hi)} wt%.`;
  if (state.typical) txt += ` Typical range ${fmtNum(state.typical[0])} to ${fmtNum(state.typical[1])} (shaded).`;
  if (p.actual !== undefined && p.actual !== null) txt += ` Measured ${fmtNum(p.actual)} (blue marker).`;
  cap.textContent = txt;
}

/* ---------------------------------------------------------------- chart */
function niceTimeStep(hours) {
  const steps = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336];
  for (const s of steps) if (hours / s <= 9) return s;
  return 720;
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
  const hours = (t1 - t0) / 3.6e6, stepH = niceTimeStep(hours), minorH = stepH >= 24 ? stepH / 4 : Math.max(stepH / 3, 1);
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
  const btn = $("files-go");
  btn.disabled = true; btn.textContent = "Reading files…";
  try {
    const body = await readJson(await fetch("/api/predict/upload", { method: "POST", body: fd }));
    applyModelMeta(body.model);
    const r = body.results[0];
    state.measured = body.context.si_history.map((p) => ({ t: parseT(p.time), v: p.si }));
    state.predicted = [toPred({ ...r, actual: undefined })];
    state.xDomain = null; state.yMax = null; state.replay = null;
    setReadout(state.predicted[0]);
    $("running").textContent = `Used ${body.context.sensor_rows_used} hourly sensor rows from ${fmtTime(parseT(body.context.sensor_from))} to ${fmtTime(parseT(body.context.sensor_to))}, and the ${state.measured.length} lab results shown before the cast.`;
    render();
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Predict next cast";
  }
}

/* ---------------------------------------------------------------- set-up */
function applyModelMeta(meta) {
  if (!meta) return;
  if (meta.typical_si_range) state.typical = meta.typical_si_range;
  if (meta.interval_level) state.level = meta.interval_level;
}

function selectTab(which) {
  for (const [tab, panel] of [["tab-replay", "panel-replay"], ["tab-files", "panel-files"]]) {
    const on = tab === which;
    $(tab).setAttribute("aria-selected", on); $(tab).tabIndex = on ? 0 : -1; $(panel).hidden = !on;
  }
  stopPlayback(); showError("");
  state.measured = []; state.predicted = []; state.replay = null; state.xDomain = null; state.yMax = null;
  $("running").textContent = ""; setReadout(null); render();
}

async function init() {
  for (const id of ["tab-replay", "tab-files"]) {
    $(id).addEventListener("click", () => selectTab(id));
    $(id).addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const other = id === "tab-replay" ? "tab-files" : "tab-replay"; selectTab(other); $(other).focus(); }
    });
  }
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
