// Trends from Tiger Data (see docs/adr/0004): Confirmed Falls per day and Motion Events per hour,
// drawn as small SVG charts. Real runs only; Simulated Falls never reach these numbers.

const NS = "http://www.w3.org/2000/svg";
const PAD = { top: 8, right: 6, bottom: 20, left: 26 };
const GAP = 2;
const ROUND = 4;

// Fixed order and color per outcome; checked with the dataviz palette validator against the page.
export const OUTCOMES = [
  { key: "settled", color: "var(--calm)", dev: "No Alert (OK or cancelled)", care: "No alert needed" },
  { key: "open", color: "var(--watch)", dev: "Not closed", care: "Not closed yet" },
  { key: "alerted", color: "var(--urgent)", dev: "Alert sent", care: "You were alerted" },
];

export async function fetchTrends() {
  const r = await fetch("/trends", { cache: "no-store" });
  if (!r.ok) throw new Error(`trends ${r.status}`);
  return r.json();
}

function node(name, attrs = {}, parent = null, ns = NS) {
  const e = ns ? document.createElementNS(ns, name) : document.createElement(name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}

function html(name, cls, text, parent) {
  const e = document.createElement(name);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  if (parent) parent.append(e);
  return e;
}

function niceMax(v, floor) {
  if (v <= floor) return floor;
  const step = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * step >= v) return m * step;
  return 10 * step;
}

const dayLabel = (iso) => {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, { month: "short", day: "numeric" });
};
const hourLabel = (t) => new Date(t * 1000).toLocaleTimeString(undefined, { hour: "numeric" });
const fmt = (v, digits) => (v == null ? "–" : Number(v).toFixed(digits));

// A bar's data end is rounded; its baseline end stays square.
function barPath(x, y, w, h, round) {
  const r = round ? Math.min(ROUND, w / 2, h) : 0;
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

function tooltip() {
  let tip = document.querySelector(".tip.trend-tip");
  if (!tip) {
    tip = html("div", "tip trend-tip", null, document.body);
    tip.hidden = true;
  }
  return tip;
}

function placeTip(tip, e) {
  tip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8)}px`;
  tip.style.top = `${Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8)}px`;
}

/** One chart: stacked bars, or a line when `line` is set. Shared axes, grid and per-column hover. */
function chart(host, { n, value, parts, line, ref, ticks, floor, tip, label }) {
  const width = Math.max(host.clientWidth, 160);
  const height = Math.max(host.clientHeight, 90);
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const values = Array.from({ length: n }, (_, i) => value(i));
  const max = niceMax(Math.max(0, ref ?? 0, ...values.filter((v) => v != null)), floor);
  const y = (v) => PAD.top + plotH * (1 - v / max);
  const band = plotW / n;
  const svg = node("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label });

  for (const v of [0, max / 2, max]) {
    node("line", { x1: PAD.left, x2: width - PAD.right, y1: y(v), y2: y(v), class: v ? "trend-grid" : "trend-axis" }, svg);
    node("text", { x: PAD.left - 5, y: y(v) + 3.5, "text-anchor": "end", class: "trend-tick" }, svg)
      .textContent = fmt(v, max < 2 && v ? 2 : 0);
  }
  for (let i = 0; i < n; i++) {
    const text = ticks(i);
    if (text) node("text", { x: PAD.left + band * (i + 0.5), y: height - 5, "text-anchor": "middle", class: "trend-tick" }, svg)
      .textContent = text;
  }
  const marks = node("g", {}, svg);
  if (line) {
    if (ref != null) {
      node("line", { x1: PAD.left, x2: width - PAD.right, y1: y(ref), y2: y(ref), class: "trend-ref" }, svg);
      node("text", { x: width - PAD.right, y: y(ref) - 4, "text-anchor": "end", class: "trend-tick" }, svg)
        .textContent = line.refLabel;
    }
    let d = "";
    values.forEach((v, i) => {
      if (v == null) return;
      d += `${d && values[i - 1] != null ? "L" : "M"}${PAD.left + band * (i + 0.5)},${y(v)}`;
    });
    node("path", { d, class: "trend-line" }, marks);
    values.forEach((v, i) => {
      if (v != null) node("circle", { cx: PAD.left + band * (i + 0.5), cy: y(v), r: 4, class: "trend-dot" }, marks);
    });
  } else {
    const w = Math.max(band - GAP, 1);
    for (let i = 0; i < n; i++) {
      const segs = parts(i).filter((p) => p.value > 0);
      let base = PAD.top + plotH;
      segs.forEach((p, k) => {
        const h = plotH * (p.value / max) - (k ? GAP : 0);
        if (h <= 0) return;
        base -= k ? GAP : 0;
        node("path", { d: barPath(PAD.left + band * i + GAP / 2, base - h, w, h, k === segs.length - 1),
                       style: `fill: ${p.color}` }, marks);
        base -= h;
      });
    }
  }
  const t = tooltip();
  const wash = node("rect", { y: PAD.top, height: plotH, width: band, class: "trend-wash", visibility: "hidden" }, svg);
  svg.insertBefore(wash, marks);
  for (let i = 0; i < n; i++) {
    const hit = node("rect", { x: PAD.left + band * i, y: 0, width: band, height, class: "trend-hit" }, svg);
    hit.addEventListener("pointerenter", (e) => {
      wash.setAttribute("x", PAD.left + band * i);
      wash.setAttribute("visibility", "visible");
      t.replaceChildren(...tip(i));
      t.hidden = false;
      placeTip(t, e);
    });
    hit.addEventListener("pointermove", (e) => placeTip(t, e));
    hit.addEventListener("pointerleave", () => {
      wash.setAttribute("visibility", "hidden");
      t.hidden = true;
    });
  }
  host.replaceChildren(svg);
}

function tipLines(head, ...rows) {
  const b = html("b", null, head);
  return [b, ...rows.flatMap((r) => [document.createElement("br"), document.createTextNode(r)])];
}

function legend(parent, mode) {
  const ul = html("ul", "trend-legend", null, parent);
  for (const o of OUTCOMES) {
    const li = html("li", null, null, ul);
    html("span", "swatch", null, li).style.background = o.color;
    li.append(o[mode]);
  }
}

function table(parent, head, rows) {
  const box = html("details", "trend-table", null, parent);
  html("summary", null, "Table", box);
  const tbl = html("table", null, null, box);
  const tr = html("tr", null, null, html("thead", null, null, tbl));
  for (const h of head) html("th", null, h, tr);
  const body = html("tbody", null, null, tbl);
  for (const row of rows) {
    const r = html("tr", null, null, body);
    for (const c of row) html("td", null, String(c), r);
  }
}

function section(parent, title, note) {
  const s = html("section", "trend", null, parent);
  const h = html("header", null, null, s);
  html("h3", null, title, h);
  if (note) html("span", "trend-note", note, h);
  return s;
}

const settled = (d) => d.ok + d.cancelled;

/**
 * Draws Trends into `host`. mode "dev" adds the Room Change chart and uses the glossary's words;
 * mode "care" uses plain words for the Caregiver and leaves Room Change out.
 */
export function renderTrends(host, data, mode, cfg = null) {
  const dev = mode === "dev";
  const { days, hours } = data;
  host.replaceChildren();
  const plots = [];

  const falls = section(host, dev ? "Confirmed Falls per day" : "Falls per day",
    dev ? "last 14 days" : null);
  legend(falls, mode);
  plots.push([html("div", "trend-plot", null, falls), {
    n: days.length, floor: 4,
    value: (i) => days[i].falls,
    parts: (i) => [
      { value: settled(days[i]), color: OUTCOMES[0].color },
      { value: days[i].open, color: OUTCOMES[1].color },
      { value: days[i].alerted, color: OUTCOMES[2].color },
    ],
    ticks: (i) => ((days.length - 1 - i) % 4 === 0 ? dayLabel(days[i].day) : ""),
    tip: (i) => {
      const d = days[i];
      const word = d.falls === 1 ? "fall" : "falls";
      if (!d.falls) return tipLines(dayLabel(d.day), dev ? "no Confirmed Falls" : "no falls");
      return dev
        ? tipLines(`${dayLabel(d.day)} · ${d.falls} ${word}`, `Resident OK ${d.ok} · Caregiver cancelled ${d.cancelled}`,
                   `Alert sent ${d.alerted} · not closed ${d.open}`)
        : tipLines(`${dayLabel(d.day)} · ${d.falls} ${word}`, `No alert needed ${settled(d)}`,
                   `You were alerted ${d.alerted}`, ...(d.open ? [`Not closed yet ${d.open}`] : []));
    },
    label: `${dev ? "Confirmed Falls" : "Falls"} per day over the last 14 days, by how each ended`,
  }]);
  table(falls, ["Day", "Falls", "No alert needed", "Alert sent", "Not closed"],
    days.map((d) => [d.day, d.falls, settled(d), d.alerted, d.open]));

  const moves = section(host, dev ? "Motion Events per hour" : "Movement, last 24 hours", dev ? "last 24 hours" : null);
  plots.push([html("div", "trend-plot", null, moves), {
    n: hours.length, floor: 4,
    value: (i) => hours[i].events,
    parts: (i) => [{ value: hours[i].events, color: "var(--series)" }],
    ticks: (i) => ((hours.length - 1 - i) % 6 === 0 ? hourLabel(hours[i].hour) : ""),
    tip: (i) => tipLines(hourLabel(hours[i].hour),
      dev ? `${hours[i].events} Motion Event${hours[i].events === 1 ? "" : "s"}`
          : `moved ${hours[i].events} time${hours[i].events === 1 ? "" : "s"}`),
    label: `${dev ? "Motion Events" : "Movement"} per hour over the last 24 hours`,
  }]);
  table(moves, ["Hour", dev ? "Motion Events" : "Times moved"], hours.map((h) => [hourLabel(h.hour), h.events]));

  if (dev) {
    const change = section(host, "Room Change per hour", "average; dashed = T_STATIC");
    plots.push([html("div", "trend-plot", null, change), {
      n: hours.length, floor: 0.5, line: { refLabel: "T_STATIC" },
      ref: cfg?.t_static ?? null,
      value: (i) => hours[i].room_change,
      ticks: (i) => ((hours.length - 1 - i) % 6 === 0 ? hourLabel(hours[i].hour) : ""),
      tip: (i) => tipLines(hourLabel(hours[i].hour), `average Room Change ${fmt(hours[i].room_change, 2)}`),
      label: "Average Room Change of Motion Events per hour over the last 24 hours",
    }]);
    table(change, ["Hour", "Average Room Change"], hours.map((h) => [hourLabel(h.hour), fmt(h.room_change, 2)]));
  }
  // Charts are sized from their boxes, so they are drawn once the sections are in the page.
  for (const [plot, opts] of plots) chart(plot, opts);
}

export function renderTrendsMessage(host, text) {
  host.replaceChildren(html("p", "trend-message", text));
}
