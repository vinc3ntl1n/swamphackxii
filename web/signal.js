export const WAVE_ROWS = [63, 88, 94, 108, 133, 141, 147, 156];
export const SPAN_S = 20;
export const GAP_S = 0.25;
export const GUTTER = { left: 48, right: 66 };
const WAVE_WIDTH = 166;
const LANES = 8;
const STEP_S = 0.05;
const MEDIAN_S = 20;
const MEDIAN_EVERY_S = 1;
const KEEP_S = 70;
const STRIP = 2048;
const WAVE_DB = 8;
const CHANGE_DB = 12;
const NEUTRAL_DB = 1.5;
const AMP_MAX = 110;
const SHAPE_MAX = 120;
const TRAIL_S = 0.5;
const ENERGY_FLOOR = 30;
const FONT = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
const BOLD = '600 12px system-ui, -apple-system, "Segoe UI", sans-serif';

export function lanesFor(width) {
  if (width === WAVE_WIDTH) return WAVE_ROWS.slice();
  if (width < 1) return [];
  return Array.from({ length: LANES }, (_, i) => Math.round((i * (width - 1)) / (LANES - 1)));
}

function hexToRgb(hex) {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function toLinear(c) {
  const x = c / 255;
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
}

function fromLinear(c) {
  const x = Math.min(1, Math.max(0, c));
  return Math.round(255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055));
}

function toOklab(rgb) {
  const [r, g, b] = rgb.map(toLinear);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function fromOklab([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    fromLinear(4.0767416621 * l - 3.3077085842 * m + 0.2309968829 * s),
    fromLinear(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    fromLinear(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  ];
}

function ramp(stops, n) {
  const labs = stops.map((hex) => toOklab(hexToRgb(hex)));
  const out = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (labs.length - 1);
    const k = Math.min(labs.length - 2, Math.floor(x));
    const f = x - k;
    out.set(fromOklab(labs[k].map((c, j) => c + (labs[k + 1][j] - c) * f)), i * 3);
  }
  return out;
}

export function readTheme(root = document.documentElement) {
  const css = getComputedStyle(root);
  const v = (name) => css.getPropertyValue(name).trim();
  const list = (name) => v(name).split(/\s+/).filter(Boolean);
  const mid = v("--heat-mid");
  return {
    surface: v("--surface"),
    ink: v("--ink"),
    ink2: v("--ink-2"),
    muted: v("--muted"),
    grid: v("--grid"),
    axis: v("--axis"),
    series: v("--series"),
    seriesWash: v("--series-wash"),
    windowWash: v("--window-wash"),
    amp: ramp(list("--heat-seq"), 256),
    neg: ramp([mid, ...list("--heat-neg")], 128),
    pos: ramp([mid, ...list("--heat-pos")], 128),
  };
}

export function watchTheme(onChange) {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", onChange);
  const options = { attributes: true, attributeFilter: ["data-theme"] };
  new MutationObserver(onChange).observe(document.documentElement, options);
}

function writeAmp(data, o, v, theme) {
  if (!(v === v)) {
    data[o + 3] = 0;
    return;
  }
  const i = Math.min(255, Math.max(0, Math.round((v / AMP_MAX) * 255))) * 3;
  data[o] = theme.amp[i];
  data[o + 1] = theme.amp[i + 1];
  data[o + 2] = theme.amp[i + 2];
  data[o + 3] = 255;
}

function writeChange(data, o, d, theme) {
  if (!(d === d)) {
    data[o + 3] = 0;
    return;
  }
  const a = Math.abs(d);
  const f = a < NEUTRAL_DB ? 0 : Math.min(1, (a - NEUTRAL_DB) / (CHANGE_DB - NEUTRAL_DB));
  const lut = d < 0 ? theme.neg : theme.pos;
  const i = Math.round(f * 127) * 3;
  data[o] = lut[i];
  data[o + 1] = lut[i + 1];
  data[o + 2] = lut[i + 2];
  data[o + 3] = 255;
}

export class SignalBuffer {
  constructor() {
    this.reset(0);
  }

  reset(width) {
    this.width = width;
    this.lanes = lanesFor(width);
    this.frames = [];
    this.medians = null;
    this.medianT = -Infinity;
    this.count = 0;
  }

  push(msg, arrival) {
    const row = Array.isArray(msg.row) ? Float32Array.from(msg.row, (x) => (x == null ? NaN : x)) : null;
    if (row && row.length !== this.width) this.reset(row.length);
    const last = this.frames[this.frames.length - 1];
    if (last && msg.t <= last.t) return null;
    const f = {
      t: msg.t,
      energy: typeof msg.energy === "number" ? msg.energy : null,
      level: msg.level,
      rssi: msg.rssi,
      hz: msg.hz,
      row,
      db: null,
      slot: this.count++ % STRIP,
      arrival,
    };
    this.frames.push(f);
    while (this.frames.length >= STRIP || f.t - this.frames[0].t > KEEP_S) this.frames.shift();
    if (row) {
      if (!this.medians || f.t - this.medianT >= MEDIAN_EVERY_S) this.updateMedians(f.t);
      f.db = new Float32Array(row.length);
      for (let k = 0; k < row.length; k++) f.db[k] = 20 * Math.log10((row[k] + 1) / (this.medians[k] + 1));
    }
    return f;
  }

  updateMedians(t) {
    const rows = [];
    for (let i = this.frames.length - 1; i >= 0 && t - this.frames[i].t <= MEDIAN_S; i--) {
      if (this.frames[i].row) rows.push(this.frames[i].row);
    }
    const m = new Float32Array(this.width);
    const col = new Float32Array(rows.length);
    for (let k = 0; k < this.width; k++) {
      let n = 0;
      for (const r of rows) if (r[k] === r[k]) col[n++] = r[k];
      if (!n) {
        m[k] = NaN;
        continue;
      }
      const s = col.subarray(0, n).sort();
      m[k] = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    }
    this.medians = m;
    this.medianT = t;
  }

  latest() {
    return this.frames[this.frames.length - 1] || null;
  }

  now(perfNow) {
    const last = this.latest();
    return last ? last.t + Math.min(0.2, Math.max(0, (perfNow - last.arrival) / 1000)) : null;
  }

  from(t) {
    let lo = 0;
    let hi = this.frames.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.frames[mid].t < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  nearest(t) {
    const i = this.from(t);
    const a = this.frames[i - 1];
    const b = this.frames[i];
    if (!a) return b || null;
    if (!b) return a;
    return t - a.t <= b.t - t ? a : b;
  }

  between(a, b) {
    const out = [];
    for (let i = this.from(a); i < this.frames.length && this.frames[i].t <= b; i++) out.push(this.frames[i]);
    return out;
  }
}

class View {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.w = 0;
    this.h = 0;
  }

  begin() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return false;
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (this.canvas.width !== pw || this.canvas.height !== ph) {
      this.canvas.width = pw;
      this.canvas.height = ph;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.clearRect(0, 0, w, h);
    this.w = w;
    this.h = h;
    return true;
  }

  plot(top, bottom) {
    const x0 = GUTTER.left;
    const x1 = Math.max(x0 + 20, this.w - GUTTER.right);
    return { x0, x1, y0: top, y1: Math.max(top + 10, this.h - bottom), h: this.h, pps: (x1 - x0) / SPAN_S };
  }

  clip(p) {
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.rect(p.x0, 0, p.x1 - p.x0, p.h);
    this.ctx.clip();
  }

  crosshair(p, hoverX, theme) {
    if (hoverX == null || hoverX < p.x0 || hoverX > p.x1) return;
    const ctx = this.ctx;
    ctx.strokeStyle = theme.ink2;
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(Math.round(hoverX) + 0.5, p.y0);
    ctx.lineTo(Math.round(hoverX) + 0.5, p.y1);
    ctx.stroke();
  }
}

export function xAt(p, now, t) {
  return p.x1 - (now - t) * p.pps;
}

export function timeAtX(canvas, clientX, now) {
  const r = canvas.getBoundingClientRect();
  const x = clientX - r.left;
  const x0 = GUTTER.left;
  const x1 = Math.max(x0 + 20, r.width - GUTTER.right);
  if (x < x0 || x > x1) return null;
  return { x, t: now - (x1 - x) / ((x1 - x0) / SPAN_S) };
}

function vline(ctx, x, y0, y1) {
  ctx.beginPath();
  ctx.moveTo(Math.round(x) + 0.5, y0);
  ctx.lineTo(Math.round(x) + 0.5, y1);
  ctx.stroke();
}

function tag(ctx, p, text, x, y, theme, font) {
  if (x < p.x0 || x > p.x1) return;
  ctx.font = font;
  const w = ctx.measureText(text).width;
  x = Math.min(p.x1 - w / 2 - 3, Math.max(p.x0 + w / 2 + 3, x));
  ctx.fillStyle = theme.surface;
  ctx.fillRect(x - w / 2 - 3, y - 1, w + 6, 14);
  ctx.fillStyle = theme.ink;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText(text, x, y);
}

export function drawMarks(view, p, now, clips, theme, { wash = true, labels = false } = {}) {
  const ctx = view.ctx;
  view.clip(p);
  ctx.setLineDash([]);
  for (const c of clips) {
    if (c.go == null) continue;
    const a = xAt(p, now, c.go - c.pre);
    const b = xAt(p, now, c.go + c.post);
    if (b < p.x0 || a > p.x1) continue;
    if (wash) {
      ctx.fillStyle = theme.windowWash;
      ctx.fillRect(a, p.y0, b - a, p.y1 - p.y0);
    }
    ctx.strokeStyle = theme.ink2;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1;
    vline(ctx, a, p.y0, p.y1);
    vline(ctx, b, p.y0, p.y1);
    ctx.globalAlpha = 1;
  }
  for (const c of clips) {
    ctx.strokeStyle = theme.muted;
    ctx.lineWidth = 1;
    for (const k of c.counts) vline(ctx, xAt(p, now, k.t), p.y0, p.y1);
    if (c.go != null) {
      ctx.strokeStyle = theme.ink;
      ctx.lineWidth = 2;
      vline(ctx, xAt(p, now, c.go), p.y0, p.y1);
    }
  }
  ctx.restore();
  if (!labels) return;
  view.clip(p);
  for (const c of clips) {
    for (const k of c.counts) tag(ctx, p, k.word, xAt(p, now, k.t), 1, theme, FONT);
    if (c.go == null) continue;
    tag(ctx, p, "GO", xAt(p, now, c.go), 1, theme, BOLD);
    const a = Math.max(p.x0, xAt(p, now, c.go + 0.6));
    const b = Math.min(p.x1, xAt(p, now, c.go + c.post));
    ctx.font = FONT;
    const text = "recorded clip";
    if (b - a > ctx.measureText(text).width + 12) tag(ctx, p, text, (a + b) / 2, 1, theme, FONT);
  }
  ctx.restore();
}

function axisLabel(ctx, text, x, y, theme, align = "right", baseline = "middle") {
  ctx.font = FONT;
  ctx.fillStyle = theme.muted;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  ctx.fillText(text, x, y);
}

export class Waveform extends View {
  constructor(canvas, buffer) {
    super(canvas);
    this.buffer = buffer;
  }

  layout() {
    const p = this.plot(17, 4);
    const lanes = this.buffer.lanes;
    const laneH = (p.y1 - p.y0) / Math.max(1, lanes.length);
    return { p, lanes, laneH, half: laneH * 0.46 };
  }

  laneAt(clientY) {
    const r = this.canvas.getBoundingClientRect();
    const { p, lanes, laneH } = this.layout();
    const i = Math.floor((clientY - r.top - p.y0) / laneH);
    return i >= 0 && i < lanes.length ? lanes[i] : null;
  }

  draw(now, clips, theme, hoverX) {
    if (!this.begin() || now == null) return;
    const ctx = this.ctx;
    const { p, lanes, laneH, half } = this.layout();
    drawMarks(this, p, now, clips, theme, { wash: true, labels: true });
    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 1;
    lanes.forEach((k, i) => {
      const c = Math.round(p.y0 + laneH * (i + 0.5)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p.x0, c);
      ctx.lineTo(p.x1, c);
      ctx.stroke();
      axisLabel(ctx, `#${k}`, p.x0 - 8, c, theme);
    });
    const frames = this.buffer.frames;
    const start = this.buffer.from(now - SPAN_S - 1);
    this.clip(p);
    ctx.strokeStyle = theme.series;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    lanes.forEach((k, i) => {
      const center = p.y0 + laneH * (i + 0.5);
      let prev = null;
      ctx.beginPath();
      for (let j = start; j < frames.length; j++) {
        const f = frames[j];
        const d = f.db ? f.db[k] : NaN;
        if (!(d === d)) {
          prev = null;
          continue;
        }
        const x = xAt(p, now, f.t);
        const y = center - Math.tanh(d / WAVE_DB) * half;
        if (prev == null || f.t - prev > GAP_S) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        prev = f.t;
      }
      ctx.stroke();
    });
    ctx.restore();
    this.crosshair(p, hoverX, theme);
  }
}

export class Heatmap extends View {
  constructor(canvas, buffer) {
    super(canvas);
    this.buffer = buffer;
    this.mode = "amp";
    this.width = -1;
    this.strips = null;
    this.theme = null;
  }

  setTheme(theme) {
    this.theme = theme;
    this.repaint();
  }

  ensure() {
    if (this.strips && this.width === this.buffer.width) return false;
    this.width = this.buffer.width;
    const n = Math.max(1, this.width);
    const make = () => {
      const c = document.createElement("canvas");
      c.width = STRIP;
      c.height = n;
      return c;
    };
    this.strips = { amp: make(), change: make() };
    this.columns = { amp: new ImageData(1, n), change: new ImageData(1, n) };
    return true;
  }

  add(f) {
    if (this.ensure()) this.repaint();
    else this.paint(f);
  }

  repaint() {
    this.ensure();
    for (const f of this.buffer.frames) this.paint(f);
  }

  paint(f) {
    if (!this.theme || !f.row || f.row.length !== this.width) return;
    const n = this.width;
    const amp = this.columns.amp.data;
    const change = this.columns.change.data;
    for (let k = 0; k < n; k++) {
      const o = (n - 1 - k) * 4;
      writeAmp(amp, o, f.row[k], this.theme);
      writeChange(change, o, f.db[k], this.theme);
    }
    this.strips.amp.getContext("2d").putImageData(this.columns.amp, f.slot, 0);
    this.strips.change.getContext("2d").putImageData(this.columns.change, f.slot, 0);
  }

  positionAt(clientY) {
    const r = this.canvas.getBoundingClientRect();
    const p = this.plot(4, 4);
    const k = Math.floor(((p.y1 - (clientY - r.top)) / (p.y1 - p.y0)) * this.width);
    return k >= 0 && k < this.width ? k : null;
  }

  draw(now, clips, theme, hoverX) {
    if (!this.begin() || now == null) return;
    const ctx = this.ctx;
    const p = this.plot(4, 4);
    const frames = this.buffer.frames;
    if (this.strips && this.width > 0) {
      const strip = this.strips[this.mode];
      ctx.imageSmoothingEnabled = false;
      this.clip(p);
      for (let i = this.buffer.from(now - SPAN_S - 1); i < frames.length; i++) {
        const f = frames[i];
        if (!f.row) continue;
        const prev = frames[i - 1];
        const w = prev && f.t - prev.t <= GAP_S ? f.t - prev.t : STEP_S;
        const xa = xAt(p, now, f.t - w);
        const xb = xAt(p, now, f.t);
        if (xb < p.x0) continue;
        ctx.drawImage(strip, f.slot, 0, 1, this.width, xa, p.y0, xb - xa + 0.75, p.y1 - p.y0);
      }
      ctx.restore();
    }
    drawMarks(this, p, now, clips, theme, { wash: false });
    for (let k = 0; k < this.width; k += 50) {
      axisLabel(ctx, String(k), p.x0 - 8, p.y1 - ((k + 0.5) / this.width) * (p.y1 - p.y0), theme);
    }
    this.crosshair(p, hoverX, theme);
  }
}

export function drawScale(canvas, theme, mode) {
  const w = canvas.width;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, 1);
  for (let x = 0; x < w; x++) {
    const f = x / (w - 1);
    if (mode === "amp") writeAmp(img.data, x * 4, f * AMP_MAX, theme);
    else writeChange(img.data, x * 4, (f * 2 - 1) * CHANGE_DB, theme);
  }
  for (let y = 0; y < canvas.height; y++) ctx.putImageData(img, 0, y);
}

export const SCALE_LABELS = { amp: ["0", String(AMP_MAX)], change: [`−${CHANGE_DB} dB`, `+${CHANGE_DB} dB`] };

function energyMax(values) {
  let m = 0;
  for (const e of values) if (e != null && e > m) m = e;
  return Math.max(ENERGY_FLOOR, Math.ceil(m / 5) * 5);
}

function energySegments(frames, x, y) {
  const segs = [];
  let seg = null;
  let prev = null;
  for (const f of frames) {
    if (f.energy == null) {
      seg = null;
      prev = null;
      continue;
    }
    if (!seg || prev == null || f.t - prev > GAP_S) {
      seg = [];
      segs.push(seg);
    }
    seg.push([x(f.t), y(f.energy)]);
    prev = f.t;
  }
  return segs;
}

function strokeSegments(ctx, segs, baseY, color, wash, width) {
  if (wash) {
    ctx.fillStyle = wash;
    for (const s of segs) {
      if (s.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(s[0][0], baseY);
      for (const [x, y] of s) ctx.lineTo(x, y);
      ctx.lineTo(s[s.length - 1][0], baseY);
      ctx.closePath();
      ctx.fill();
    }
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for (const s of segs) {
    ctx.beginPath();
    s.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
  }
}

function threshold(ctx, p, y, color, text, below, theme) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(p.x0, Math.round(y) + 0.5);
  ctx.lineTo(p.x1, Math.round(y) + 0.5);
  ctx.stroke();
  ctx.setLineDash([]);
  axisLabel(ctx, text, p.x1 + 6, below ? y + 1 : y - 1, theme, "left", below ? "top" : "bottom");
}

export class EnergyChart extends View {
  constructor(canvas, buffer) {
    super(canvas);
    this.buffer = buffer;
  }

  draw(now, clips, theme, hoverX, cfg) {
    if (!this.begin() || now == null) return;
    const ctx = this.ctx;
    const p = this.plot(8, 20);
    const frames = this.buffer.between(now - SPAN_S - 1, now + 1);
    const top = energyMax(frames.map((f) => f.energy));
    const y = (e) => p.y1 - (Math.min(e, top) / top) * (p.y1 - p.y0);
    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 1;
    for (let e = 0; e <= top; e += 10) {
      const yy = Math.round(y(e)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p.x0, yy);
      ctx.lineTo(p.x1, yy);
      ctx.stroke();
      axisLabel(ctx, String(e), p.x0 - 8, yy, theme);
    }
    drawMarks(this, p, now, clips, theme, { wash: true });
    if (cfg) {
      threshold(ctx, p, y(cfg.t_on), theme.ink2, `T_ON ${cfg.t_on}`, false, theme);
      threshold(ctx, p, y(cfg.t_off), theme.muted, `T_OFF ${cfg.t_off}`, true, theme);
    }
    this.clip(p);
    const segs = energySegments(frames, (t) => xAt(p, now, t), y);
    strokeSegments(ctx, segs, p.y1, theme.series, theme.seriesWash, 2);
    const last = segs.length ? segs[segs.length - 1] : null;
    const end = last ? last[last.length - 1] : null;
    if (end && end[0] > p.x1 - 0.5 * p.pps) {
      ctx.beginPath();
      ctx.arc(end[0], end[1], 6, 0, Math.PI * 2);
      ctx.fillStyle = theme.surface;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(end[0], end[1], 4, 0, Math.PI * 2);
      ctx.fillStyle = theme.series;
      ctx.fill();
    }
    ctx.restore();
    for (let s = SPAN_S; s >= 0; s -= 5) {
      const x = p.x1 - s * p.pps;
      const align = s === SPAN_S ? "left" : s === 0 ? "right" : "center";
      axisLabel(ctx, s ? `−${s} s` : "now", x, p.y1 + 5, theme, align, "top");
    }
    this.crosshair(p, hoverX, theme);
  }
}

export class ShapeChart extends View {
  constructor(canvas, buffer) {
    super(canvas);
    this.buffer = buffer;
  }

  draw(theme) {
    if (!this.begin()) return;
    const ctx = this.ctx;
    const p = this.plot(6, 18);
    const n = this.buffer.width;
    const y = (v) => p.y1 - (Math.min(v, SHAPE_MAX) / SHAPE_MAX) * (p.y1 - p.y0);
    const x = (k) => p.x0 + (n > 1 ? k / (n - 1) : 0.5) * (p.x1 - p.x0);
    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 1;
    for (let v = 0; v <= SHAPE_MAX; v += 50) {
      const yy = Math.round(y(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p.x0, yy);
      ctx.lineTo(p.x1, yy);
      ctx.stroke();
      axisLabel(ctx, String(v), p.x0 - 8, yy, theme);
    }
    if (n < 1) return;
    ctx.strokeStyle = theme.muted;
    for (const k of this.buffer.lanes) {
      ctx.beginPath();
      ctx.moveTo(Math.round(x(k)) + 0.5, p.y1);
      ctx.lineTo(Math.round(x(k)) + 0.5, p.y1 + 5);
      ctx.stroke();
    }
    axisLabel(ctx, "0", p.x0, p.y1 + 6, theme, "left", "top");
    axisLabel(ctx, String(n - 1), p.x1, p.y1 + 6, theme, "right", "top");
    const note = "subcarrier position (ticks = waveform lanes)";
    axisLabel(ctx, note, (p.x0 + p.x1) / 2, p.y1 + 6, theme, "center", "top");
    const last = this.buffer.latest();
    if (!last) return;
    const trail = this.buffer.between(last.t - TRAIL_S, last.t).filter((f) => f.row);
    ctx.lineJoin = "round";
    ctx.strokeStyle = theme.series;
    trail.forEach((f, i) => {
      const newest = i === trail.length - 1;
      ctx.globalAlpha = newest ? 1 : 0.08 + (0.4 * (i + 1)) / trail.length;
      ctx.lineWidth = newest ? 2 : 1;
      ctx.beginPath();
      let pen = false;
      for (let k = 0; k < f.row.length; k++) {
        const v = f.row[k];
        if (!(v === v)) {
          pen = false;
          continue;
        }
        if (pen) ctx.lineTo(x(k), y(v));
        else ctx.moveTo(x(k), y(v));
        pen = true;
      }
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }
}

export function sparkline(canvas, frames, go, cfg, theme) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const a = go - cfg.pre_s;
  const span = cfg.pre_s + cfg.post_s;
  const x = (t) => 1 + ((t - a) / span) * (w - 2);
  const top = energyMax(frames.map((f) => f.energy));
  const y = (e) => h - 2 - (Math.min(e, top) / top) * (h - 4);
  ctx.strokeStyle = theme.muted;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(0, Math.round(y(cfg.t_on)) + 0.5);
  ctx.lineTo(w, Math.round(y(cfg.t_on)) + 0.5);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = theme.ink;
  ctx.lineWidth = 1.5;
  vline(ctx, x(go), 0, h);
  if (!frames.some((f) => f.energy != null)) {
    axisLabel(ctx, "no data", w / 2, h / 2, theme, "center");
    return;
  }
  strokeSegments(ctx, energySegments(frames, x, y), h, theme.series, null, 1.5);
}

export function connect(onMessage, onState, url = `ws://${location.host}/ws`) {
  let ws = null;
  const open = () => {
    ws = new WebSocket(url);
    ws.onopen = () => onState("open");
    ws.onmessage = (e) => {
      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch {
        return;
      }
      onMessage(msg);
    };
    ws.onclose = () => {
      onState("closed");
      setTimeout(open, 1000);
    };
  };
  open();
  return {
    send(msg) {
      if (!ws || ws.readyState !== WebSocket.OPEN) return false;
      ws.send(JSON.stringify(msg));
      return true;
    },
  };
}
