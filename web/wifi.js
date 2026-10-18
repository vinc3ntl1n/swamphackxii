// <pulse-wifi>: a small static fan of signal bands, drawn in currentColor so it takes the color of its context.
//   dir     where the bands point, degrees clockwise from straight up (default 45, up and to the right)
//   spread  how wide the fan opens, in degrees (default 80)
//   bands   number of arcs (default 3)
//   state   "live" (default) or "off"; off greys out and fades the outer bands
//   pixel   draw on a square grid instead of smooth arcs; the value is the cell size (try 1 to 2)
// Size it with font-size or width/height; it is 1em square by default.

const NS = "http://www.w3.org/2000/svg";
const DOT = 1.6;
const GAP = 4.5;
const STROKE = 1.8;

function point(r, deg) {
  const a = (deg * Math.PI) / 180;
  return [r * Math.sin(a), -r * Math.cos(a)];
}

class PulseWifi extends HTMLElement {
  static observedAttributes = ["dir", "spread", "bands", "pixel"];

  connectedCallback() {
    if (!this.hasAttribute("aria-hidden") && !this.hasAttribute("role")) this.setAttribute("aria-hidden", "true");
    this.draw();
  }

  attributeChangedCallback() {
    if (this.isConnected) this.draw();
  }

  num(name, fallback) {
    const v = parseFloat(this.getAttribute(name));
    return Number.isFinite(v) ? v : fallback;
  }

  draw() {
    const dir = this.num("dir", 45);
    const spread = Math.min(Math.max(this.num("spread", 80), 10), 180);
    const bands = Math.max(1, Math.round(this.num("bands", 3)));
    const from = dir - spread / 2;
    const to = dir + spread / 2;

    const svg = document.createElementNS(NS, "svg");
    const box = [-DOT, -DOT, DOT, DOT];
    const grow = ([x, y], pad) => {
      box[0] = Math.min(box[0], x - pad);
      box[1] = Math.min(box[1], y - pad);
      box[2] = Math.max(box[2], x + pad);
      box[3] = Math.max(box[3], y + pad);
    };

    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("r", DOT);
    dot.setAttribute("class", "wifi-dot");
    svg.append(dot);

    for (let i = 1; i <= bands; i++) {
      const r = i * GAP;
      const a = point(r, from);
      const b = point(r, to);
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", `M${a[0].toFixed(2)} ${a[1].toFixed(2)} A${r} ${r} 0 0 1 ${b[0].toFixed(2)} ${b[1].toFixed(2)}`);
      path.setAttribute("class", "wifi-band");
      path.style.setProperty("--i", i);
      path.style.setProperty("--n", bands);
      svg.append(path);
      // The arc's extent is its two ends plus any compass point it sweeps past.
      grow(a, 1);
      grow(b, 1);
      for (let k = Math.ceil(from / 90) * 90; k <= to; k += 90) grow(point(r, k), 1);
    }

    const cell = this.num("pixel", 0);
    if (cell > 0) this.pixelate(svg, box, cell, dir, spread, bands);
    svg.setAttribute("viewBox", `${box[0]} ${box[1]} ${box[2] - box[0]} ${box[3] - box[1]}`);
    this.replaceChildren(svg);
  }

  // Swap the smooth shapes for filled grid cells: a cell is on when its center lands on the dot or a band.
  pixelate(svg, box, cell, dir, spread, bands) {
    const half = Math.max(STROKE / 2, cell * 0.55);
    const runs = Array.from({ length: bands + 1 }, () => "");
    for (let y = Math.floor(box[1] / cell) * cell; y < box[3]; y += cell) {
      for (let x = Math.floor(box[0] / cell) * cell; x < box[2]; x += cell) {
        const cx = x + cell / 2;
        const cy = y + cell / 2;
        const d = Math.hypot(cx, cy);
        let i = -1;
        if (d <= DOT + cell * 0.35) i = 0;
        else {
          const off = ((((Math.atan2(cx, -cy) * 180) / Math.PI - dir) % 360) + 540) % 360 - 180;
          const band = Math.round(d / GAP);
          if (Math.abs(off) <= spread / 2 && band >= 1 && band <= bands && Math.abs(d - band * GAP) <= half) i = band;
        }
        if (i >= 0) runs[i] += `M${+x.toFixed(3)} ${+y.toFixed(3)}h${cell}v${cell}h${-cell}z`;
      }
    }
    const parts = svg.children;
    for (let i = 0; i <= bands; i++) {
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", runs[i]);
      path.setAttribute("class", parts[i].getAttribute("class"));
      path.setAttribute("style", parts[i].getAttribute("style") || "");
      path.setAttribute("shape-rendering", "crispEdges");
      parts[i].replaceWith(path);
    }
  }
}

if (!customElements.get("pulse-wifi")) customElements.define("pulse-wifi", PulseWifi);
