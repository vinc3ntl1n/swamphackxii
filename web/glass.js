// Liquid glass behind existing elements, using liquid-glass-react as small React islands.
// Each host keeps its own content; the glass is a layer underneath it, sized to the host.
import { createElement as h, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import LiquidGlass from "liquid-glass-react";

function Glass({ host, radius, options }) {
  const hostRef = useRef(host);
  const [size, setSize] = useState({ w: host.clientWidth, h: host.clientHeight });
  useEffect(() => {
    const observer = new ResizeObserver(() => setSize({ w: host.clientWidth, h: host.clientHeight }));
    observer.observe(host);
    return () => observer.disconnect();
  }, [host]);
  const fill = {
    width: `${size.w}px`,
    height: `${size.h}px`,
    borderRadius: `${radius}px`,
    background: "var(--glass-tint, transparent)",
  };
  return h(
    LiquidGlass,
    {
      cornerRadius: radius,
      padding: "0px",
      mouseContainer: hostRef,
      style: { position: "absolute", top: "50%", left: "50%" },
      ...options,
    },
    h("div", { style: fill }),
  );
}

export function mountGlass(host, { radius, ...options }) {
  const layer = document.createElement("div");
  layer.className = "lg-layer";
  layer.setAttribute("aria-hidden", "true");
  host.prepend(layer);
  createRoot(layer).render(h(Glass, { host, radius, options }));
  host.classList.add("lg-on");
}
