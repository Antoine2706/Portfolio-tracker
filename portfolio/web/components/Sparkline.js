/* Sparkline({ values, color, area, width, height, baseline, endDot, class })
   A tiny ECharts line with no axes, no tooltip. Sized by its container unless width/height given. */

import { html, useRef } from "/static/vendor/preact-htm.module.js";
import { useChart, sparklineOption } from "/static/lib/charts.js";

export function Sparkline({ values = [], color, area = true, width, height, baseline, endDot = true, class: cls = "", style }) {
  const ref = useRef(null);
  const key = values.length ? `${values.length}:${values[0]}:${values[values.length - 1]}` : "";
  useChart(ref, () => (values && values.length > 1 ? sparklineOption(values, { color, area, baseline, endDot }) : null), [key, color, area, baseline, endDot]);
  const sz = [width ? `width:${width}px` : "", height ? `height:${height}px` : "", style || ""].filter(Boolean).join(";");
  return html`<div ref=${ref} class=${["sparkline", cls].filter(Boolean).join(" ")} style=${sz} aria-hidden="true"></div>`;
}

export default Sparkline;
