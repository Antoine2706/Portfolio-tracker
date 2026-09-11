/* Segmented({ options: [{ value, label, disabled, tip }] | ["1M","3M"], value, onChange, size: "sm"|"md", ariaLabel })
   RangeSelector({ value, onChange }) — the 1M/3M/6M/YTD/1Y/ALL preset. */

import { html } from "/static/vendor/preact-htm.module.js";

export function Segmented({ options = [], value, onChange, size = "md", ariaLabel, class: cls = "" }) {
  const opts = options.map((o) => (typeof o === "object" && o !== null ? o : { value: o, label: String(o) }));
  const onKey = (e, i) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const dir = e.key === "ArrowRight" ? 1 : -1;
    let j = i;
    for (let k = 0; k < opts.length; k++) {
      j = (j + dir + opts.length) % opts.length;
      if (!opts[j].disabled) break;
    }
    onChange && onChange(opts[j].value);
    const btn = e.currentTarget.parentElement.children[j];
    if (btn) btn.focus();
  };
  return html`<div class=${["segmented", size === "sm" ? "sm" : "", cls].filter(Boolean).join(" ")} role="group" aria-label=${ariaLabel}>
    ${opts.map((o, i) => html`<button key=${o.value} type="button" class="segmented-item" aria-pressed=${String(o.value) === String(value) ? "true" : "false"}
      disabled=${o.disabled} data-tip=${o.tip} onClick=${() => onChange && onChange(o.value)} onKeyDown=${(e) => onKey(e, i)}>${o.label}</button>`)}
  </div>`;
}

export const RANGES = ["1M", "3M", "6M", "YTD", "1Y", "ALL"];

/** Number of calendar days back for a range key, or null for ALL/YTD (compute from dates). */
export function rangeDays(key) {
  switch (key) {
    case "1M": return 31;
    case "3M": return 92;
    case "6M": return 183;
    case "1Y": return 366;
    default: return null;
  }
}

/** Start index into a sorted ISO date array for a range key. */
export function rangeStartIndex(dates, key) {
  if (!dates || !dates.length) return 0;
  const last = new Date(dates[dates.length - 1]);
  let cutoff;
  if (key === "YTD") cutoff = new Date(last.getFullYear(), 0, 1);
  else {
    const d = rangeDays(key);
    if (d == null) return 0;
    cutoff = new Date(last.getTime() - d * 86400000);
  }
  let i = 0;
  while (i < dates.length - 1 && new Date(dates[i]) < cutoff) i++;
  return i;
}

export function RangeSelector({ value = "ALL", onChange, size = "sm", options = RANGES }) {
  return html`<${Segmented} options=${options} value=${value} onChange=${onChange} size=${size} ariaLabel="Range" />`;
}

export default Segmented;
