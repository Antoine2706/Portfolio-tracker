/* KpiTile({ label, value, delta, deltaLabel, deltaFormat, polarity, sub, sparkline,
             sparkColor, size: "sm"|"md"|"hero", help, valueClass, loading })
   - value: an already formatted string (proportional figures, never tabular).
   - delta: a number (fraction, formatted with deltaFormat = "pct"|"money"|"pp"|fn) or a string.
   - polarity: "pos"|"neg"|"" colours the value itself (use for P&L); deltas are
     coloured by their own sign. upIsGood=false flips the delta colour (e.g. volatility). */

import { html } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";
import { Sparkline } from "/static/components/Sparkline.js";
import { Help } from "/static/components/Tooltip.js";
import { useStore } from "/static/lib/store.js";
import * as fmt from "/static/lib/format.js";

function formatDelta(delta, kind, currency) {
  if (delta == null) return null;
  if (typeof delta === "string") return delta;
  if (typeof kind === "function") return kind(delta);
  switch (kind) {
    case "money": return fmt.money(delta, currency, { signed: true });
    case "money-compact": return fmt.money(delta, currency, { signed: true, compact: true });
    case "pp": return fmt.pp(delta, { signed: true });
    case "num": return fmt.num(delta, { signed: true });
    case "pct":
    default: return fmt.pct(delta, { signed: true });
  }
}

export function KpiTile({ label, value, delta, deltaLabel, deltaFormat = "pct", currency = "EUR", polarity = "", upIsGood = true,
                          sub, sparkline, sparkColor, size = "md", help, valueClass = "", loading, class: cls = "", onClick }) {
  const storeLoading = useStore((s) => s.loading);
  const isLoading = loading ?? storeLoading;
  const deltaNum = typeof delta === "number" ? delta : null;
  let deltaClass = "";
  if (deltaNum != null && deltaNum !== 0) {
    const up = deltaNum > 0;
    deltaClass = (up === upIsGood) ? "pos" : "neg";
  }
  const deltaText = formatDelta(delta, deltaFormat, currency);
  const arrowName = deltaNum == null || deltaNum === 0 ? null : deltaNum > 0 ? "arrowUpRight" : "arrowDownRight";
  return html`<div class=${["card", "kpi", isLoading ? "is-loading" : "", onClick ? "interactive" : "", cls].filter(Boolean).join(" ")} onClick=${onClick}>
    <div class="kpi-label">${label}${help ? html`<${Help} text=${help} />` : null}</div>
    <div class="kpi-row">
      <div class="grow">
        <div class=${["kpi-value", size === "sm" ? "sm" : size === "hero" ? "hero" : "", polarity, valueClass].filter(Boolean).join(" ")}>${value == null || value === "" ? fmt.DASH : value}</div>
        ${deltaText != null || sub ? html`<div class="kpi-meta">
          ${deltaText != null ? html`<span class=${["kpi-delta", deltaClass].filter(Boolean).join(" ")}>
            ${arrowName ? html`<${Icon} name=${arrowName} size=${12} stroke=${2} />` : null}${deltaText}
          </span>` : null}
          ${deltaLabel ? html`<span class="kpi-sub">${deltaLabel}</span>` : null}
          ${sub && !deltaLabel ? html`<span class="kpi-sub">${sub}</span>` : null}
        </div>` : null}
        ${sub && deltaLabel ? html`<div class="kpi-meta"><span class="kpi-sub">${sub}</span></div>` : null}
      </div>
      ${sparkline && sparkline.length > 1 ? html`<div class="kpi-spark"><${Sparkline} values=${sparkline} color=${sparkColor} /></div>` : null}
    </div>
  </div>`;
}

export default KpiTile;
