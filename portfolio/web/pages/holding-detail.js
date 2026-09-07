/* Holding detail — the drawer body for #/holdings/:isin. Placeholder: shows
   what the snapshot already knows about the holding; the real version fetches
   GET /api/holdings/{isin} (lib/api.js getHoldingDetail) for prices, markers,
   stats and transactions. Props: { isin, snapshot }. */

import { html } from "/static/vendor/preact-htm.module.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Badge } from "/static/components/Badge.js";
import { Sparkline } from "/static/components/Sparkline.js";
import { useStore } from "/static/lib/store.js";
import * as fmt from "/static/lib/format.js";

export default function HoldingDetailPanel({ isin }) {
  const snapshot = useStore((s) => s.snapshot);
  const h = snapshot && ((snapshot.holdings || []).find((x) => x.isin === isin) || (snapshot.watchlist || []).find((x) => x.isin === isin));
  if (!h) {
    return html`<${EmptyState} icon="search" title="Not in the snapshot" body=${`No holding or watchlist item with ISIN ${isin}.`} />`;
  }
  const isHolding = "quantity" in h;
  return html`<div class="stack gap-4">
    <div class="row between">
      <div class="row">
        ${h.price_is_stale ? html`<${Badge} kind="warn" icon="clock">stale<//>` : null}
        ${!isHolding ? html`<${Badge} kind="neutral" icon="eye">watchlist<//>` : null}
      </div>
      <div style="width:120px;height:32px"><${Sparkline} values=${h.sparkline || []} /></div>
    </div>
    <dl class="definition-list">
      <dt>Price</dt><dd class="num">${fmt.money(h.price, h.price_currency)} <span class="faint small">${h.price_as_of ? fmt.dateTime(h.price_as_of) : ""}</span></dd>
      ${isHolding ? html`
        <dt>Quantity</dt><dd class="num">${fmt.qty(h.quantity)}</dd>
        <dt>Value</dt><dd class="num">${fmt.money(h.value)}</dd>
        <dt>Unrealised</dt><dd class=${"num " + fmt.polarityClass(h.unrealised)}>${fmt.money(h.unrealised, "EUR", { signed: true })} (${fmt.pct(h.unrealised_pct, { signed: true })})</dd>
        <dt>Weight</dt><dd class="num">${fmt.pct(h.weight)}</dd>
        <dt>Risk share</dt><dd class="num">${fmt.pct(h.risk_share)}</dd>
      ` : null}
      <dt>Volatility</dt><dd class="num">${fmt.pct(h.volatility)}</dd>
    </dl>
    ${isHolding && h.price_note ? html`<p class="small muted">${h.price_note}</p>` : null}
    <${EmptyState} compact icon="hammer" title="Detail being built" body="Price history with trade markers, drawdown, statistics, transactions and correlations will render here from GET /api/holdings/{isin}." />
  </div>`;
}
