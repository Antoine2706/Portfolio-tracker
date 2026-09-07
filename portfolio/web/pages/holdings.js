/* Holdings — the dense table. Props: { route, snapshot }.

   Layout order, and why:
   1. Notices first: price-fetch failures from meta, then the per-row warnings
      and stale prices as ONE graded notice. The same warnings also sit next to
      the number they qualify (an amber marker on the row), so a reader who
      skips the notice still cannot miss them.
   2. The holdings table: instrument, trend, quantity, cost, price with its
      provenance, day, value, P&L, weight (with a thin inline bar), risk share,
      divergence, volatility. Sorted by value by default; a totals footer.
      A holding behaving as expected is visually silent — only gains/losses,
      stale prices and more-risk-than-capital carry colour.
   3. The watchlist (priced but not held) and, if any, the unpriced list.
   Row click -> #/holdings/{isin}, which app.js turns into the detail drawer. */

import { html, useMemo, useState } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { DataTable } from "/static/components/DataTable.js";
import { Badge } from "/static/components/Badge.js";
import { Notice } from "/static/components/Notice.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Sparkline } from "/static/components/Sparkline.js";
import { Button } from "/static/components/Button.js";
import { Icon } from "/static/components/Icons.js";
import { toast } from "/static/components/Toast.js";
import { navigate } from "/static/lib/router.js";
import { exportHoldingsCsv, exportWorkbook, errorMessage } from "/static/lib/api.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- page-local helpers ---------------- */

/** meta.failures -> alert items for the graded Notice. */
function failureAlerts(meta) {
  const failures = (meta && meta.failures) || [];
  return failures.map((f) => ({ code: `fetch:${f.key}`, severity: "SERIOUS", title: `${f.name || f.key}: price fetch failed`, detail: f.message, isins: [], route: "#/instruments" }));
}

/** Per-row warnings and stale prices -> alert items. */
function rowAlerts(holdings) {
  const out = [];
  for (const h of holdings) {
    if (h.price_is_stale) out.push({ code: `stale:${h.isin}`, severity: "WARNING", title: `${fmt.shortName(h.name, 40)}: price is stale`, detail: h.price_note, isins: [h.isin], route: `#/holdings/${h.isin}` });
    if (h.price == null) out.push({ code: `unpriced:${h.isin}`, severity: "SERIOUS", title: `${fmt.shortName(h.name, 40)}: no price`, detail: h.price_note || "Excluded from value, weight and risk until a price arrives.", isins: [h.isin], route: `#/holdings/${h.isin}` });
    for (const w of h.warnings || []) out.push({ code: `warn:${h.isin}:${w.slice(0, 24)}`, severity: "WARNING", title: fmt.shortName(h.name, 40), detail: w, isins: [h.isin], route: `#/holdings/${h.isin}` });
    if (h.fx_note) out.push({ code: `fx:${h.isin}`, severity: "INFO", title: fmt.shortName(h.name, 40), detail: h.fx_note, isins: [h.isin], route: `#/holdings/${h.isin}` });
  }
  return out;
}

/** Two-line numeric cell: main figure (polarity-coloured) with a secondary figure beneath. */
function twoLine(main, sub, polarityOf) {
  const cls = polarityOf == null ? "" : fmt.polarityClass(polarityOf);
  return html`<span class=${cls}>${main}</span>${sub != null ? html`<span class="cell-sub">${sub}</span>` : null}`;
}

/** Instrument cell: name, symbol · ISIN caption, and an amber marker when the row has warnings. */
function instrumentCell(row) {
  const notes = [];
  if (row.price_is_stale) notes.push("Price is stale: " + row.price_note);
  if (row.price == null) notes.push("No price: excluded from value and risk.");
  for (const w of row.warnings || []) notes.push(w);
  if (row.fx_note) notes.push(row.fx_note);
  return html`<div class="row" style="gap:6px">
    <div class="truncate" style="max-width:260px" title=${row.name}>
      <span>${fmt.text(row.name)}</span>
      <span class="cell-sub">${[row.symbol, row.isin].filter(Boolean).join(" · ")}</span>
    </div>
    ${notes.length ? html`<span class="warn-mark" title=${notes.join("\n")} aria-label=${notes.join(" ")} tabindex="0"><${Icon} name="alertTriangle" size=${13} stroke=${2} /></span>` : null}
  </div>`;
}

const BAR_MAX_PX = 44;
function weightCell(row, v) {
  if (v == null) return fmt.DASH;
  const w = Math.max(2, Math.round(Math.min(1, Math.max(0, v)) * BAR_MAX_PX));
  return html`<span class="row" style="justify-content:flex-end;gap:8px">
    <span>${fmt.pct(v)}</span>
    <span class="cell-bar" style=${`width:${w}px;margin-left:0;opacity:.8`} aria-hidden="true"></span>
  </span>`;
}

function divergenceCell(row, v) {
  if (v == null) return fmt.DASH;
  // More risk than capital is the direction worth a colour; less is benign and stays quiet.
  const cls = v > 0.0005 ? "neg" : "muted";
  return html`<span class=${cls} title=${v > 0 ? "Carries more of the risk than of the capital" : "Carries less of the risk than of the capital"}>${fmt.pp(v)}</span>`;
}

function priceCell(row, v) {
  if (v == null) return html`<span class="faint" title=${row.price_note}>${fmt.DASH}</span>`;
  return html`<span class="row" style="justify-content:flex-end;gap:6px">
    <span title=${row.price_note}>${fmt.money(v, row.price_currency)}</span>
    ${row.price_is_stale ? html`<${Badge} kind="warn" icon="clock" title=${row.price_note}>stale<//>` : null}
  </span>`;
}

function sparkCell(row) {
  const values = row.sparkline || [];
  if (values.length < 2) return html`<span class="faint">${fmt.DASH}</span>`;
  const up = values[values.length - 1] >= values[0];
  return html`<span class="cell-spark" title=${`Last ${values.length} closes`}><${Sparkline} values=${values} area=${false} endDot=${false} color=${up ? "var(--color-text-3)" : "var(--color-text-3)"} /></span>`;
}

/* ---------------- tables ---------------- */

function holdingColumns(base) {
  return [
    { key: "name", label: "Instrument", primary: true, render: instrumentCell, renderFooter: (r, v) => v, sortable: true },
    { key: "sparkline", label: "30 d", sortable: false, render: sparkCell, width: 84, class: "center" },
    { key: "quantity", label: "Qty", format: "qty", numeric: true },
    { key: "avg_cost", label: "Avg cost", format: "money", formatOptions: { currency: base }, numeric: true, title: `Average cost per unit, ${base}` },
    { key: "price", label: "Price", numeric: true, render: priceCell, title: "Last price in its quote currency; hover for provenance" },
    { key: "day_change", label: "Day", numeric: true, render: (r, v) => twoLine(fmt.money(v, base, { signed: true }), fmt.pct(r.day_change_pct, { signed: true }), v),
      renderFooter: (r, v) => twoLine(fmt.money(v, base, { signed: true }), fmt.pct(r.day_change_pct, { signed: true }), v) },
    { key: "value", label: "Value", format: "money", formatOptions: { currency: base }, numeric: true },
    { key: "unrealised", label: "P&L", numeric: true, render: (r, v) => twoLine(fmt.money(v, base, { signed: true }), fmt.pct(r.unrealised_pct, { signed: true }), v),
      renderFooter: (r, v) => twoLine(fmt.money(v, base, { signed: true }), fmt.pct(r.unrealised_pct, { signed: true }), v), title: "Unrealised gain or loss and its share of cost" },
    { key: "weight", label: "Weight", numeric: true, render: weightCell, renderFooter: (r, v) => fmt.pct(v), title: "Share of portfolio value" },
    { key: "risk_share", label: "Risk share", format: "pct", numeric: true, title: "Share of portfolio volatility this holding contributes" },
    { key: "divergence", label: "Divergence", numeric: true, render: divergenceCell, title: "Risk share minus weight, in percentage points" },
    { key: "volatility", label: "Volatility", format: "pct", numeric: true, title: "Annualised standalone volatility over the risk window" },
  ];
}

function HoldingsTable({ holdings, totals, base, selected, onOpen }) {
  const columns = useMemo(() => holdingColumns(base), [base]);
  const footer = useMemo(() => (totals ? {
    name: html`<span class="row" style="gap:6px"><span>${holdings.length} holdings</span>${totals.unpriced_holdings ? html`<span class="faint" style="font-weight:400">· ${totals.unpriced_holdings} unpriced</span>` : null}</span>`,
    day_change: totals.day_change, day_change_pct: totals.day_change_pct,
    value: totals.value,
    unrealised: totals.unrealised, unrealised_pct: totals.unrealised_pct,
    weight: holdings.length ? 1 : null,
  } : null), [totals, holdings.length]);
  return html`<${DataTable} columns=${columns} rows=${holdings} rowKey="isin" sort=${{ key: "value", dir: "desc" }}
    onRowClick=${(row) => onOpen(row.isin)} selectedKey=${selected} footer=${footer} densityToggle
    toolbar=${html`<span class="faint small">${holdings.length} priced ${holdings.length === 1 ? "position" : "positions"} · click a row for detail</span>`}
    caption="Holdings" empty="No holdings" />`;
}

function watchColumns() {
  return [
    { key: "name", label: "Instrument", primary: true, render: (r, v) => html`<div class="truncate" style="max-width:280px" title=${v}><span>${fmt.text(v)}</span><span class="cell-sub">${[r.symbol, r.isin].filter(Boolean).join(" · ")}</span></div>` },
    { key: "sparkline", label: "30 d", sortable: false, render: sparkCell, width: 84, class: "center" },
    { key: "price", label: "Price", numeric: true, render: (r, v) => (v == null ? fmt.DASH : fmt.money(v, r.price_currency)) },
    { key: "day_change_pct", label: "Day", numeric: true, render: (r, v) => html`<span class=${fmt.polarityClass(v)}>${fmt.pct(v, { signed: true })}</span>` },
    { key: "volatility", label: "Volatility", format: "pct", numeric: true, title: "Annualised standalone volatility" },
    { key: "in_risk_model", label: "Risk model", sortable: true, render: (r, v) => (v
      ? html`<${Badge} kind="good" icon="check" title="Enough overlapping history to simulate with">in model<//>`
      : html`<${Badge} kind="neutral" outline title="Not enough history to sit in the covariance matrix">no history<//>`) },
  ];
}

function WatchlistTable({ items, selected, onOpen }) {
  const columns = useMemo(() => watchColumns(), []);
  return html`<${DataTable} columns=${columns} rows=${items} rowKey="isin" sort=${{ key: "name", dir: "asc" }} onRowClick=${(row) => onOpen(row.isin)} selectedKey=${selected} caption="Watchlist" empty="Nothing on the watchlist" />`;
}

/* ---------------- page ---------------- */

export default function HoldingsPage({ route, snapshot }) {
  const [busy, setBusy] = useState(null);
  const selected = route && route.params ? route.params.isin : null;
  const base = (snapshot && snapshot.meta && snapshot.meta.base_currency) || "EUR";
  const holdings = (snapshot && snapshot.holdings) || [];
  const watchlist = (snapshot && snapshot.watchlist) || [];
  const unpriced = (snapshot && snapshot.unpriced) || [];
  const totals = snapshot && snapshot.totals;

  const alerts = useMemo(() => [...failureAlerts(snapshot && snapshot.meta), ...rowAlerts(holdings)], [snapshot]);
  const unpricedAlerts = useMemo(() => unpriced.map((isin) => ({ code: `unpriced:${isin}`, severity: "SERIOUS", title: isin, detail: "Held, but no price could be found — excluded from value, weight and the risk model.", isins: [isin], route: `#/instruments/${isin}` })), [unpriced]);

  const open = (isin) => navigate(`/holdings/${encodeURIComponent(isin)}`);
  const doExport = async (kind) => {
    if (busy) return;
    setBusy(kind);
    try { await (kind === "csv" ? exportHoldingsCsv() : exportWorkbook()); }
    catch (err) { toast.error(errorMessage(err), { title: "Export failed" }); }
    finally { setBusy(null); }
  };

  const subtitle = snapshot
    ? `${holdings.length} ${holdings.length === 1 ? "holding" : "holdings"} worth ${fmt.money(totals && totals.value, base)} · ${watchlist.length} on the watchlist · ${fmt.delayNote(snapshot.meta.prices_as_of, 15)}`
    : "Every position with price provenance, weight and risk share.";

  const actions = html`
    <${Button} variant="secondary" size="sm" icon="download" loading=${busy === "csv"} onClick=${() => doExport("csv")} title="Download holdings.csv">CSV<//>
    <${Button} variant="secondary" size="sm" icon="download" loading=${busy === "xlsx"} onClick=${() => doExport("xlsx")} title="Download the full workbook (holdings, transactions, risk)">Workbook<//>
  `;

  if (!snapshot) {
    return html`<${Page} title="Holdings" subtitle=${subtitle} actions=${actions}>
      <${Card}><div class="faint small" style="padding:24px;text-align:center">Loading the snapshot…</div><//>
    <//>`;
  }

  return html`<${Page} title="Holdings" subtitle=${subtitle} actions=${actions} wide>
    <div class="stack gap-4">
      <${Notice} alerts=${alerts} visible=${3} onNavigate=${navigate} />
      <${Card} flush>
        ${holdings.length
          ? html`<${HoldingsTable} holdings=${holdings} totals=${totals} base=${base} selected=${selected} onOpen=${open} />`
          : html`<${EmptyState} icon="holdings" title="No holdings" body="Record a BUY on the Transactions page and the position appears here with its price, weight and risk share."
              action=${html`<${Button} variant="primary" icon="plus" onClick=${() => navigate("/transactions?new=1")}>Record a transaction<//>`} />`}
      <//>

      <${Card} title="Watchlist" caption="Priced and in the universe, but not held. Open one to see its history and correlation with the portfolio, or send it to the simulator." flush>
        ${watchlist.length
          ? html`<${WatchlistTable} items=${watchlist} selected=${selected} onOpen=${open} />`
          : html`<${EmptyState} compact icon="eye" title="Nothing on the watchlist" body="Instruments in the universe that you do not hold show up here with a price and a trend." />`}
      <//>

      ${unpriced.length ? html`<${Notice} alerts=${unpricedAlerts} summary=${`${unpriced.length} ${unpriced.length === 1 ? "holding has" : "holdings have"} no price`} visible=${3} onNavigate=${navigate} />` : null}
    </div>
  <//>`;
}
