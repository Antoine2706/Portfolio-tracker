/* Holdings — the dense table. Props: { route, snapshot }.

   Layout order, and why:
   1. A sticky summary bar: value, count, priced/unpriced, prices-as-of with
      the delay, benchmark — the context that every number below assumes.
   2. One graded notice for what qualifies the numbers: fetch failures from
      meta, unpriced holdings, stale prices, per-row warnings. The same
      warnings also sit on the row (an amber dot with the list on hover), so a
      reader who skips the notice still cannot miss them.
   3. The table: instrument, trend, price with its provenance, day, quantity,
      cost, value, unrealised, weight (with a thin inline bar), risk share,
      divergence, volatility. Sorted by value by default; a totals footer;
      search, exports and density in the toolbar. A holding behaving as
      expected is visually silent — only gains/losses, stale prices and
      more-risk-than-capital carry colour.
   4. The watchlist (priced but not held) with the same shape and fewer columns.
   Row click -> #/holdings/{isin}, which app.js turns into the detail drawer.
   Nothing is computed here beyond filtering, sorting and formatting. */

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
import { Input } from "/static/components/Field.js";
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

/** Every note that qualifies a row, in the order the tooltip lists them. */
function rowNotes(h) {
  const notes = [];
  if (h.price == null) notes.push("No price: excluded from value, weight and the risk model.");
  if (h.price_is_stale) notes.push(`Price is stale — ${h.price_note}`);
  for (const w of h.warnings || []) notes.push(w);
  if (h.fx_note) notes.push(h.fx_note);
  return notes;
}

/** Per-row warnings, stale and unpriced -> alert items. */
function rowAlerts(holdings, unpriced) {
  const out = [];
  const seen = new Set();
  for (const h of holdings) {
    const name = h.short_name || h.name;
    if (h.price == null) { seen.add(h.isin); out.push({ code: `unpriced:${h.isin}`, severity: "SERIOUS", title: `${name}: no price`, detail: h.price_note || "Held, but no price could be found — excluded from value, weight and the risk model.", isins: [h.isin], route: `#/holdings/${h.isin}` }); }
    if (h.price_is_stale) out.push({ code: `stale:${h.isin}`, severity: "WARNING", title: `${name}: price is stale`, detail: h.price_note, isins: [h.isin], route: `#/holdings/${h.isin}` });
    for (const w of h.warnings || []) out.push({ code: `warn:${h.isin}:${w.slice(0, 24)}`, severity: "WARNING", title: name, detail: w, isins: [h.isin], route: `#/holdings/${h.isin}` });
    if (h.fx_note) out.push({ code: `fx:${h.isin}`, severity: "INFO", title: name, detail: h.fx_note, isins: [h.isin], route: `#/holdings/${h.isin}` });
  }
  for (const isin of unpriced || []) {
    if (seen.has(isin)) continue;
    out.push({ code: `unpriced:${isin}`, severity: "SERIOUS", title: `${isin}: no price`, detail: "Held, but no price could be found — excluded from value, weight and the risk model.", isins: [isin], route: `#/instruments/${isin}` });
  }
  return out;
}

function matches(h, q) {
  if (!q) return true;
  const hay = `${h.name || ""} ${h.isin || ""} ${h.symbol || ""}`.toLowerCase();
  return q.split(/\s+/).filter(Boolean).every((part) => hay.includes(part));
}

/* ---------------- cells ---------------- */

/** Instrument cell: name, symbol · ISIN caption, and an amber dot listing the row's warnings. */
function instrumentCell(row, v) {
  const notes = rowNotes(row);
  return html`<div class="hd-name-cell">
    <span class="hd-name-line">
      <span class="hd-name-text" title=${row.name}>${fmt.text(v)}</span>
      ${notes.length ? html`<span class="hd-warn" data-tip=${notes.join("\n")} tabindex="0" role="img" aria-label=${`Warnings: ${notes.join(" ")}`}></span>` : null}
    </span>
    <span class="cell-sub">${[row.symbol, row.isin].filter(Boolean).join(" · ")}</span>
  </div>`;
}

function sparkCell(row) {
  const values = row.sparkline || [];
  if (values.length < 2) return html`<span class="faint">${fmt.DASH}</span>`;
  return html`<span class="cell-spark" title=${`Last ${values.length} closes`}><${Sparkline} values=${values} area=${false} endDot=${false} /></span>`;
}

function priceCell(row, v) {
  if (v == null) return html`<span class="faint" title=${row.price_note}>${fmt.DASH}</span>`;
  return html`<span title=${row.price_note}>${fmt.money(v, row.price_currency)}</span>
    ${row.price_is_stale ? html`<span class="hd-price-stale"><${Badge} kind="warn" icon="clock" title=${row.price_note}>stale<//></span>` : null}`;
}

/** Two-line numeric cell: main figure (polarity-coloured) with a secondary figure beneath. */
function twoLine(main, sub, polarityOf) {
  return html`<span class="hd-cell-two"><span class=${fmt.polarityClass(polarityOf)}>${main}</span>${sub != null ? html`<span class="cell-sub">${sub}</span>` : null}</span>`;
}

function weightCell(row, v) {
  if (v == null) return fmt.DASH;
  const w = Math.round(Math.min(1, Math.max(0, v)) * 100);
  return html`<span class="hd-weight"><span>${fmt.pct(v)}</span><span class="hd-weight-bar" aria-hidden="true"><span style=${`width:${w}%`}></span></span></span>`;
}

const DIVERGENCE_TITLE = {
  "div-warm": "Carries more of the risk than of the capital",
  "div-cool": "Carries less of the risk than of the capital",
  "div-zero": "Risk and capital share are in line",
};

function divergenceCell(row, v) {
  if (v == null) return fmt.DASH;
  const cls = fmt.divergenceClass(v);
  return html`<span class=${cls} title=${DIVERGENCE_TITLE[cls]}>${fmt.pp(v)}</span>`;
}

/* ---------------- tables ---------------- */

function holdingColumns(base) {
  return [
    { key: "name", label: "Instrument", primary: true, render: instrumentCell, renderFooter: (r, v) => v, class: "hd-name" },
    { key: "sparkline", label: "30 d", sortable: false, render: sparkCell, width: 84, class: "center", title: "Last 30 closes in the quote currency" },
    { key: "price", label: "Price", numeric: true, render: priceCell, title: "Last price in its quote currency; hover for provenance" },
    { key: "day_change", label: "Day", numeric: true, render: (r, v) => twoLine(fmt.money(v, base, { signed: true }), fmt.pct(r.day_change_pct, { signed: true }), v),
      renderFooter: (r, v) => twoLine(fmt.money(v, base, { signed: true }), fmt.pct(r.day_change_pct, { signed: true }), v), title: "Change since the previous close" },
    { key: "quantity", label: "Qty", format: "qty", numeric: true },
    { key: "avg_cost", label: "Avg cost", format: "money", formatOptions: { currency: base }, numeric: true, title: `Average cost per unit, ${base}` },
    { key: "value", label: "Value", format: "money", formatOptions: { currency: base }, numeric: true },
    { key: "unrealised", label: "Unrealised", numeric: true, render: (r, v) => twoLine(fmt.money(v, base, { signed: true }), fmt.pct(r.unrealised_pct, { signed: true }), v),
      renderFooter: (r, v) => twoLine(fmt.money(v, base, { signed: true }), fmt.pct(r.unrealised_pct, { signed: true }), v), title: "Unrealised gain or loss and its share of cost" },
    { key: "weight", label: "Weight", numeric: true, render: weightCell, renderFooter: (r, v) => (v == null ? "" : fmt.pct(v)), title: "Share of portfolio value" },
    { key: "risk_share", label: "Risk share", format: "pct", numeric: true, title: "Share of portfolio volatility this holding contributes" },
    { key: "divergence", label: "Divergence", numeric: true, render: divergenceCell, title: "Risk share minus weight, in percentage points; warm = more risk than capital" },
    { key: "volatility", label: "Volatility", format: "pct", numeric: true, title: "Annualised standalone volatility over the risk window" },
  ];
}

function HoldingsTable({ holdings, allCount, totals, base, selected, onOpen, toolbar, toolbarRight }) {
  const columns = useMemo(() => holdingColumns(base), [base]);
  const complete = holdings.length === allCount;
  const footer = useMemo(() => (totals && complete ? {
    name: html`<span class="row" style="gap:6px"><span>${fmt.count(holdings.length, "holding")}</span>${totals.unpriced_holdings ? html`<span class="faint" style="font-weight:400">· ${fmt.int(totals.unpriced_holdings)} unpriced</span>` : null}</span>`,
    day_change: totals.day_change, day_change_pct: totals.day_change_pct,
    value: totals.value,
    unrealised: totals.unrealised, unrealised_pct: totals.unrealised_pct,
    weight: totals.priced_holdings ? 1 : null,
  } : null), [totals, holdings.length, complete]);
  return html`<${DataTable} columns=${columns} rows=${holdings} rowKey="isin" sort=${{ key: "value", dir: "desc" }}
    onRowClick=${(row) => onOpen(row.isin)} selectedKey=${selected} footer=${footer} densityToggle
    toolbar=${toolbar} toolbarRight=${toolbarRight} caption="Holdings" empty=${complete ? "No holdings" : "No holding matches the filter"} />`;
}

function watchColumns() {
  return [
    { key: "name", label: "Instrument", primary: true, render: instrumentCell, class: "hd-name" },
    { key: "sparkline", label: "30 d", sortable: false, render: sparkCell, width: 84, class: "center", title: "Last 30 closes in the quote currency" },
    { key: "price", label: "Price", numeric: true, render: (r, v) => (v == null ? fmt.DASH : fmt.money(v, r.price_currency)) },
    { key: "day_change_pct", label: "Day", numeric: true, render: (r, v) => html`<span class=${fmt.polarityClass(v)}>${fmt.pct(v, { signed: true })}</span>` },
    { key: "volatility", label: "Volatility", format: "pct", numeric: true, title: "Annualised standalone volatility" },
    { key: "in_risk_model", label: "Risk model", render: (r, v) => (v
      ? html`<${Badge} kind="good" icon="check" title="Enough overlapping history to simulate with">in model<//>`
      : html`<${Badge} kind="neutral" outline title="Not enough history to sit in the covariance matrix">no history<//>`) },
  ];
}

function WatchlistTable({ items, selected, onOpen }) {
  const columns = useMemo(() => watchColumns(), []);
  return html`<${DataTable} columns=${columns} rows=${items} rowKey="isin" sort=${{ key: "name", dir: "asc" }} onRowClick=${(row) => onOpen(row.isin)} selectedKey=${selected} caption="Watchlist" empty="Nothing on the watchlist" />`;
}

/* ---------------- summary bar ---------------- */

function SummaryBar({ snapshot, base, holdings, watchlist, delay }) {
  const t = snapshot.totals || {};
  const m = snapshot.meta;
  const bench = (snapshot.benchmarks || []).find((b) => b.symbol === snapshot.selected_benchmark);
  const benchName = bench ? (bench.index || bench.label || bench.symbol) : snapshot.selected_benchmark;
  return html`<div class="hd-bar" role="region" aria-label="Holdings summary">
    <span class="hd-bar-item"><span class="hd-bar-value">${fmt.money(t.value, base)}</span><span>value</span></span>
    <span class="hd-bar-sep"></span>
    <span class="hd-bar-item"><b>${fmt.int(holdings.length)}</b>${holdings.length === 1 ? "holding" : "holdings"}</span>
    <span class="hd-bar-item"><b>${fmt.int(t.priced_holdings)}</b>priced${t.unpriced_holdings ? html`<span>, </span><b class="warn">${fmt.int(t.unpriced_holdings)}</b><span class="warn">unpriced</span>` : null}</span>
    <span class="hd-bar-sep"></span>
    <span class="hd-bar-item truncate" title=${`Snapshot generated ${fmt.dateTime(m.generated_at)} · provider ${m.provider}`}>${fmt.delayNote(m.prices_as_of, delay)}</span>
    <span class="hd-bar-sep"></span>
    <span class="hd-bar-item truncate" title="Benchmark used for beta and the comparison lines">vs ${benchName}</span>
    <span class="spacer"></span>
    <span class="hd-bar-item faint">${fmt.count(watchlist.length, "instrument")} on the watchlist</span>
  </div>`;
}

/* ---------------- page ---------------- */

export default function HoldingsPage({ route, snapshot }) {
  const [busy, setBusy] = useState(null);
  const [query, setQuery] = useState("");
  const selected = route && route.params ? route.params.isin : null;
  const base = (snapshot && snapshot.meta && snapshot.meta.base_currency) || "EUR";
  const holdings = (snapshot && snapshot.holdings) || [];
  const watchlist = (snapshot && snapshot.watchlist) || [];
  const unpriced = (snapshot && snapshot.unpriced) || [];
  const totals = snapshot && snapshot.totals;

  const alerts = useMemo(() => [...failureAlerts(snapshot && snapshot.meta), ...rowAlerts(holdings, unpriced)], [snapshot]);
  const delay = useMemo(() => { const d = holdings.map((h) => h.price_delay_minutes).filter((v) => v != null); return d.length ? Math.max(...d) : null; }, [holdings]);
  const q = query.trim().toLowerCase();
  const filtered = useMemo(() => holdings.filter((h) => matches(h, q)), [holdings, q]);
  const filteredWatch = useMemo(() => watchlist.filter((w) => matches(w, q)), [watchlist, q]);

  const open = (isin) => navigate(`/holdings/${encodeURIComponent(isin)}`);
  const doExport = async (kind) => {
    if (busy) return;
    setBusy(kind);
    try { await (kind === "csv" ? exportHoldingsCsv() : exportWorkbook()); }
    catch (err) { toast.error(errorMessage(err), { title: "Export failed" }); }
    finally { setBusy(null); }
  };

  const subtitle = snapshot
    ? `Every position with its price provenance, weight and share of risk. Click a row for the detail; ← → move between holdings once it is open.`
    : "Every position with price provenance, weight and risk share.";

  if (!snapshot) {
    return html`<${Page} title="Holdings" subtitle=${subtitle}>
      <${Card}><${EmptyState} icon="clock" title="Loading the snapshot…" body="Holdings, prices and the risk shares arrive in one request." /><//>
    <//>`;
  }

  const toolbar = html`
    <div class="hd-search">
      <${Icon} name="search" size=${14} />
      <${Input} value=${query} onInput=${(e) => setQuery(e.target.value)} placeholder="Filter by name, symbol or ISIN" aria-label="Filter holdings" spellcheck="false" autocomplete="off" />
      ${query ? html`<button type="button" class="hd-search-clear" title="Clear filter" aria-label="Clear filter" onClick=${() => setQuery("")}><${Icon} name="x" size=${12} /></button>` : null}
    </div>
    <span class="hd-count">${q ? `${filtered.length} of ${holdings.length}` : fmt.count(holdings.length, "position")}</span>`;
  const toolbarRight = html`
    <${Button} size="sm" variant="ghost" icon="download" loading=${busy === "csv"} onClick=${() => doExport("csv")} title="Download holdings.csv">Export CSV<//>
    <${Button} size="sm" variant="ghost" icon="download" loading=${busy === "xlsx"} onClick=${() => doExport("xlsx")} title="Download the full workbook (holdings, transactions, risk)">Export XLSX<//>`;

  return html`<${Page} title="Holdings" subtitle=${subtitle} class="hd-page" wide
    actions=${html`<${Button} variant="secondary" icon="plus" onClick=${() => navigate("/transactions?new=1")}>Record transaction<//>`}>
    <div class="stack gap-4">
      <${SummaryBar} snapshot=${snapshot} base=${base} holdings=${holdings} watchlist=${watchlist} delay=${delay} />
      <${Notice} alerts=${alerts} visible=${3} onNavigate=${navigate} />
      <${Card} flush class="hd-tablebox">
        ${holdings.length
          ? html`<${HoldingsTable} holdings=${filtered} allCount=${holdings.length} totals=${totals} base=${base} selected=${selected} onOpen=${open} toolbar=${toolbar} toolbarRight=${toolbarRight} />`
          : html`<${EmptyState} icon="holdings" title="No holdings" body="Record a BUY on the Transactions page and the position appears here with its price, weight and risk share."
              action=${html`<${Button} variant="primary" icon="plus" onClick=${() => navigate("/transactions?new=1")}>Record a transaction<//>`} />`}
      <//>

      <${Card} title="Watchlist" caption="Priced and in the universe, but not held. Open one to see its history and correlation with the portfolio, or send it to the simulator." flush class="hd-tablebox"
        actions=${html`<${Button} size="sm" variant="secondary" icon="plus" onClick=${() => navigate("/instruments")}>Add instrument<//>`}>
        ${watchlist.length
          ? html`<${WatchlistTable} items=${filteredWatch} selected=${selected} onOpen=${open} />`
          : html`<${EmptyState} compact icon="eye" title="Nothing on the watchlist" body="Instruments in the universe that you do not hold show up here with a price and a trend." />`}
      <//>
    </div>
  <//>`;
}
