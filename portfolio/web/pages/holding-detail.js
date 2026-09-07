/* Holding detail — the drawer body for #/holdings/:isin. Props: { isin }.

   The drawer title/subtitle (name, symbol · ISIN) are set by app.js from the
   snapshot, so this panel starts with what the snapshot already knows —
   provenance, price, day change, the figures from the holdings table — and
   fills in the rest when GET /api/holdings/{isin} arrives. No skeletons: the
   header is real from the first paint and the sections below appear once.

   Order, top to bottom: identity + price provenance + actions; the figures
   the holdings table shows (so the reader never has to go back); the price
   chart with the trades marked on it and its drawdown; statistics over the
   risk window; correlation against every other holding; the ledger for this
   instrument. */

import { html, useEffect, useMemo, useState } from "/static/vendor/preact-htm.module.js";
import { Card } from "/static/components/Card.js";
import { Chart } from "/static/components/Chart.js";
import { DataTable } from "/static/components/DataTable.js";
import { Badge, TxTypeBadge } from "/static/components/Badge.js";
import { Banner } from "/static/components/Banner.js";
import { Button } from "/static/components/Button.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { RangeSelector, rangeStartIndex } from "/static/components/Segmented.js";
import { useStore } from "/static/lib/store.js";
import { navigate } from "/static/lib/router.js";
import { getHoldingDetail, errorMessage } from "/static/lib/api.js";
import { lineOption, drawdownOption, seriesTable, tipElement } from "/static/lib/charts.js";
import { tokens } from "/static/lib/theme.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- page-local helpers ---------------- */

function polarity(v) { return fmt.polarityClass(v); }

/** A small labelled figure for the stat grid. */
function Stat({ label, value, sub, cls = "", title }) {
  return html`<div class="stack" style="gap:2px;min-width:0" title=${title}>
    <span class="caps" style="font-size:10px">${label}</span>
    <span class=${["num", "strong", cls].filter(Boolean).join(" ")} style="font-size:var(--fs-md);line-height:1.2">${value == null || value === "" ? fmt.DASH : value}</span>
    ${sub ? html`<span class="xs faint num">${sub}</span>` : null}
  </div>`;
}

const TRAILING = [["1w", "1W"], ["1m", "1M"], ["3m", "3M"], ["6m", "6M"], ["ytd", "YTD"], ["1y", "1Y"], ["all", "ALL"]];

/** Price chart: adjusted close as an area, BUY/SELL markers as distinct glyphs
    on the price of that day, one crosshair tooltip that names the trade. */
function priceOption({ dates, values, markers, name, currency }) {
  const t = tokens();
  const opt = lineOption({ series: [{ name: "Price", values, area: true, color: t.accent }], dates, yFormat: (v) => fmt.money(v, currency), currency });
  const idx = new Map(dates.map((d, i) => [d, i]));
  const mk = (type) => {
    const data = dates.map(() => null);
    for (const m of markers) {
      if (String(m.type).toUpperCase() !== type) continue;
      const i = idx.get(m.date);
      if (i == null || values[i] == null) continue;
      data[i] = { value: values[i], quantity: m.quantity, price: m.price, currency: m.currency, type };
    }
    return data;
  };
  const buys = mk("BUY"), sells = mk("SELL");
  const marker = (nm, data, symbol, color, rotate) => ({
    name: nm, type: "scatter", data, symbol, symbolSize: 11, symbolRotate: rotate, z: 5,
    itemStyle: { color, borderColor: t.chartSurface, borderWidth: 2 },
    emphasis: { scale: 1.3, label: { show: true, position: "top", color: t.text, fontSize: 11, formatter: (p) => (p.data && p.data.quantity != null ? `${nm} ${fmt.qty(p.data.quantity)}` : nm) } },
    tooltip: { show: true },
  });
  const hasBuy = buys.some((d) => d != null), hasSell = sells.some((d) => d != null);
  if (hasBuy) opt.series.push(marker("Buy", buys, "triangle", t.accent, 0));
  if (hasSell) opt.series.push(marker("Sell", sells, "triangle", t.serious, 180));
  if (hasBuy || hasSell) {
    opt.legend = {
      show: true, top: 0, right: 8, itemWidth: 12, itemHeight: 8, itemGap: 14,
      textStyle: { color: t.text2, fontSize: 11 }, inactiveColor: t.text3,
      data: [{ name: "Price", icon: "roundRect" }, ...(hasBuy ? [{ name: "Buy", icon: "triangle" }] : []), ...(hasSell ? [{ name: "Sell", icon: "path://M0,0 L10,0 L5,8 Z" }] : [])],
    };
    opt.grid = { ...opt.grid, top: 32 };
  }
  opt.tooltip.formatter = (params) => {
    const list = (Array.isArray(params) ? params : [params]).filter((p) => p.value != null && !(p.data && typeof p.data === "object" && p.data.value == null));
    if (!list.length) return "";
    const rows = list.map((p) => {
      if (p.seriesType === "scatter") {
        const d = p.data || {};
        return { name: `${d.type || p.seriesName} ${fmt.qty(d.quantity)} @ ${fmt.money(d.price, d.currency || currency)}`, value: "", color: p.color, kind: "swatch" };
      }
      return { name: name || p.seriesName, value: fmt.money(p.value, currency), color: p.color, kind: "line" };
    });
    return tipElement(fmt.date(list[0].axisValue), rows);
  };
  return opt;
}

/** Horizontal diverging bar (HTML) for a correlation in [-1, 1]: warm right of zero, cool left. */
function CorrBar({ value }) {
  const v = value == null ? 0 : Math.max(-1, Math.min(1, value));
  const w = Math.round(Math.abs(v) * 50);
  const color = v >= 0 ? "var(--div-warm)" : "var(--div-cool)";
  return html`<span style="position:relative;display:inline-block;width:100%;height:8px;border-radius:2px;background:var(--color-surface-3)" aria-hidden="true">
    <span style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--color-border-strong)"></span>
    <span style=${`position:absolute;top:0;bottom:0;${v >= 0 ? "left:50%" : `left:${50 - w}%`};width:${w}%;background:${color};border-radius:2px`}></span>
  </span>`;
}

const voided = (row, content) => (row.voided ? html`<span style="text-decoration:line-through;opacity:.6" title=${row.void_reason ? `Voided: ${row.void_reason}` : "Voided"}>${content}</span>` : content);

const TX_COLUMNS = [
  { key: "date", label: "Date", width: 96, render: (r, v) => voided(r, fmt.date(v)) },
  { key: "type", label: "Type", sortable: true, render: (r, v) => html`<span class="row" style="gap:4px"><${TxTypeBadge} type=${v} />${r.voided ? html`<${Badge} kind="neutral" outline title=${r.void_reason || "Voided"}>void<//>` : null}</span>` },
  { key: "quantity", label: "Qty", numeric: true, render: (r, v) => voided(r, fmt.qty(v)) },
  { key: "price_per_unit", label: "Price", numeric: true, render: (r, v) => voided(r, fmt.money(v, r.currency)) },
  { key: "fees", label: "Fees", numeric: true, render: (r, v) => voided(r, fmt.money(v, r.currency)) },
  { key: "net", label: "Net", numeric: true, title: "Signed cash effect in the transaction currency", render: (r, v) => voided(r, html`<span class=${polarity(v)}>${fmt.money(v, r.currency, { signed: true })}</span>`) },
];

/* ---------------- panel ---------------- */

export default function HoldingDetailPanel({ isin }) {
  const snapshot = useStore((s) => s.snapshot);
  const theme = useStore((s) => s.theme);
  const base = (snapshot && snapshot.meta && snapshot.meta.base_currency) || "EUR";
  const snapHolding = snapshot && ((snapshot.holdings || []).find((x) => x.isin === isin) || (snapshot.watchlist || []).find((x) => x.isin === isin));

  const [state, setState] = useState({ detail: null, error: null, loading: true, isin: null });
  const [range, setRange] = useState("1Y");
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null, detail: s.isin === isin ? s.detail : null, isin }));
    getHoldingDetail(isin).then((detail) => { if (alive) setState({ detail, error: null, loading: false, isin }); })
      .catch((err) => { if (alive) setState((s) => ({ ...s, error: err, loading: false, isin })); });
    return () => { alive = false; };
  }, [isin, tick, snapshot && snapshot.meta && snapshot.meta.generated_at]);

  const detail = state.isin === isin ? state.detail : null;
  const h = (detail && detail.holding) || snapHolding;
  const inst = detail && detail.instrument;
  const isHolding = !!(h && "quantity" in h);
  const currency = (h && h.price_currency) || (inst && inst.quote_currency) || base;

  // ---- price chart (client-side range slice) ----
  const prices = detail && detail.prices ? detail.prices : { dates: [], values: [] };
  const start = useMemo(() => rangeStartIndex(prices.dates, range), [prices.dates, range]);
  const sliced = useMemo(() => ({
    dates: prices.dates.slice(start), values: prices.values.slice(start),
    dd: detail && detail.drawdown ? detail.drawdown.values.slice(start) : [],
  }), [prices, start, detail]);
  const markers = (detail && detail.markers) || [];
  const beforeHistory = useMemo(() => (prices.dates.length ? markers.filter((m) => m.date < prices.dates[0]).length : 0), [markers, prices.dates]);
  const inRange = useMemo(() => markers.filter((m) => sliced.dates.length && m.date >= sliced.dates[0]), [markers, sliced.dates]);

  const priceTable = useMemo(() => seriesTable({ dates: sliced.dates, series: [{ name: "Price", values: sliced.values }, { name: "Drawdown", values: sliced.dd }], format: (v) => fmt.num(v, { decimals: 2 }) }), [sliced]);

  if (!h && !state.loading && state.error) {
    return html`<${EmptyState} icon="search" title="Not available" body=${errorMessage(state.error)} action=${html`<${Button} variant="secondary" onClick=${() => setTick(tick + 1)}>Try again<//>`} />`;
  }
  if (!h) {
    return html`<div class="faint small" style="padding:24px;text-align:center">Loading ${isin}…</div>`;
  }

  const stats = (detail && detail.stats) || {};
  const others = ((detail && detail.correlations) || []).map((p) => ({
    isin: p.a === isin ? p.b : p.a, name: p.a === isin ? p.b_name : p.a_name, value: p.correlation, sentence: p.sentence,
  })).sort((a, b) => (b.value ?? -2) - (a.value ?? -2));
  const threshold = (snapshot && snapshot.risk && snapshot.risk.threshold) || 0.85;

  return html`<div class="stack gap-4">
    ${/* ---- identity, provenance, actions ---- */ null}
    <div class="stack" style="gap:10px">
      <div class="row wrap" style="gap:6px">
        <${Badge} kind="neutral">${fmt.text(h.asset_class || (inst && inst.asset_class))}<//>
        ${!isHolding ? html`<${Badge} kind="neutral" outline icon="eye">watchlist<//>` : null}
        ${h.price_is_stale ? html`<${Badge} kind="warn" icon="clock" title=${h.price_note}>stale price<//>` : null}
        ${inst && !inst.active ? html`<${Badge} kind="warn">inactive<//>` : null}
        <span class="small muted">${[h.issuer || (inst && inst.issuer), h.exchange || (inst && inst.exchange), h.symbol].filter(Boolean).join(" · ")}</span>
      </div>
      <div class="row between" style="align-items:flex-end">
        <div>
          <div class="row" style="gap:10px;align-items:baseline">
            <span class="strong" style="font-size:var(--fs-2xl);letter-spacing:-.02em;line-height:1.1">${fmt.money(h.price, currency)}</span>
            ${h.day_change_pct != null ? html`<span class=${["num", polarity(h.day_change_pct)].join(" ")} style="font-size:var(--fs-md)">
              ${fmt.arrow(h.day_change_pct)} ${isHolding && h.day_change != null ? fmt.money(h.day_change, base, { signed: true }) + " · " : ""}${fmt.pct(h.day_change_pct, { signed: true })}
            </span>` : null}
          </div>
          <div class="xs faint" style="margin-top:4px">${h.price_note || (h.price == null ? "No price available." : fmt.delayNote(h.price_as_of, h.price_delay_minutes))}${h.fx_note ? ` · ${h.fx_note}` : ""}</div>
        </div>
        <div class="row" style="gap:6px;flex-shrink:0">
          <${Button} size="sm" variant="secondary" icon="plus" onClick=${() => navigate(`/transactions?isin=${encodeURIComponent(isin)}&new=1`)}>Record<//>
          <${Button} size="sm" variant="primary" icon="sliders" onClick=${() => navigate(`/simulator?isin=${encodeURIComponent(isin)}`)}>Simulate<//>
        </div>
      </div>
      ${h.warnings && h.warnings.length ? html`<${Banner} kind="warn">${h.warnings.join(" ")}<//>` : null}
    </div>

    ${/* ---- the figures the holdings table shows ---- */ null}
    ${isHolding ? html`<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px 16px;padding:12px 14px;border:1px solid var(--color-border);border-radius:var(--radius-card)">
      <${Stat} label="Value" value=${fmt.money(h.value, base)} />
      <${Stat} label="P&L" value=${fmt.money(h.unrealised, base, { signed: true })} sub=${fmt.pct(h.unrealised_pct, { signed: true }) + " of cost"} cls=${polarity(h.unrealised)} />
      <${Stat} label="Weight" value=${fmt.pct(h.weight)} sub="of portfolio value" />
      <${Stat} label="Risk share" value=${fmt.pct(h.risk_share)} sub="of portfolio volatility" />
      <${Stat} label="Divergence" value=${fmt.pp(h.divergence)} sub=${h.divergence == null ? "not in the risk model" : h.divergence > 0 ? "more risk than capital" : "less risk than capital"} cls=${h.divergence != null && h.divergence > 0.0005 ? "neg" : ""} />
      <${Stat} label="Quantity" value=${fmt.qty(h.quantity)} sub=${`${fmt.int(h.transaction_count)} ${h.transaction_count === 1 ? "transaction" : "transactions"}`} />
      <${Stat} label="Avg cost" value=${fmt.money(h.avg_cost, base)} sub=${"cost basis " + fmt.money(h.cost_basis, base)} />
      <${Stat} label="Realised" value=${fmt.money(h.realised, base, { signed: true })} sub=${`${fmt.money(h.dividends, base)} dividends · ${fmt.money(h.fees, base)} fees`} cls=${polarity(h.realised)} />
    </div>` : html`<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px 16px;padding:12px 14px;border:1px solid var(--color-border);border-radius:var(--radius-card)">
      <${Stat} label="Volatility" value=${fmt.pct(h.volatility)} sub="annualised, standalone" />
      <${Stat} label="Risk model" value=${h.in_risk_model ? "in model" : "no history"} sub=${h.in_risk_model ? "can be simulated" : "too little overlapping history"} />
      <${Stat} label="Held" value="—" sub="not in the portfolio" />
    </div>`}

    ${/* ---- errors / loading for the fetched part ---- */ null}
    ${state.error ? html`<${Banner} kind="error" title="Could not load the history." action=${html`<${Button} size="sm" variant="secondary" onClick=${() => setTick(tick + 1)}>Retry<//>`}>${errorMessage(state.error)}<//>` : null}
    ${!detail && state.loading ? html`<div class="faint small" style="padding:12px 0;text-align:center">Loading price history, statistics and transactions…</div>` : null}

    ${detail ? html`
      ${/* ---- price + drawdown ---- */ null}
      <${Chart} title="Price" caption=${`Adjusted close in ${currency}${inRange.length ? `, with ${inRange.length} ${inRange.length === 1 ? "trade" : "trades"} marked` : ""}${beforeHistory ? ` · ${beforeHistory} ${beforeHistory === 1 ? "trade predates" : "trades predate"} the price history` : ""}`}
        actions=${html`<${RangeSelector} value=${range} onChange=${setRange} />`}
        height=${210} deps=${[sliced, theme, currency]} table=${priceTable} empty="No price history"
        buildOption=${() => (sliced.dates.length ? priceOption({ dates: sliced.dates, values: sliced.values, markers: inRange, name: "Price", currency }) : null)} />
      ${sliced.dd.length ? html`<${Chart} title="Drawdown" caption="Distance below the running peak of the price, same range." height=${120} deps=${[sliced, theme]}
        buildOption=${() => drawdownOption(sliced.dates, sliced.dd, { name: "Drawdown" })} />` : null}

      ${/* ---- statistics ---- */ null}
      <${Card} title="Statistics" caption=${`Over the ${fmt.int(snapshot && snapshot.lookback)}-day risk window, against the whole portfolio.`} tight>
        <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px 16px;padding-top:4px">
          <${Stat} label="Volatility" value=${fmt.pct(stats.volatility)} sub="annualised" />
          <${Stat} label="Max drawdown" value=${fmt.pct(stats.max_drawdown)} sub=${"now " + fmt.pct(stats.current_drawdown)} />
          <${Stat} label="Beta vs portfolio" value=${fmt.num(stats.beta_vs_portfolio)} sub="its move per 1% portfolio move" />
          <${Stat} label="Correlation" value=${fmt.num(stats.correlation_vs_portfolio)} sub="with the portfolio" />
          <${Stat} label="Contribution" value=${fmt.pct(stats.contribution, { signed: true })} sub="of portfolio return" cls=${polarity(stats.contribution)} />
          <${Stat} label="P&L contribution" value=${fmt.money(stats.pnl_contribution, base, { signed: true })} sub="over the value history" cls=${polarity(stats.pnl_contribution)} />
        </div>
        <div class="caps" style="font-size:10px;margin-top:14px">Trailing returns</div>
        <div style="display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;margin-top:6px">
          ${TRAILING.map(([k, label]) => html`<div key=${k} class="stack" style="gap:1px">
            <span class="xs faint">${label}</span>
            <span class=${["num", "small", "strong", polarity(stats[`trailing_${k}`])].join(" ")}>${fmt.pct(stats[`trailing_${k}`], { signed: true })}</span>
          </div>`)}
        </div>
      <//>

      ${/* ---- correlations ---- */ null}
      <${Card} title="Correlation with other holdings" caption=${`Daily returns over the risk window. Above ${fmt.num(threshold, { decimals: 2 })} the two move as one bet.`} tight>
        ${others.length ? html`<div class="stack" style="gap:6px;padding-top:4px">
          ${others.map((o) => html`<div key=${o.isin} class="row" style="gap:10px" title=${o.sentence || `${fmt.shortName(o.name, 60)}: ${fmt.num(o.value)}`}>
            <a href=${`#/holdings/${o.isin}`} class="truncate small" style="width:150px;flex-shrink:0;color:var(--color-text)" onClick=${(e) => { e.preventDefault(); navigate(`/holdings/${o.isin}`); }}>${fmt.shortName(o.name, 26)}</a>
            <span class="grow"><${CorrBar} value=${o.value} /></span>
            <span class=${["num", "small", "strong", o.value != null && o.value >= threshold ? "neg" : ""].join(" ")} style="width:44px;text-align:right">${fmt.num(o.value)}</span>
          </div>`)}
        </div>` : html`<div class="faint small" style="padding:8px 0">No other holding in the risk model to compare with.</div>`}
      <//>

      ${/* ---- ledger ---- */ null}
      <${Card} title="Transactions" caption=${detail.transactions && detail.transactions.length ? `${detail.transactions.length} for this instrument, newest first.` : "None recorded for this instrument."} flush>
        <${DataTable} columns=${TX_COLUMNS} rows=${detail.transactions || []} rowKey="id" sort=${{ key: "date", dir: "desc" }} empty="No transactions" loading=${false}
          onRowClick=${(row) => navigate(`/transactions/${encodeURIComponent(row.id)}`)} />
      <//>
    ` : null}
  </div>`;
}
