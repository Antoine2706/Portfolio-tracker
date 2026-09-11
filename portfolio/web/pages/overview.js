/* Overview — the front page, answerable in five seconds. Props: { route, snapshot }.

   Layout order, and why:
   1. What is wrong with the data (meta.failures) before any number that
      depends on it.
   2. The figures: portfolio value as the one hero number, then today's move,
      unrealised, realised (with dividends and fees) and invested — the P&L
      identity in one row so nothing has to be reconciled elsewhere.
   3. Value history next to the allocation: how the money moved and where it
      sits now, on one line of sight; exposure by class / currency / issuer
      sits under the allocation as compact segmented bars.
   4. Risk headline, the alerts feed and today's movers: what to look at.
   5. The watchlist strip and trailing returns, the calm summary at the bottom.
   First run (user mode, empty ledger) is a welcome card with the three steps
   on a procedural contour backdrop — the only decoration on the page.

   Nothing is computed here beyond formatting, sorting, filtering and range
   slicing; the two client-side sums (value − invested in the tooltip,
   realised + dividends − fees on the tile) are display arithmetic the page
   is asked to show side by side with their terms. */

import { html, useMemo, useState } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { KpiTile } from "/static/components/KpiTile.js";
import { Chart } from "/static/components/Chart.js";
import { Notice } from "/static/components/Notice.js";
import { Badge, SeverityBadge } from "/static/components/Badge.js";
import { Button } from "/static/components/Button.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Icon } from "/static/components/Icons.js";
import { Sparkline } from "/static/components/Sparkline.js";
import { Segmented, RangeSelector, RANGES, rangeStartIndex } from "/static/components/Segmented.js";
import { useStore } from "/static/lib/store.js";
import { navigate, href } from "/static/lib/router.js";
import { lineOption, barOption, treemapOption, seriesTable, tipElement, seriesColor, foldOther } from "/static/lib/charts.js";
import { tokens } from "/static/lib/theme.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- page-local helpers ---------------- */

function failureAlerts(meta) {
  const failures = (meta && meta.failures) || [];
  return failures.map((f) => ({ code: `fetch:${f.key}`, severity: "SERIOUS", title: `${f.name || f.key}: price fetch failed`, detail: f.message, isins: [], route: "#/instruments" }));
}

const SEV_RANK = { CRITICAL: 3, SERIOUS: 2, WARNING: 1, INFO: 0 };

/** First non-null value at or after `from` in a series. */
function firstFinite(values, from = 0) {
  for (let i = from; i < values.length; i++) if (values[i] != null && Number.isFinite(values[i])) return { i, v: values[i] };
  return null;
}

/** Route link that keeps middle-click / copy-link working but navigates in-app on click. */
function go(path) {
  return (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button === 1) return;
    e.preventDefault();
    navigate(path);
  };
}

/** Value history: value (accent area) with invested as a thin muted line, the
    benchmark optionally re-based to the portfolio value at the range start
    (one axis, common base), BUY/SELL flows as small triangles on the value
    line, and one crosshair tooltip listing value, invested, gain and trades. */
function valueOption({ dates, value, invested, benchmark, benchmarkName, flows, currency }) {
  const t = tokens();
  const series = [
    { name: "Value", values: value, color: t.accent, area: true },
    { name: "Invested", values: invested, color: t.text3, width: 1 },
  ];
  if (benchmark) series.push({ name: benchmarkName || "Benchmark", values: benchmark, color: t.series[1], width: 1.5 });
  const opt = lineOption({ series, dates, yFormat: "money-compact", currency });

  const idx = new Map(dates.map((d, i) => [d, i]));
  const buckets = { BUY: dates.map(() => null), SELL: dates.map(() => null) };
  let hasBuy = false, hasSell = false;
  for (const f of flows || []) {
    const type = String(f.type || "").toUpperCase();
    if (type !== "BUY" && type !== "SELL") continue;
    const i = idx.get(f.date);
    if (i == null || value[i] == null) continue;
    const arr = buckets[type];
    if (!arr[i]) arr[i] = { value: value[i], flows: [] };
    arr[i].flows.push(f);
    if (type === "BUY") hasBuy = true; else hasSell = true;
  }
  const marker = (name, data, color, rotate) => ({
    name, type: "scatter", data, symbol: "triangle", symbolRotate: rotate, symbolSize: 9, z: 6,
    itemStyle: { color, borderColor: t.chartSurface, borderWidth: 1.5 },
    emphasis: { scale: 1.3 },
  });
  if (hasBuy) opt.series.push(marker("Buys", buckets.BUY, t.good, 0));
  if (hasSell) opt.series.push(marker("Sells", buckets.SELL, t.bad, 180));
  if (hasBuy || hasSell) {
    opt.legend = { ...opt.legend, show: true, data: [...series.map((s) => s.name), ...(hasBuy ? [{ name: "Buys", icon: "triangle" }] : []), ...(hasSell ? [{ name: "Sells", icon: "path://M0,0 L10,0 L5,8 Z" }] : [])] };
  }

  opt.tooltip.formatter = (params) => {
    const list = Array.isArray(params) ? params : [params];
    const rows = [];
    const foot = [];
    let v = null, inv = null;
    for (const p of list) {
      if (p.seriesType === "scatter") {
        for (const f of (p.data && p.data.flows) || []) foot.push(`${f.type} ${fmt.money(Math.abs(f.amount), currency)} · ${f.short_name || f.name}`);
        continue;
      }
      if (p.value == null) continue;
      if (p.seriesName === "Value") v = p.value;
      if (p.seriesName === "Invested") inv = p.value;
      rows.push({ name: p.seriesName, value: fmt.money(p.value, currency), color: p.color, kind: "line" });
    }
    if (!rows.length && !foot.length) return "";
    if (v != null && inv != null) rows.push({ name: "Gain", value: fmt.money(v - inv, currency, { signed: true }), polarity: fmt.polarityClass(v - inv) });
    const el = tipElement(fmt.date(list[0].axisValue), rows, foot.length ? foot.join("\n") : undefined);
    if (foot.length) el.lastChild.style.whiteSpace = "pre-line";
    return el;
  };
  return opt;
}

/* ---------------- KPI strip ---------------- */

function KpiRow({ totals, meta, base }) {
  const t = totals || {};
  const dayNull = t.day_change == null;
  const dayReason = !t.priced_holdings ? "no priced holdings" : "no previous close to compare with";
  const netRealised = [t.realised, t.dividends, t.fees].every((v) => v != null) ? t.realised + t.dividends - t.fees : null;
  const holdingsSub = t.priced_holdings != null
    ? `${fmt.int(t.priced_holdings)} priced${t.unpriced_holdings ? `, ${fmt.int(t.unpriced_holdings)} unpriced` : ""}${meta ? ` · ${fmt.int(meta.transaction_count)} ${meta.transaction_count === 1 ? "transaction" : "transactions"}` : ""}`
    : null;
  return html`<div class="ov-kpis">
    <${KpiTile} class="ov-hero" size="hero" label="Portfolio value" value=${fmt.money(t.value, base)} sub=${holdingsSub}
      help="Market value of every priced holding in the base currency, at the last delayed price. Unpriced holdings are excluded." />
    <${KpiTile} label="Day change" value=${dayNull ? fmt.DASH : fmt.money(t.day_change, base, { signed: true })} polarity=${fmt.polarityClass(t.day_change)}
      delta=${dayNull ? null : t.day_change_pct} deltaFormat="pct" deltaLabel=${dayNull ? null : "vs previous close"} sub=${dayNull ? dayReason : null}
      help="Change in value since the previous close, across priced holdings." />
    <${KpiTile} label="Unrealised P&L" value=${fmt.money(t.unrealised, base, { signed: true })} polarity=${fmt.polarityClass(t.unrealised)}
      delta=${t.unrealised_pct} deltaFormat="pct" deltaLabel="of cost" help="Value minus cost basis of the open positions. The percentage is against cost, not against invested cash." />
    <${KpiTile} label="Realised" value=${fmt.money(netRealised, base, { signed: true })} polarity=${fmt.polarityClass(netRealised)}
      sub=${`${fmt.money(t.realised, base, { signed: true })} realised · ${fmt.money(t.dividends, base)} dividends · ${fmt.money(t.fees, base)} fees`}
      help="Realised gains and losses (average-cost method) plus dividends received, minus every fee in the ledger." />
    <${KpiTile} label="Invested" value=${fmt.money(t.invested, base)} sub=${`cost basis ${fmt.money(t.cost_basis, base)}`}
      help="Net external cash put in: buys plus fees, minus sells and dividends taken out. Cost basis is what the open positions cost." />
  </div>`;
}

/* ---------------- value history ---------------- */

function ValueChart({ perf, benchmarks, selectedBenchmark, base }) {
  const theme = useStore((s) => s.theme);
  const [range, setRange] = useState("ALL");
  const [showBench, setShowBench] = useState(false);
  const ok = !!(perf && perf.available && perf.dates && perf.dates.length > 1);
  const bench = (benchmarks || []).find((b) => b.symbol === selectedBenchmark);
  const benchName = bench ? (bench.index || bench.label || bench.symbol) : "Benchmark";

  const rangeOptions = useMemo(() => RANGES.map((k) => {
    const shorter = ok && k !== "ALL" && k !== "YTD" && rangeStartIndex(perf.dates, k) === 0;
    return { value: k, label: k, disabled: shorter, tip: shorter ? "History is shorter than this range" : undefined };
  }), [perf, ok]);

  const slice = useMemo(() => {
    if (!ok) return null;
    const start = rangeStartIndex(perf.dates, range);
    const dates = perf.dates.slice(start);
    const value = perf.value.slice(start);
    const invested = perf.invested.slice(start);
    let benchmark = null;
    if (showBench && perf.benchmark_index && perf.benchmark_index.length) {
      const bi = perf.benchmark_index.slice(start);
      const b0 = firstFinite(bi), v0 = b0 ? firstFinite(value, b0.i) : null;
      if (b0 && v0 && b0.v) benchmark = bi.map((b) => (b == null ? null : (b / b0.v) * v0.v));
    }
    const flows = (perf.flows || []).filter((f) => f.date >= dates[0]);
    return { dates, value, invested, benchmark, flows };
  }, [perf, range, showBench, ok]);

  const table = useMemo(() => (slice ? seriesTable({
    dates: slice.dates,
    series: [{ name: "Value", values: slice.value }, { name: "Invested", values: slice.invested }, ...(slice.benchmark ? [{ name: `${benchName} (re-based)`, values: slice.benchmark }] : [])],
    format: "money", currency: base,
  }) : null), [slice, base, benchName]);

  if (!ok) {
    return html`<${Card} title="Value" caption="Market value against cash invested." class="col-8">
      <${EmptyState} compact icon="performance" title="No value history yet" body=${(perf && perf.reason) || "The value history needs at least one priced holding with a transaction."} />
    <//>`;
  }

  const caption = `Market value against cash invested${slice && slice.benchmark ? `, with ${benchName} re-based to the portfolio value at the start of the range` : ""}. Triangles are buys and sells — hover for the trade.`;
  return html`<${Chart} title="Value" caption=${caption} height=${300} class="col-8 ov-value"
    actions=${html`
      <${Button} size="sm" variant="ghost" aria-pressed=${showBench ? "true" : "false"} onClick=${() => setShowBench(!showBench)}
        title=${showBench ? `Hide ${benchName}` : `Overlay ${benchName}, indexed to the same base`}
        style=${showBench ? "box-shadow: inset 0 0 0 1px var(--color-border-strong)" : ""}>vs ${benchName}<//>
      <${RangeSelector} value=${range} onChange=${setRange} options=${rangeOptions} />`}
    deps=${[slice, theme]} table=${table}
    buildOption=${() => (slice ? valueOption({ ...slice, benchmarkName: benchName, currency: base }) : null)} />`;
}

/* ---------------- allocation + exposure ---------------- */

// Base currency only. Asset class and issuer were both largely legible from
// the treemap beside them -- the tiles are already the holdings, and a reader
// who knows the holdings knows their wrappers and providers. Currency is the
// one of the three that the treemap genuinely cannot show, and a book split
// 60/40 between euro and dollar assets is carrying a real exposure that
// appears nowhere else on the page.
const EXPOSURE_DIMS = [["base_currency", "Currency exposure"]];

function ExposureBars({ exposure }) {
  const t = tokens();
  const rows = EXPOSURE_DIMS.map(([key, label]) => {
    const slices = (exposure && exposure[key]) || [];
    const items = foldOther(slices.map((s) => ({ name: s.label, value: s.weight, count: s.count })), { limit: 8, key: "value" });
    return { key, label, items };
  });
  if (!rows.some((r) => r.items.length)) return html`<div class="ov-expo-empty">No exposure breakdown yet.</div>`;
  return html`<div class="ov-expo">
    ${rows.map((r) => html`<div key=${r.key} class="ov-expo-row">
      <span class="ov-expo-label">${r.label}</span>
      <div class="ov-expo-bar" role="img" aria-label=${`${r.label}: ${r.items.map((i) => `${i.name} ${fmt.pct(i.value)}`).join(", ")}`}>
        ${r.items.map((i, k) => html`<span key=${i.name} class="ov-expo-seg" style=${`flex:${Math.max(0.001, i.value || 0)} 1 0;background:${i.isOther ? t.seriesOther : seriesColor(k, t)}`}
          title=${`${i.name} · ${fmt.pct(i.value)} · ${fmt.count(i.count ?? (i.members ? i.members.reduce((s, m) => s + (m.count || 0), 0) : null), "holding")}`}></span>`)}
      </div>
      <div class="ov-expo-legend">
        ${r.items.map((i, k) => html`<span key=${i.name} class="ov-expo-key"><span class="swatch" style=${`background:${i.isOther ? t.seriesOther : seriesColor(k, t)}`}></span>${i.name} <b>${fmt.pct(i.value)}</b></span>`)}
      </div>
    </div>`)}
  </div>`;
}

function AllocationCard({ holdings, exposure, base }) {
  const theme = useStore((s) => s.theme);
  const [view, setView] = useState("treemap");
  const items = useMemo(() => {
    const rows = holdings.filter((h) => h.value != null && h.value > 0).sort((a, b) => b.value - a.value)
      .map((h) => ({
        isin: h.isin, name: h.short_name || h.name, full: h.name,
        value: h.value, weight: h.weight, change: h.day_change_pct,
        sub: [h.symbol, h.isin].filter(Boolean).join(" · "),
      }));
    // A tile is identified by its label when clicked, so two holdings that
    // shorten to the same label would send both clicks to the first one.
    const seen = new Map();
    for (const r of rows) seen.set(r.name, (seen.get(r.name) || 0) + 1);
    return rows.map((r) => (seen.get(r.name) > 1 && r.isin
      ? { ...r, name: `${r.name} · ${r.isin.slice(-4)}` } : r));
  }, [holdings]);
  const table = useMemo(() => ({
    columns: [
      { key: "name", label: "Holding", render: (r, v) => html`<div class="truncate" style="max-width:220px" title=${r.full}>${v}<span class="cell-sub">${r.sub}</span></div>` },
      { key: "value", label: "Value", format: "money", formatOptions: { currency: base }, numeric: true },
      { key: "weight", label: "Weight", format: "pct", numeric: true },
      { key: "change", label: "Today", format: "pct", formatOptions: { signed: true }, numeric: true, polarity: true },
    ],
    rows: items.map((i) => ({ id: i.isin, ...i })),
  }), [items, base]);
  const open = (it) => { if (it) navigate(`/holdings/${it.isin}`); };
  const onClick = (p) => {
    if (p.seriesType === "treemap") open(items.find((x) => x.name === p.name));
    else if (p.seriesType === "bar") open(items[p.dataIndex]);
  };
  const build = () => {
    if (!items.length) return null;
    if (view === "bars") {
      return barOption({ categories: items.map((i) => i.name), values: items.map((i) => i.weight ?? 0), horizontal: true, format: "pct", showLabels: true });
    }
    return treemapOption({ items, currency: base });
  };
  const height = view === "bars" ? Math.max(300, items.length * 26 + 40) : 300;
  const priced = items.filter((i) => i.change != null).length;
  return html`<${Chart} title="Allocation" class="col-4"
    caption=${view === "bars"
      ? "Weight of each holding in portfolio value. Click a bar for the holding."
      : (priced
        ? "Area is value, colour is today's move — green up, red down, deepest at the day's largest mover. Click a tile for the holding."
        : "Area is value. No day change is available yet, so colour carries nothing. Click a tile for the holding.")}
    height=${height} deps=${[items, view, theme]} table=${table} onEvents=${{ click: onClick }} empty="No priced holdings"
    actions=${html`<${Segmented} size="sm" ariaLabel="Allocation view" value=${view} onChange=${setView}
      options=${[{ value: "treemap", label: "Treemap" }, { value: "bars", label: "Bars" }]} />`}
    footer=${html`<${ExposureBars} exposure=${exposure} />`}
    buildOption=${build} />`;
}

/* ---------------- risk headline ---------------- */

function Stat({ label, value, sub, warning }) {
  return html`<div class="ov-stat">
    <span class="ov-stat-label">${label}</span>
    <span class="ov-stat-value">${value == null || value === "" ? fmt.DASH : value}
      ${warning ? html`<span class="warn-mark" title=${warning} tabindex="0" aria-label=${warning}><${Icon} name="alertTriangle" size=${12} stroke=${2} /></span>` : null}</span>
    ${sub ? html`<span class="ov-stat-sub" title=${sub}>${sub}</span>` : null}
  </div>`;
}

function RiskCard({ risk }) {
  const byKey = useMemo(() => Object.fromEntries(((risk && risk.metrics) || []).map((m) => [m.key, m])), [risk]);
  const link = html`<a class="ov-link" href=${href("/risk")} onClick=${go("/risk")}>Open risk →</a>`;
  if (!risk || !risk.available) {
    return html`<${Card} title="Risk" actions=${link} class="col-4">
      <${EmptyState} compact icon="risk" title="Risk model not available" body=${(risk && risk.reason) || "The risk model has nothing to work with yet."} />
    <//>`;
  }
  const head = risk.headline;
  const m = (k) => byKey[k] || {};
  const benchShort = risk.beta_benchmark ? fmt.shortName(risk.beta_benchmark.replace(/\s*\(.*\)$/, ""), 22) : "benchmark";
  return html`<${Card} title="Risk" caption=${risk.window ? `${fmt.int(risk.window.effective)}-day window to ${fmt.date(risk.window.last_date)}` : null} actions=${link} class="col-4">
    <div class="stack gap-3">
      ${head ? html`<p class="ov-lead">${head.sentence}</p>` : html`<p class="ov-lead muted">Risk and capital are in line across the book.</p>`}
      <div class="ov-riskgrid">
        <${Stat} label="Effective holdings" value=${m("effective_holdings").display || (risk.effective_holdings != null ? `${fmt.num(risk.effective_holdings, { decimals: 1 })} of ${fmt.int(risk.actual_holdings)}` : null)} sub="independent bets, not positions" warning=${m("effective_holdings").warning} />
        <${Stat} label="Volatility, annualised" value=${m("volatility").display || fmt.pct(risk.volatility)} sub=${risk.volatility_multiple != null ? `${fmt.mult(risk.volatility_multiple)} a broad index` : null} warning=${m("volatility").warning} />
        <${Stat} label="Max drawdown" value=${m("max_drawdown").display || fmt.pct(risk.max_drawdown)} sub=${risk.current_drawdown != null ? `now ${fmt.pct(risk.current_drawdown)} from the peak` : null} warning=${m("max_drawdown").warning} />
        <${Stat} label=${`Beta vs ${benchShort}`} value=${m("beta").display || fmt.num(risk.beta)} sub="its move per 1% index move" warning=${m("beta").warning} />
      </div>
    </div>
  <//>`;
}

/* ---------------- alerts ---------------- */

function AlertItem({ a }) {
  const path = a.route || "/holdings";
  return html`<li>
    <a class="ov-alert" href=${href(path)} onClick=${go(path)} title=${`Open ${path.replace(/^#?\//, "")}`}>
      <${SeverityBadge} severity=${a.severity} showIcon=${false} />
      <span class="ov-alert-text">
        <span class="ov-alert-title">${a.title}</span>
        ${a.detail ? html`<span class="ov-alert-detail">${a.detail}</span>` : null}
      </span>
      <span class="ov-alert-go" aria-hidden="true">→</span>
    </a>
  </li>`;
}

function AlertsCard({ alerts }) {
  const sorted = useMemo(() => [...(alerts || [])].sort((a, b) => (SEV_RANK[b.severity] ?? 0) - (SEV_RANK[a.severity] ?? 0) || a.title.localeCompare(b.title)), [alerts]);
  const loud = sorted.filter((a) => a.severity !== "INFO");
  const info = sorted.filter((a) => a.severity === "INFO");
  const caption = sorted.length ? `${fmt.count(sorted.length, "item")}, worst first` : null;
  return html`<${Card} title="Needs attention" caption=${caption} class="col-4" flush=${sorted.length > 0}>
    ${sorted.length ? html`
      ${loud.length ? html`<ul class="ov-alerts">${loud.map((a) => html`<${AlertItem} key=${a.code + a.title} a=${a} />`)}</ul>` : null}
      ${info.length ? (loud.length ? html`<details class="ov-more">
          <summary><${Icon} name="chevronRight" size=${14} />${fmt.count(info.length, "more note")}</summary>
          <ul class="ov-alerts">${info.map((a) => html`<${AlertItem} key=${a.code + a.title} a=${a} />`)}</ul>
        </details>` : html`<ul class="ov-alerts">${info.map((a) => html`<${AlertItem} key=${a.code + a.title} a=${a} />`)}</ul>`) : null}`
      : html`<div class="ov-ok"><span class="ov-ok-icon"><${Icon} name="checkCircle" size=${15} /></span><span>Nothing needs attention. No stale prices, no concentration or correlation breaches.</span></div>`}
  <//>`;
}

/* ---------------- movers + watchlist ---------------- */

function MoverRow({ h, base }) {
  const path = `/holdings/${h.isin}`;
  return html`<a class="ov-row" href=${href(path)} onClick=${go(path)} title=${h.name}>
    <span class="ov-row-text">
      <span class="ov-row-name" title=${h.name}>${h.short_name || h.name}</span>
      <span class="ov-row-sub">${h.symbol || h.isin}${h.weight != null ? ` · ${fmt.pct(h.weight)} of value` : ""}</span>
    </span>
    <span class="ov-row-spark"><${Sparkline} values=${h.sparkline || []} area=${false} endDot=${false} /></span>
    <span class="ov-row-num">
      <span class=${["ov-row-pct", fmt.polarityClass(h.day_change_pct)].join(" ")}>${fmt.pct(h.day_change_pct, { signed: true })}</span>
      <span class="ov-row-amt">${fmt.money(h.day_change, base, { signed: true })}</span>
    </span>
  </a>`;
}

function MoversCard({ holdings, base }) {
  const movers = useMemo(() => holdings.filter((h) => h.day_change_pct != null).sort((a, b) => Math.abs(b.day_change_pct) - Math.abs(a.day_change_pct)).slice(0, 5), [holdings]);
  return html`<${Card} title="Today's movers" caption=${movers.length ? "Largest moves since the previous close" : null} class="col-4" tight>
    ${movers.length
      ? html`<div class="ov-list">${movers.map((h) => html`<${MoverRow} key=${h.isin} h=${h} base=${base} />`)}</div>`
      : html`<div class="ov-watch-empty">No day change available yet.</div>`}
  <//>`;
}

function WatchRow({ w }) {
  const path = `/holdings/${w.isin}`;
  return html`<a class="ov-row" href=${href(path)} onClick=${go(path)} title=${w.name}>
    <span class="ov-row-text">
      <span class="ov-row-name" title=${w.name}>${w.short_name || w.name}</span>
      <span class="ov-row-sub">${w.symbol || w.isin}</span>
    </span>
    ${w.in_risk_model
      ? html`<span class="ov-watch-flag good" data-tip="In the risk model: enough overlapping history to simulate with" data-tip-pos="top" aria-label="In the risk model"><${Icon} name="check" size=${11} /></span>`
      : html`<span class="ov-watch-flag" data-tip="Not enough history yet to sit in the covariance matrix" data-tip-pos="top" aria-label="No history yet"><${Icon} name="clock" size=${11} /></span>`}
    <span class="ov-row-spark"><${Sparkline} values=${w.sparkline || []} area=${false} endDot=${false} /></span>
    <span class="ov-row-price">${fmt.money(w.price, w.price_currency)}</span>
    <span class="ov-row-num" style="width:56px"><span class=${["ov-row-pct", fmt.polarityClass(w.day_change_pct)].join(" ")}>${fmt.pct(w.day_change_pct, { signed: true })}</span></span>
  </a>`;
}

function WatchlistStrip({ watchlist }) {
  return html`<${Card} title="Watchlist" caption="Priced instruments in the universe that are not held. Open one for its history and correlation with the portfolio." tight
    actions=${html`<a class="ov-link" href=${href("/instruments")} onClick=${go("/instruments")}>Add instrument →</a>`}>
    ${watchlist.length
      ? html`<div class="ov-watch">${watchlist.map((w) => html`<${WatchRow} key=${w.isin} w=${w} />`)}</div>`
      : html`<div class="ov-watch-empty">Nothing on the watchlist. Instruments in the universe that you do not hold show up here.</div>`}
  <//>`;
}

/* ---------------- trailing returns ---------------- */

const TRAILING = [["1w", "1 week"], ["1m", "1 month"], ["3m", "3 months"], ["6m", "6 months"], ["ytd", "Year to date"], ["1y", "1 year"], ["all", "Since start"]];

function TrailingStrip({ perf, benchName }) {
  const ok = perf && perf.available && perf.trailing;
  if (!ok) return null;
  const tr = perf.trailing || {};
  const bt = perf.benchmark_trailing || {};
  return html`<${Card} title="Trailing returns" caption=${`Time-weighted. Portfolio above, ${benchName} below.${perf.summary && perf.summary.last_date ? ` To ${fmt.date(perf.summary.last_date)}.` : ""}`} tight>
    <div class="ov-trailing">
      ${TRAILING.map(([k, label]) => html`<div key=${k} class="ov-trail">
        <span class="xs faint">${label}</span>
        <span class=${["ov-trail-value", fmt.polarityClass(tr[k])].join(" ")}>${fmt.pct(tr[k], { signed: true })}</span>
        <span class="ov-trail-bench" title=${benchName}>${fmt.pct(bt[k], { signed: true })} <span class="faint">index</span></span>
      </div>`)}
    </div>
  <//>`;
}

/* ---------------- first run ---------------- */

/** Closed Catmull-Rom spline as an SVG path. */
function smoothClosed(pts) {
  const n = pts.length;
  const f = (v) => v.toFixed(1);
  let d = `M${f(pts[0][0])},${f(pts[0][1])}`;
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
    d += `C${f(p1[0] + (p2[0] - p0[0]) / 6)},${f(p1[1] + (p2[1] - p0[1]) / 6)} ${f(p2[0] - (p3[0] - p1[0]) / 6)},${f(p2[1] - (p3[1] - p1[1]) / 6)} ${f(p2[0])},${f(p2[1])}`;
  }
  return d + "Z";
}

/** Concentric, gently perturbed rings around a hill — a contour map, deterministic. */
function contours({ cx, cy, rings, step, seed, squash = 0.62 }) {
  const out = [];
  for (let k = 1; k <= rings; k++) {
    const r = k * step;
    const pts = [];
    const n = 44;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const w = 1 + 0.17 * Math.sin(3 * a + seed) + 0.10 * Math.sin(5 * a - seed * 1.3 + k * 0.12) + 0.06 * Math.sin(8 * a + seed * 0.7 + k * 0.25);
      pts.push([cx + r * w * Math.cos(a), cy + r * w * squash * Math.sin(a)]);
    }
    out.push(smoothClosed(pts));
  }
  return out;
}

const CONTOUR_PATHS = [
  ...contours({ cx: 960, cy: 140, rings: 12, step: 30, seed: 1.3 }),
  ...contours({ cx: 330, cy: 400, rings: 7, step: 28, seed: 4.1, squash: 0.7 }),
];

function ContourBackdrop() {
  return html`<div class="ov-welcome-bg" aria-hidden="true">
    <svg viewBox="0 0 1200 420" preserveAspectRatio="xMidYMid slice">
      ${CONTOUR_PATHS.map((d, i) => html`<path key=${i} d=${d} fill="none" stroke="currentColor" stroke-width="1" vector-effect="non-scaling-stroke" />`)}
    </svg>
  </div>`;
}

function Welcome({ meta }) {
  return html`<section class="card ov-welcome">
    <${ContourBackdrop} />
    <div class="ov-welcome-fade" aria-hidden="true"></div>
    <div class="ov-welcome-body">
      <div class="row" style="gap:8px"><${Badge} kind="live" dot>LIVE<//><span class="xs faint">your own ledger · ${fmt.count(meta ? meta.instrument_count : null, "instrument")} in the universe</span></div>
      <h2 class="ov-welcome-title">Nothing is held yet.</h2>
      <p class="ov-welcome-lead">Three steps stand between this page and your portfolio. Prices, weights, the risk model and the alerts appear as soon as the first BUY is in the ledger.</p>
      <div class="ov-steps">
        <div class="ov-step">
          <span class="ov-step-n">1</span>
          <span class="ov-step-title">Add an instrument</span>
          <span class="ov-step-body">Type an ISIN. The listing is resolved, its price history checked, and the instrument joins the universe.</span>
          <${Button} variant="primary" size="sm" icon="instruments" onClick=${() => navigate("/instruments?new=1")}>Add by ISIN<//>
        </div>
        <div class="ov-step">
          <span class="ov-step-n">2</span>
          <span class="ov-step-title">Record a transaction</span>
          <span class="ov-step-body">Enter a buy by hand, or import the CSV your broker exports — the columns are detected and previewed first.</span>
          <${Button} variant="secondary" size="sm" icon="transactions" onClick=${() => navigate("/transactions?new=1")}>Record or import<//>
        </div>
        <div class="ov-step">
          <span class="ov-step-n">3</span>
          <span class="ov-step-title">Come back here</span>
          <span class="ov-step-body">Value, day change, allocation, risk and what needs attention — the whole book, answerable in five seconds.</span>
          <${Button} variant="ghost" size="sm" icon="settings" onClick=${() => navigate("/settings")}>Or explore the demo data<//>
        </div>
      </div>
    </div>
  </section>`;
}

/* ---------------- page ---------------- */

export default function OverviewPage({ snapshot }) {
  const loading = useStore((s) => s.loading);
  const meta = snapshot && snapshot.meta;
  const base = (meta && meta.base_currency) || "EUR";
  const holdings = (snapshot && snapshot.holdings) || [];
  const watchlist = (snapshot && snapshot.watchlist) || [];
  const perf = snapshot && snapshot.performance;
  const bench = snapshot && (snapshot.benchmarks || []).find((b) => b.symbol === snapshot.selected_benchmark);
  const benchName = bench ? (bench.index || bench.label || bench.symbol) : "benchmark";
  const failures = useMemo(() => failureAlerts(meta), [meta]);
  const delay = useMemo(() => { const d = holdings.map((h) => h.price_delay_minutes).filter((v) => v != null); return d.length ? Math.max(...d) : null; }, [holdings]);

  const subtitle = snapshot
    ? `${fmt.count(holdings.length, "holding")} · ${fmt.count(watchlist.length, "instrument")} on the watchlist · vs ${benchName} · ${fmt.int(snapshot.lookback)}-day risk window · ${fmt.delayNote(meta.prices_as_of, delay)}`
    : "Value, P&L, allocation and what needs attention today.";

  if (!snapshot) {
    return html`<${Page} title="Overview" subtitle=${subtitle}>
      <${Card}><${EmptyState} icon=${loading ? "clock" : "inbox"} title=${loading ? "Loading the snapshot…" : "No snapshot"} body=${loading ? "One request carries everything this page shows." : "The snapshot failed to load; retry from the banner above."} /><//>
    <//>`;
  }

  if (!holdings.length) {
    const firstRun = meta.mode === "user" && !meta.transaction_count;
    return html`<${Page} title="Overview" subtitle=${subtitle}>
      <div class="stack gap-4">
        <${Notice} alerts=${failures} onNavigate=${navigate} />
        ${firstRun ? html`<${Welcome} meta=${meta} />` : html`<${Card}>
          <${EmptyState} icon="holdings" title="No open positions" body=${meta.mode === "seed" ? "The demo book has no open positions." : "Every position in the ledger has been closed. Record a BUY and the book comes back to life."}
            action=${html`<${Button} variant="primary" icon="transactions" onClick=${() => navigate("/transactions?new=1")}>Record a transaction<//>`} />
        <//>`}
        <${WatchlistStrip} watchlist=${watchlist} />
      </div>
    <//>`;
  }

  return html`<${Page} title="Overview" subtitle=${subtitle}>
    <div class="stack gap-4">
      <${Notice} alerts=${failures} summary=${failures.length ? `${failures.length} price ${failures.length === 1 ? "fetch" : "fetches"} failed — the figures below exclude ${failures.length === 1 ? "that instrument" : "those instruments"}` : undefined} onNavigate=${navigate} />
      <${KpiRow} totals=${snapshot.totals} meta=${meta} base=${base} />
      <div class="grid">
        <${ValueChart} perf=${perf} benchmarks=${snapshot.benchmarks} selectedBenchmark=${snapshot.selected_benchmark} base=${base} />
        <${AllocationCard} holdings=${holdings} exposure=${snapshot.exposure} base=${base} />
      </div>
      <div class="grid">
        <${RiskCard} risk=${snapshot.risk} />
        <${AlertsCard} alerts=${snapshot.alerts} />
        <${MoversCard} holdings=${holdings} base=${base} />
      </div>
      <${WatchlistStrip} watchlist=${watchlist} />
      <${TrailingStrip} perf=${perf} benchName=${benchName} />
    </div>
  <//>`;
}
