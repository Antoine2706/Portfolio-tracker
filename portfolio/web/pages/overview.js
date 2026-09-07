/* Overview — the daily cockpit. Props: { route, snapshot }.

   Layout order, and why:
   1. What is wrong with the data (meta.failures) before any number that
      depends on it.
   2. The figures: portfolio value as the one hero number with today's move,
      then unrealised, realised, invested and fees — the whole P&L identity
      in one row so nothing has to be reconciled elsewhere.
   3. Value history next to the allocation treemap: how the money moved and
      where it sits now, on one line of sight.
   4. Risk headline, the alerts feed and today's movers: what to look at.
   5. Trailing returns against the benchmark, the calm summary at the bottom.
   With no holdings at all the page is a welcome card that explains demo
   versus live data — the only place a decorative image is allowed. */

import { html, useMemo, useState } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { KpiTile } from "/static/components/KpiTile.js";
import { Chart } from "/static/components/Chart.js";
import { Notice } from "/static/components/Notice.js";
import { Badge, SeverityBadge } from "/static/components/Badge.js";
import { Button } from "/static/components/Button.js";
import { Icon } from "/static/components/Icons.js";
import { Sparkline } from "/static/components/Sparkline.js";
import { RangeSelector, rangeStartIndex } from "/static/components/Segmented.js";
import { useStore } from "/static/lib/store.js";
import { navigate } from "/static/lib/router.js";
import { lineOption, treemapOption, seriesTable, tipElement } from "/static/lib/charts.js";
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

/** Value history: value (accent) with invested as a thin muted line, the
    benchmark index optionally re-based to the portfolio value at the range
    start (one axis, common base), and every external flow as a small marker
    that names itself in the crosshair tooltip. */
function valueOption({ dates, value, invested, benchmark, benchmarkName, flows, currency }) {
  const t = tokens();
  const series = [
    { name: "Value", values: value, color: t.accent, area: true },
    { name: "Invested", values: invested, color: t.text3, width: 1 },
  ];
  if (benchmark) series.push({ name: benchmarkName || "Benchmark", values: benchmark, color: t.series[1], width: 1.5 });
  const opt = lineOption({ series, dates, yFormat: "money-compact", currency });
  // flows as markers on the value line
  const idx = new Map(dates.map((d, i) => [d, i]));
  const marks = dates.map(() => null);
  let any = false;
  for (const f of flows || []) {
    const i = idx.get(f.date);
    if (i == null || value[i] == null) continue;
    if (!marks[i]) marks[i] = { value: value[i], flows: [] };
    marks[i].flows.push(f);
    any = true;
  }
  if (any) {
    opt.series.push({
      name: "Flows", type: "scatter", data: marks, symbol: "circle", symbolSize: 8, z: 6,
      itemStyle: { color: t.text2, borderColor: t.chartSurface, borderWidth: 2 },
      emphasis: { scale: 1.4 },
    });
    opt.legend.data = [...(opt.legend.data || series.map((s) => s.name)), { name: "Flows", icon: "circle" }];
  }
  opt.tooltip.formatter = (params) => {
    const list = (Array.isArray(params) ? params : [params]).filter((p) => p.value != null && !(p.data && typeof p.data === "object" && p.data.value == null));
    if (!list.length) return "";
    const rows = [];
    let foot = null;
    for (const p of list) {
      if (p.seriesType === "scatter") {
        const fl = (p.data && p.data.flows) || [];
        foot = fl.map((f) => `${f.type} ${fmt.money(Math.abs(f.amount), currency)} · ${fmt.shortName(f.name, 28)}`).join("\n");
        continue;
      }
      rows.push({ name: p.seriesName, value: fmt.money(p.value, currency), color: p.color, kind: "line" });
    }
    const el = tipElement(fmt.date(list[0].axisValue), rows, foot);
    if (foot) el.lastChild.style.whiteSpace = "pre-line";
    return el;
  };
  return opt;
}

/* ---------------- sections ---------------- */

function KpiRow({ totals, meta, base }) {
  const t = totals || {};
  const dayDelta = t.day_change == null ? null : t.day_change;
  const dayFmt = (v) => `${fmt.money(v, base, { signed: true })} · ${fmt.pct(t.day_change_pct, { signed: true })}`;
  return html`<div class="grid" style="grid-template-columns:repeat(5,minmax(0,1fr))">
    <${KpiTile} class="col-12" style="grid-column:span 1" size="hero" label="Portfolio value" value=${fmt.money(t.value, base)}
      delta=${dayDelta} deltaFormat=${dayFmt} deltaLabel="today" sub=${t.priced_holdings != null ? `${fmt.int(t.priced_holdings)} priced${t.unpriced_holdings ? `, ${fmt.int(t.unpriced_holdings)} unpriced` : ""}` : null}
      help="Market value of every priced holding in the base currency, at the last delayed price." />
    <${KpiTile} label="Unrealised P&L" value=${fmt.money(t.unrealised, base, { signed: true })} polarity=${fmt.polarityClass(t.unrealised)}
      delta=${t.unrealised_pct} deltaFormat="pct" deltaLabel="of cost" help="Value minus cost basis of the open positions. The percentage is against cost, not against invested cash." />
    <${KpiTile} label="Realised P&L" value=${fmt.money(t.realised, base, { signed: true })} polarity=${fmt.polarityClass(t.realised)}
      sub=${`${fmt.money(t.dividends, base)} in dividends on top`} help="Gains and losses locked in by sells, average-cost method. Dividends are shown separately and are not part of this figure." />
    <${KpiTile} label="Invested" value=${fmt.money(t.invested, base)} sub=${`cost basis ${fmt.money(t.cost_basis, base)}`}
      help="Net external cash put in: buys plus fees, minus sells and dividends taken out. Cost basis is what the open positions cost." />
    <${KpiTile} label="Fees paid" value=${fmt.money(t.fees, base)} sub=${meta ? `${fmt.int(meta.transaction_count)} transactions` : null}
      help="Every fee in the ledger. Fees are paid from outside, so they depress the return without changing the value." />
  </div>`;
}

function ValueChart({ perf, benchmarks, selectedBenchmark, base }) {
  const theme = useStore((s) => s.theme);
  const [range, setRange] = useState("ALL");
  const [showBench, setShowBench] = useState(false);
  const ok = perf && perf.available && perf.dates && perf.dates.length > 1;
  const bench = (benchmarks || []).find((b) => b.symbol === selectedBenchmark);
  const benchName = bench ? (bench.index || bench.label || bench.symbol) : "Benchmark";

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

  const caption = ok
    ? `Market value against cash invested${slice && slice.benchmark ? `, with ${benchName} re-based to the portfolio value at the start of the range` : ""}. Dots are external flows — hover one to see the trade.`
    : (perf && perf.reason) || "No value history yet.";

  return html`<${Chart} title="Value" caption=${caption} height=${300} class="col-8"
    actions=${html`
      <${Button} size="sm" variant="ghost" aria-pressed=${showBench ? "true" : "false"} onClick=${() => setShowBench(!showBench)} title=${showBench ? `Hide ${benchName}` : `Overlay ${benchName}, indexed to the same base`}
        style=${showBench ? "box-shadow: inset 0 0 0 1px var(--color-border-strong)" : ""}>vs ${benchName}<//>
      <${RangeSelector} value=${range} onChange=${setRange} />`}
    deps=${[slice, theme]} table=${table} empty=${caption}
    buildOption=${() => (slice ? valueOption({ ...slice, benchmarkName: benchName, currency: base }) : null)} />`;
}

function AllocationTreemap({ holdings, base }) {
  const theme = useStore((s) => s.theme);
  const items = useMemo(() => holdings.filter((h) => h.value != null && h.value > 0).map((h) => ({ name: h.name, value: h.value, sub: [h.symbol, h.isin].filter(Boolean).join(" · ") })), [holdings]);
  const table = useMemo(() => ({
    columns: [
      { key: "name", label: "Holding", render: (r, v) => html`<div class="truncate" style="max-width:200px" title=${v}>${fmt.shortName(v, 32)}<span class="cell-sub">${r.sub}</span></div>` },
      { key: "value", label: "Value", format: "money", formatOptions: { currency: base }, numeric: true },
      { key: "weight", label: "Weight", format: "pct", numeric: true },
    ],
    rows: holdings.filter((h) => h.value != null).map((h) => ({ id: h.isin, name: h.name, sub: [h.symbol, h.isin].filter(Boolean).join(" · "), value: h.value, weight: h.weight })),
  }), [holdings, base]);
  const onClick = (p) => {
    const h = holdings.find((x) => x.name === p.name) || null;
    if (h) navigate(`/holdings/${h.isin}`);
  };
  return html`<${Chart} title="Allocation" caption="Area is value; darker is larger. Click a tile for the holding." height=${300} class="col-4"
    deps=${[items, theme]} table=${table} onEvents=${{ click: onClick }} empty="No priced holdings"
    buildOption=${() => (items.length ? treemapOption({ items, currency: base }) : null)} />`;
}

function RiskCard({ risk }) {
  const byKey = useMemo(() => Object.fromEntries(((risk && risk.metrics) || []).map((m) => [m.key, m])), [risk]);
  const actions = html`<${Button} size="sm" variant="ghost" iconRight="chevronRight" onClick=${() => navigate("/risk")}>Risk<//>`;
  if (!risk || !risk.available) {
    return html`<${Card} title="Risk" actions=${actions} class="col-4">
      <div class="small muted" style="display:flex;gap:8px;align-items:flex-start"><span class="faint" style="margin-top:2px"><${Icon} name="info" size=${14} /></span><span>${(risk && risk.reason) || "The risk model has nothing to work with yet."}</span></div>
    <//>`;
  }
  const head = risk.headline;
  const m = (k) => byKey[k] || {};
  const Row = ({ label, value, sub, warning }) => html`<div class="stack" style="gap:1px;min-width:0">
    <span class="xs faint">${label}</span>
    <span class="row" style="gap:6px"><span class="strong num" style="font-size:var(--fs-lg);line-height:1.2">${value == null || value === "" ? fmt.DASH : value}</span>
      ${warning ? html`<span class="warn-mark" title=${warning} tabindex="0"><${Icon} name="alertTriangle" size=${12} stroke=${2} /></span>` : null}</span>
    ${sub ? html`<span class="xs muted truncate" title=${sub}>${sub}</span>` : null}
  </div>`;
  return html`<${Card} title="Risk" caption=${risk.window ? `${fmt.int(risk.window.effective)}-day window to ${fmt.date(risk.window.last_date)}` : null} actions=${actions} class="col-4">
    <div class="stack gap-3">
      ${head ? html`<p class="small" style="line-height:1.5">${head.sentence}</p>` : null}
      <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 16px">
        <${Row} label="Effective holdings" value=${m("effective_holdings").display || (risk.effective_holdings != null ? `${fmt.num(risk.effective_holdings, { decimals: 1 })} of ${fmt.int(risk.actual_holdings)}` : null)} sub="independent bets, not positions" warning=${m("effective_holdings").warning} />
        <${Row} label="Volatility, annualised" value=${m("volatility").display || fmt.pct(risk.volatility)} sub=${risk.volatility_multiple != null ? `${fmt.mult(risk.volatility_multiple)} a broad index` : null} warning=${m("volatility").warning} />
        <${Row} label="Max drawdown" value=${m("max_drawdown").display || fmt.pct(risk.max_drawdown)} sub=${risk.current_drawdown != null ? `now ${fmt.pct(risk.current_drawdown)} below the peak` : null} warning=${m("max_drawdown").warning} />
        <${Row} label=${`Beta vs ${risk.beta_benchmark ? fmt.shortName(risk.beta_benchmark.replace(/\s*\(.*\)$/, ""), 22) : "benchmark"}`} value=${m("beta").display || fmt.num(risk.beta)} sub="its move per 1% index move" warning=${m("beta").warning} />
      </div>
    </div>
  <//>`;
}

function AlertsCard({ alerts }) {
  const sorted = useMemo(() => [...(alerts || [])].sort((a, b) => (SEV_RANK[b.severity] ?? 0) - (SEV_RANK[a.severity] ?? 0) || a.title.localeCompare(b.title)), [alerts]);
  return html`<${Card} title="Needs attention" caption=${sorted.length ? `${sorted.length} ${sorted.length === 1 ? "item" : "items"}, worst first` : null} class="col-4" flush=${sorted.length > 0}>
    ${sorted.length ? html`<ul style="list-style:none;margin:0;padding:6px 0;max-height:300px;overflow:auto">
      ${sorted.map((a) => html`<li key=${a.code + a.title}>
        <a href=${a.route || "#/holdings"} class="row" style="gap:10px;align-items:flex-start;padding:8px 16px;color:inherit;text-decoration:none" onClick=${(e) => { e.preventDefault(); navigate(a.route || "/holdings"); }}
          onMouseEnter=${(e) => { e.currentTarget.style.background = "var(--color-hover)"; }} onMouseLeave=${(e) => { e.currentTarget.style.background = ""; }}>
          <span style="margin-top:1px"><${SeverityBadge} severity=${a.severity} showIcon=${false} /></span>
          <span class="stack" style="gap:2px;min-width:0">
            <span class="small" style="font-weight:500;color:var(--color-text)">${a.title}</span>
            ${a.detail ? html`<span class="xs muted">${a.detail}</span>` : null}
          </span>
        </a>
      </li>`)}
    </ul>` : html`<div class="row small muted" style="gap:8px"><span class="pos"><${Icon} name="checkCircle" size=${15} /></span><span>Nothing needs attention. No stale prices, no concentration or correlation breaches.</span></div>`}
  <//>`;
}

function MoversCard({ holdings, watchlist }) {
  const movers = useMemo(() => {
    const all = [
      ...holdings.map((h) => ({ isin: h.isin, name: h.name, symbol: h.symbol, pct: h.day_change_pct, sparkline: h.sparkline, held: true })),
      ...watchlist.map((w) => ({ isin: w.isin, name: w.name, symbol: w.symbol, pct: w.day_change_pct, sparkline: w.sparkline, held: false })),
    ].filter((x) => x.pct != null);
    const sorted = [...all].sort((a, b) => b.pct - a.pct);
    const n = Math.min(3, Math.floor(sorted.length / 2) || sorted.length);
    const best = sorted.slice(0, n);
    const worst = sorted.slice(Math.max(n, sorted.length - n)).reverse();
    return { best, worst, total: all.length };
  }, [holdings, watchlist]);
  const Row = ({ m }) => html`<a href=${`#/holdings/${m.isin}`} class="row" style="gap:10px;padding:5px 0;color:inherit;text-decoration:none" onClick=${(e) => { e.preventDefault(); navigate(`/holdings/${m.isin}`); }} title=${m.name}>
    <span class="stack grow" style="gap:0">
      <span class="small truncate" style="color:var(--color-text)">${fmt.shortName(m.name, 26)}</span>
      <span class="xs faint">${m.symbol || m.isin}${m.held ? "" : " · watchlist"}</span>
    </span>
    <span style="width:64px;height:20px;flex-shrink:0"><${Sparkline} values=${m.sparkline || []} area=${false} endDot=${false} /></span>
    <span class=${["num", "small", "strong", fmt.polarityClass(m.pct)].join(" ")} style="width:56px;text-align:right">${fmt.pct(m.pct, { signed: true })}</span>
  </a>`;
  return html`<${Card} title="Today's movers" caption=${movers.total ? "Day change across holdings and watchlist" : null} class="col-4" tight>
    ${movers.total ? html`<div style="display:grid;grid-template-columns:1fr;gap:4px">
      <div class="caps" style="font-size:10px;margin-top:2px">Up</div>
      ${movers.best.map((m) => html`<${Row} key=${m.isin} m=${m} />`)}
      <div class="caps" style="font-size:10px;margin-top:6px">Down</div>
      ${movers.worst.map((m) => html`<${Row} key=${m.isin} m=${m} />`)}
    </div>` : html`<div class="faint small" style="padding:8px 0">No day change available yet.</div>`}
  <//>`;
}

const TRAILING = [["1w", "1 week"], ["1m", "1 month"], ["3m", "3 months"], ["6m", "6 months"], ["ytd", "Year to date"], ["1y", "1 year"], ["all", "Since start"]];

function TrailingStrip({ perf, benchName }) {
  const ok = perf && perf.available && perf.trailing;
  const tr = (ok && perf.trailing) || {};
  const bt = (ok && perf.benchmark_trailing) || {};
  return html`<${Card} title="Trailing returns" caption=${`Time-weighted, portfolio above ${benchName} below.${perf && perf.summary && perf.summary.last_date ? ` To ${fmt.date(perf.summary.last_date)}.` : ""}`} tight>
    <div style="display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;padding-top:4px">
      ${TRAILING.map(([k, label]) => html`<div key=${k} class="stack" style="gap:2px;padding:6px 10px;border-radius:var(--radius);background:var(--color-surface-2)">
        <span class="xs faint">${label}</span>
        <span class=${["num", "strong", fmt.polarityClass(tr[k])].join(" ")} style="font-size:var(--fs-lg);line-height:1.2">${fmt.pct(tr[k], { signed: true })}</span>
        <span class="xs muted num">${fmt.pct(bt[k], { signed: true })} <span class="faint">${benchName ? "index" : ""}</span></span>
      </div>`)}
    </div>
  <//>`;
}

function Welcome({ meta }) {
  const demo = !meta || meta.mode === "seed";
  return html`<div class="card" style="position:relative;overflow:hidden;min-height:360px">
    <div aria-hidden="true" style="position:absolute;inset:0;background:linear-gradient(135deg, var(--color-accent-soft), transparent 60%), radial-gradient(60% 80% at 100% 0%, rgba(57,135,229,.18), transparent)"></div>
    <img src="/static/assets/hero.webp" alt="" aria-hidden="true" onError=${(e) => { e.currentTarget.style.display = "none"; }}
      style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.16;pointer-events:none" />
    <div style="position:relative;padding:40px 40px 36px;max-width:640px">
      <div class="row" style="gap:8px;margin-bottom:14px"><${Badge} kind=${demo ? "demo" : "live"} dot>${demo ? "DEMO" : "LIVE"}<//><span class="xs faint">${demo ? "synthetic seed data" : "your own ledger"}</span></div>
      <h2 style="font-size:var(--fs-2xl);letter-spacing:-.02em;margin-bottom:10px">${demo ? "Nothing is held in the demo book yet." : "No holdings yet."}</h2>
      <p class="muted" style="max-width:520px;line-height:1.55">
        ${demo
          ? "Demo mode runs on a synthetic universe with deterministic prices, so every page can be explored without touching your data. Switch to live data in Settings when you are ready to load your own instruments and transactions."
          : "Add the instruments you own to the universe, then record the buys. Positions, prices, weights and the risk model appear as soon as the first transaction is in the ledger."}
      </p>
      <div class="row wrap" style="gap:8px;margin-top:20px">
        <${Button} variant="primary" icon="instruments" onClick=${() => navigate("/instruments")}>Instruments<//>
        <${Button} variant="secondary" icon="transactions" onClick=${() => navigate("/transactions?new=1")}>Record a transaction<//>
        <${Button} variant="ghost" icon="settings" onClick=${() => navigate("/settings")}>${demo ? "Switch to live data" : "Settings"}<//>
      </div>
    </div>
  </div>`;
}

/* ---------------- page ---------------- */

export default function OverviewPage({ snapshot }) {
  const meta = snapshot && snapshot.meta;
  const base = (meta && meta.base_currency) || "EUR";
  const holdings = (snapshot && snapshot.holdings) || [];
  const watchlist = (snapshot && snapshot.watchlist) || [];
  const perf = snapshot && snapshot.performance;
  const bench = snapshot && (snapshot.benchmarks || []).find((b) => b.symbol === snapshot.selected_benchmark);
  const benchName = bench ? (bench.index || bench.label || bench.symbol) : "benchmark";
  const failures = useMemo(() => failureAlerts(meta), [meta]);

  const subtitle = snapshot
    ? `${holdings.length} ${holdings.length === 1 ? "holding" : "holdings"} · ${watchlist.length} on the watchlist · vs ${benchName} · ${fmt.int(snapshot.lookback)}-day risk window · ${fmt.delayNote(meta.prices_as_of, 15)}`
    : "Value, P&L, allocation and what needs attention today.";

  if (!snapshot) {
    return html`<${Page} title="Overview" subtitle=${subtitle}>
      <${Card}><div class="faint small" style="padding:24px;text-align:center">Loading the snapshot…</div><//>
    <//>`;
  }

  if (!holdings.length) {
    return html`<${Page} title="Overview" subtitle=${subtitle}>
      <div class="stack gap-4">
        <${Notice} alerts=${failures} onNavigate=${navigate} />
        <${Welcome} meta=${meta} />
        ${watchlist.length ? html`<${Card} title="Watchlist" caption="Priced instruments in the universe that you do not hold." tight>
          <div class="stack" style="gap:4px">${watchlist.map((w) => html`<a key=${w.isin} href=${`#/holdings/${w.isin}`} class="row small" style="gap:10px;padding:4px 0;color:inherit" onClick=${(e) => { e.preventDefault(); navigate(`/holdings/${w.isin}`); }}>
            <span class="grow truncate">${w.name}</span><span class="faint xs">${w.symbol || w.isin}</span>
            <span class="num" style="width:90px;text-align:right">${fmt.money(w.price, w.price_currency)}</span>
            <span class=${["num", fmt.polarityClass(w.day_change_pct)].join(" ")} style="width:60px;text-align:right">${fmt.pct(w.day_change_pct, { signed: true })}</span>
          </a>`)}</div>
        <//>` : null}
      </div>
    <//>`;
  }

  return html`<${Page} title="Overview" subtitle=${subtitle}>
    <div class="stack gap-4">
      <${Notice} alerts=${failures} summary=${failures.length ? `${failures.length} price ${failures.length === 1 ? "fetch" : "fetches"} failed — the figures below exclude ${failures.length === 1 ? "that instrument" : "those instruments"}` : undefined} onNavigate=${navigate} />
      <${KpiRow} totals=${snapshot.totals} meta=${meta} base=${base} />
      <div class="grid">
        <${ValueChart} perf=${perf} benchmarks=${snapshot.benchmarks} selectedBenchmark=${snapshot.selected_benchmark} base=${base} />
        <${AllocationTreemap} holdings=${holdings} base=${base} />
      </div>
      <div class="grid">
        <${RiskCard} risk=${snapshot.risk} />
        <${AlertsCard} alerts=${snapshot.alerts} />
        <${MoversCard} holdings=${holdings} watchlist=${watchlist} />
      </div>
      <${TrailingStrip} perf=${perf} benchName=${benchName} />
    </div>
  <//>`;
}
