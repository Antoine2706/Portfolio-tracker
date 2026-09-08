/* Performance — "how has it done, against what". Props: { route, snapshot }.

   Layout order, and why:
   1. What qualifies the numbers (performance.warnings, performance.missing)
      before any number.
   2. The figures: total return (TWR) with annualised and money-weighted next
      to it, because the two returns answer different questions and showing
      one without the other invites the wrong conclusion; then volatility,
      Sharpe and the max drawdown.
   3. Growth against the benchmark, both indexed to 100 (one axis, never two),
      with the trailing returns table beside it.
   4. Drawdown and what each holding contributed.
   5. The monthly heatmap with a calendar-year table.
   6. Rolling volatility and rolling beta as two small multiples on one
      canvas; per-holding value as a stacked area.
   7. Everything else from the summary, with its plain-English label.
   The client computes nothing beyond slicing, re-basing to 100, folding
   beyond eight series into Other, and the portfolio − benchmark difference. */

import { html, useMemo, useState, useRef, useLayoutEffect } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { KpiTile } from "/static/components/KpiTile.js";
import { Chart } from "/static/components/Chart.js";
import { DataTable, colorCell } from "/static/components/DataTable.js";
import { Notice } from "/static/components/Notice.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Button } from "/static/components/Button.js";
import { Segmented, RangeSelector, rangeStartIndex } from "/static/components/Segmented.js";
import { navigate } from "/static/lib/router.js";
import { lineOption, drawdownOption, monthlyHeatmapOption, divergingBarOption, smallMultiplesOption, seriesTable, matrixTable, categoryTable, tipElement, seriesColor, foldOther } from "/static/lib/charts.js";
import { tokens } from "/static/lib/theme.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- page-local helpers ---------------- */

/** First finite value at or after `from`. */
function firstFinite(values, from = 0) {
  for (let i = from; i < values.length; i++) if (values[i] != null && Number.isFinite(values[i])) return { i, v: values[i] };
  return null;
}

/** Re-base a series to 100 at its first finite value. */
function rebase(values) {
  const f = firstFinite(values);
  if (!f || !f.v) return values.map(() => null);
  return values.map((v) => (v == null ? null : (v / f.v) * 100));
}

/** performance.warnings / performance.missing → graded Notice items. */
function perfAlerts(perf, holdings) {
  const out = [];
  for (const isin of perf.missing || []) {
    const h = (holdings || []).find((x) => x.isin === isin);
    out.push({ code: `missing:${isin}`, severity: "SERIOUS", title: `${h ? (h.short_name || h.name) : isin}: no price history`, detail: "Excluded from every figure on this page.", isins: [isin], route: `#/instruments/${isin}` });
  }
  (perf.warnings || []).forEach((w, i) => out.push({ code: `warn:${i}`, severity: "WARNING", title: w, detail: "", isins: [], route: "" }));
  return out;
}

/** Container width, so cell labels can be hidden when they would not fit. */
function useWidth(ref) {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ro = new ResizeObserver((entries) => { for (const e of entries) setW(Math.round(e.contentRect.width)); });
    ro.observe(el);
    setW(Math.round(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, []);
  return w;
}

const TRAILING = [["1w", "1 week"], ["1m", "1 month"], ["3m", "3 months"], ["6m", "6 months"], ["ytd", "Year to date"], ["1y", "1 year"], ["all", "Since start"]];

/* ---------------- sections ---------------- */

function KpiStrip({ s }) {
  return html`<div class="perf-kpis">
    <${KpiTile} label="Total return" value=${fmt.pct(s.total_return, { signed: true })} polarity=${fmt.polarityClass(s.total_return)} sub="time-weighted (TWR)"
      help="What the holdings did, with the effect of deposits and withdrawals removed. Comparable to a benchmark." />
    <${KpiTile} label="Annualised" value=${fmt.pct(s.annualised_return, { signed: true })} polarity=${fmt.polarityClass(s.annualised_return)} sub="geometric, 252 days a year"
      help="The total return converted to a yearly rate; over a short history it extrapolates." />
    <${KpiTile} label="Money-weighted" value=${fmt.pct(s.money_weighted, { signed: true })} polarity=${fmt.polarityClass(s.money_weighted)} sub="IRR of your cash flows"
      help="What your money did, timing included: the internal rate of return of every flow against today's value. Not comparable to a benchmark, but it is the return you experienced." />
    <${KpiTile} label="Volatility" value=${fmt.pct(s.volatility)} sub="annualised" help="Standard deviation of daily returns, scaled to a year. A broad European equity index sits near 15%." />
    <${KpiTile} label="Sharpe" value=${fmt.num(s.sharpe)} sub="return per unit of risk" help="Annualised mean return divided by volatility, risk-free rate 0. Above 1 is good; negative means the period lost money." />
    <${KpiTile} label="Max drawdown" value=${fmt.pct(s.max_drawdown)} polarity=${fmt.polarityClass(s.max_drawdown)}
      sub=${s.current_drawdown != null ? `now ${fmt.pct(s.current_drawdown)} from the peak` : "current drawdown —"}
      help="The worst peak-to-trough fall of the return index. The caption is where the index sits now against its highest point." />
  </div>`;
}

function GrowthChart({ perf, benchName, base }) {
  const [range, setRange] = useState("ALL");
  const slice = useMemo(() => {
    const start = rangeStartIndex(perf.dates, range);
    return { dates: perf.dates.slice(start), index: rebase(perf.index.slice(start)), bench: rebase((perf.benchmark_index || []).slice(start)), start };
  }, [perf, range]);
  const table = useMemo(() => seriesTable({ dates: slice.dates, series: [{ name: "Portfolio", values: slice.index }, { name: benchName, values: slice.bench }], format: "index" }), [slice, benchName]);
  const build = () => {
    const t = tokens();
    const opt = lineOption({
      dates: slice.dates, yFormat: "index", currency: base, baseline: 100, endLabels: true,
      series: [{ name: "Portfolio", values: slice.index, color: t.series[0] }, { name: benchName, values: slice.bench, color: t.series[1], width: 1.5 }],
    });
    opt.tooltip.formatter = (params) => {
      const list = Array.isArray(params) ? params : [params];
      if (!list.length) return "";
      const rows = list.map((p) => ({ name: p.seriesName, color: p.color, kind: "line", value: fmt.num(p.value, { decimals: 1 }) }));
      const p = list.find((x) => x.seriesIndex === 0), b = list.find((x) => x.seriesIndex === 1);
      const foot = p && b && p.value != null && b.value != null ? `Spread ${fmt.pp((p.value - b.value) / 100)} · since ${fmt.date(slice.dates[0])}` : null;
      return tipElement(fmt.date(list[0].axisValue), rows, foot);
    };
    return opt;
  };
  const caption = `Both re-based to 100 at ${fmt.date(slice.dates[0])}${range === "ALL" ? "" : ", the start of the selected range"}. Time-weighted, so deposits and withdrawals do not move it.`;
  return html`<${Chart} class="col-8" title=${`Growth of 100 vs ${benchName}`} caption=${caption} height=${300}
    actions=${html`<${RangeSelector} value=${range} onChange=${setRange} />`}
    option=${build} deps=${[slice, benchName, base]} table=${table} empty="No return history" />`;
}

function TrailingTable({ perf, benchName }) {
  const tr = perf.trailing || {}, bt = perf.benchmark_trailing || {};
  const rows = useMemo(() => TRAILING.map(([k, label]) => ({ id: k, period: label, p: tr[k] ?? null, b: bt[k] ?? null, d: tr[k] != null && bt[k] != null ? tr[k] - bt[k] : null })), [perf]);
  const columns = useMemo(() => [
    { key: "period", label: "Period", sortable: false },
    { key: "p", label: "Portfolio", numeric: true, sortable: false, render: colorCell("pct-signed") },
    { key: "b", label: "Index", numeric: true, sortable: false, render: colorCell("pct-signed"), title: benchName },
    { key: "d", label: "Diff", numeric: true, sortable: false, render: colorCell("pp"), title: "Portfolio minus benchmark, percentage points" },
  ], [benchName]);
  return html`<${Card} class="col-4 perf-periods" title="Trailing returns" caption=${`Time-weighted, to ${fmt.date(perf.summary && perf.summary.last_date)}. Index is ${benchName}.`} flush>
    <${DataTable} columns=${columns} rows=${rows} density="compact" caption="Trailing returns against the benchmark" />
  <//>`;
}

function DrawdownChart({ perf, benchName, s }) {
  const table = useMemo(() => seriesTable({ dates: perf.dates, series: [{ name: "Portfolio", values: perf.drawdown }, { name: benchName, values: perf.benchmark_drawdown }], format: "pct" }), [perf, benchName]);
  const caption = `Fall from the running peak. Worst ${fmt.pct(s.max_drawdown)}${s.current_drawdown != null ? `, now ${fmt.pct(s.current_drawdown)} below the peak` : ""}.`;
  return html`<${Chart} class="col-6" title="Drawdown" caption=${caption} height=${240}
    option=${() => drawdownOption(perf.dates, perf.drawdown, { benchmark: perf.benchmark_drawdown, benchmarkName: benchName })} deps=${[perf, benchName]} table=${table} empty="No drawdown history" />`;
}

function ContributionChart({ perf, base }) {
  const items = useMemo(() => [...(perf.contributions || [])].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)), [perf]);
  const names = items.map((c) => c.short_name || c.name);
  const values = items.map((c) => c.contribution);
  const table = useMemo(() => {
    const t = categoryTable({ categories: names, values, format: "pct-signed", label: "Holding", valueLabel: "Contribution", extra: [{ key: "pnl", label: "P&L", numeric: true, render: colorCell("money-signed", { currency: base }) }] });
    t.rows.forEach((r, i) => { r.pnl = items[i].pnl; r.id = items[i].isin; });
    t.columns[1].render = colorCell("pct-signed");
    return t;
  }, [items, base]);
  return html`<${Chart} class="col-6" title="Contribution by holding" caption="Each holding's P&L as a share of the portfolio's value at the start of the history, largest first. Hover for the amount." height=${Math.max(200, items.length * 30 + 56)}
    option=${() => divergingBarOption({ names, values, format: "pct-signed", palette: "pnl", labels: { pos: "added", neg: "cost", zero: "flat" }, sentences: items.map((c) => `P&L ${fmt.money(c.pnl, base, { signed: true })}`) })}
    deps=${[items]} table=${table} empty="No contributions yet" />`;
}

const MONTH_MODES = [{ value: "portfolio", label: "Portfolio" }, { value: "benchmark", label: "Benchmark" }, { value: "difference", label: "Difference" }];

function MonthlyHeatmap({ perf, benchName }) {
  const [mode, setMode] = useState("portfolio");
  const wrap = useRef(null);
  const width = useWidth(wrap);
  const years = useMemo(() => [...new Set((perf.monthly || []).map((m) => m.year))].sort(), [perf]);
  const cols = [...fmt.MONTHS_SHORT, "Year"];
  const pick = (m) => (mode === "portfolio" ? m.ret : mode === "benchmark" ? m.benchmark_ret : (m.ret != null && m.benchmark_ret != null ? m.ret - m.benchmark_ret : null));
  const matrix = useMemo(() => years.map((y) => {
    const row = Array.from({ length: 12 }, (_, i) => { const m = perf.monthly.find((x) => x.year === y && x.month === i + 1); return m ? pick(m) : null; });
    const yr = (perf.yearly || []).find((x) => x.year === y);
    row.push(yr ? pick(yr) : null);
    return row;
  }), [perf, years, mode]);
  const table = useMemo(() => matrixTable({ xLabels: cols, yLabels: years.map(String), matrix, format: (v) => fmt.pct(v, { signed: true }), cornerLabel: "Year" }), [matrix, years]);
  // a label needs ~44px of cell; the y axis and card padding take ~64px
  const showValues = width > 0 && (width - 64) / cols.length >= 44;
  const what = mode === "portfolio" ? "Portfolio" : mode === "benchmark" ? benchName : `Portfolio minus ${benchName}`;
  const caption = `${what}, compounded per calendar month; the last column is the calendar year. Green = gain, red = loss, neutral at zero${showValues ? "" : " — hover a cell for the value"}.`;
  return html`<div class="col-8" ref=${wrap}>
    <${Chart} title="Monthly returns" caption=${caption} height=${Math.max(140, years.length * 40 + 64)}
      actions=${html`<${Segmented} size="sm" ariaLabel="Series" value=${mode} onChange=${setMode} options=${MONTH_MODES} />`}
      option=${() => (years.length ? monthlyHeatmapOption(years, cols, matrix, { showValues }) : null)} deps=${[matrix, years, showValues]} table=${table} empty="No monthly returns yet" />
  </div>`;
}

function YearsTable({ perf, benchName }) {
  const rows = useMemo(() => (perf.yearly || []).map((y) => ({ id: y.year, year: String(y.year), p: y.ret, b: y.benchmark_ret, d: y.ret != null && y.benchmark_ret != null ? y.ret - y.benchmark_ret : null })), [perf]);
  const columns = useMemo(() => [
    { key: "year", label: "Year", sortable: false },
    { key: "p", label: "Portfolio", numeric: true, sortable: false, render: colorCell("pct-signed") },
    { key: "b", label: "Index", numeric: true, sortable: false, render: colorCell("pct-signed"), title: benchName },
    { key: "d", label: "Diff", numeric: true, sortable: false, render: colorCell("pp"), title: "Portfolio minus benchmark, percentage points" },
  ], [benchName]);
  return html`<${Card} class="col-4 perf-periods" title="Calendar years" caption="The first and last years are partial." flush>
    <${DataTable} columns=${columns} rows=${rows} density="compact" empty="No full month yet" caption="Calendar-year returns against the benchmark" />
  <//>`;
}

function RollingChart({ perf, benchName }) {
  const rv = perf.rolling_volatility || { dates: [], values: [] };
  const rb = perf.rolling_beta || { dates: [], values: [] };
  const aligned = useMemo(() => {
    const dates = [...new Set([...(rv.dates || []), ...(rb.dates || [])])].sort();
    const mv = new Map((rv.dates || []).map((d, i) => [d, rv.values[i]]));
    const mb = new Map((rb.dates || []).map((d, i) => [d, rb.values[i]]));
    return { dates, vol: dates.map((d) => mv.get(d) ?? null), beta: dates.map((d) => mb.get(d) ?? null) };
  }, [perf]);
  const table = useMemo(() => ({
    columns: [
      { key: "date", label: "Date", format: "date", width: 120 },
      { key: "vol", label: "Rolling volatility", format: "pct", numeric: true },
      { key: "beta", label: "Rolling beta", format: "num", numeric: true },
    ],
    rows: aligned.dates.map((d, i) => ({ id: d, date: d, vol: aligned.vol[i], beta: aligned.beta[i] })),
  }), [aligned]);
  const height = 320;
  const build = () => (aligned.dates.length > 1 ? smallMultiplesOption({
    dates: aligned.dates, height,
    panels: [
      { name: "Rolling volatility, 63 trading days, annualised", values: aligned.vol, yFormat: "pct" },
      { name: `Rolling beta vs ${benchName}, 63 trading days`, values: aligned.beta, yFormat: "num", markLines: [{ value: 1, label: "1.0" }] },
    ],
  }) : null);
  return html`<${Chart} class="col-6" title="Rolling risk" caption="Two panels on one time axis, never two scales. Beta of 1.0 would move exactly with the index." height=${height}
    option=${build} deps=${[aligned, benchName]} table=${table} empty="Needs 63 trading days" />`;
}

function PerHoldingChart({ perf, holdings, base }) {
  const data = useMemo(() => {
    const names = perf.holding_names || {};
    const entries = Object.entries(perf.per_holding_value || {}).map(([isin, values]) => {
      // closed positions are not in `holdings` any more; the API names them
      const h = (holdings || []).find((x) => x.isin === isin);
      const name = (h && h.name) || names[isin] || isin;
      const last = [...values].reverse().find((v) => v != null) ?? 0;
      return { isin, name, values, value: last };
    });
    const folded = foldOther(entries.map((e) => ({ ...e })), { limit: 8 });
    const series = folded.map((e) => (e.isOther
      ? { name: "Other", values: perf.dates.map((_, i) => { let s = 0, any = false; for (const m of e.members) { if (m.values[i] != null) { s += m.values[i]; any = true; } } return any ? s : null; }) }
      : { name: e.name, values: e.values }));
    return series;
  }, [perf, holdings]);
  const table = useMemo(() => seriesTable({ dates: perf.dates, series: data, format: "money", currency: base }), [data, perf, base]);
  const legend = html`<div class="perf-legend">${data.map((s, i) => html`<span key=${s.name} class="legend-item"><span class="swatch" style=${`background:${seriesColor(i)}`}></span>${s.name}</span>`)}</div>`;
  const build = () => {
    if (!data.length) return null;
    const opt = lineOption({ dates: perf.dates, series: data, yFormat: "money-compact", currency: base, stacked: true });
    opt.legend = { show: false };
    opt.grid.top = 12;
    return opt;
  };
  return html`<${Chart} class="col-6" title="Value by holding" caption="Market value of each position over time, stacked; the largest today sits at the bottom." height=${320}
    legend=${data.length > 1 ? legend : null} option=${build} deps=${[data, base]} table=${table} empty="No per-holding history" />`;
}

function SummaryCard({ s, benchName }) {
  const Row = ({ label, note, value, cls = "" }) => html`<div class="perf-def">
    <span class="perf-def-label">${label}${note ? html`<span class="perf-def-note">${note}</span>` : null}</span>
    <span class=${["perf-def-value", cls].filter(Boolean).join(" ")}>${value == null || value === "" ? fmt.DASH : value}</span>
  </div>`;
  const pol = fmt.polarityClass;
  return html`<${Card} title="Summary" caption=${`Every remaining figure from the performance summary. ${fmt.int(s.observations)} daily observations, ${fmt.dateRange(s.first_date, s.last_date)}.`}>
    <div class="perf-summary">
      <div>
        <${Row} label="Sortino ratio" note="Like Sharpe, but only moves below zero count as risk." value=${fmt.num(s.sortino)} />
        <${Row} label="Calmar ratio" note="Annualised return divided by the worst peak-to-trough fall." value=${fmt.num(s.calmar)} />
        <${Row} label="Best day" note="Largest single-day gain." value=${fmt.pct(s.best_day, { signed: true })} cls=${pol(s.best_day)} />
        <${Row} label="Worst day" note="Largest single-day loss." value=${fmt.pct(s.worst_day, { signed: true })} cls=${pol(s.worst_day)} />
        <${Row} label="Best month" note="Best calendar month, compounded." value=${fmt.pct(s.best_month, { signed: true })} cls=${pol(s.best_month)} />
        <${Row} label="Worst month" note="Worst calendar month, compounded." value=${fmt.pct(s.worst_month, { signed: true })} cls=${pol(s.worst_month)} />
        <${Row} label="Positive / negative months" note="Exactly-zero months are not counted." value=${`${fmt.int(s.positive_months)} up · ${fmt.int(s.negative_months)} down`} />
        <${Row} label="Observations" note="Trading days in the history." value=${fmt.int(s.observations)} />
        <${Row} label="First date" note="The first day with a price for every holding then held." value=${fmt.date(s.first_date)} />
        <${Row} label="Last date" value=${fmt.date(s.last_date)} />
      </div>
      <div>
        <${Row} label=${`${benchName} total return`} note="The benchmark over the same dates, so the two are comparable." value=${fmt.pct(s.benchmark_total_return, { signed: true })} cls=${pol(s.benchmark_total_return)} />
        <${Row} label="Beta" note=${`How much the portfolio has moved for every 1% move of ${benchName}.`} value=${fmt.num(s.beta)} />
        <${Row} label="Alpha, annualised" note="The return not explained by the benchmark: the intercept of the daily regression, times 252." value=${fmt.pct(s.alpha, { signed: true })} cls=${pol(s.alpha)} />
        <${Row} label="Correlation" note="How closely daily moves track the benchmark, from −1 to 1." value=${fmt.num(s.correlation)} />
        <${Row} label="Tracking error" note="Annualised standard deviation of the daily difference to the benchmark: how far the portfolio strays." value=${fmt.pct(s.tracking_error)} />
        <${Row} label="Information ratio" note="Average outperformance per unit of tracking error. Negative means the straying cost money." value=${fmt.num(s.information_ratio)} cls=${pol(s.information_ratio)} />
      </div>
    </div>
  <//>`;
}

/* ---------------- page ---------------- */

export default function PerformancePage({ snapshot }) {
  const meta = snapshot && snapshot.meta;
  const base = (meta && meta.base_currency) || "EUR";
  const perf = snapshot && snapshot.performance;
  const holdings = (snapshot && snapshot.holdings) || [];
  const bench = snapshot && (snapshot.benchmarks || []).find((b) => b.symbol === snapshot.selected_benchmark);
  const benchName = bench ? (bench.index || bench.label || bench.symbol) : "Benchmark";
  const alerts = useMemo(() => (perf ? perfAlerts(perf, holdings) : []), [perf, holdings]);

  if (!snapshot) {
    return html`<${Page} title="Performance" subtitle="Time-weighted return, drawdowns, monthly returns and contributions against the benchmark.">
      <${Card}><div class="faint small" style="padding:24px;text-align:center">Loading the snapshot…</div><//>
    <//>`;
  }

  const ok = perf && perf.available && perf.dates && perf.dates.length > 1 && perf.summary;
  const s = ok ? perf.summary : null;
  const subtitle = ok
    ? `Time-weighted against ${benchName} · ${fmt.dateRange(s.first_date, s.last_date)} · ${fmt.count(s.observations, "trading day")} · ${fmt.delayNote(meta.prices_as_of, 15)}`
    : "Time-weighted return, drawdowns, monthly returns and contributions against the benchmark.";

  if (!ok) {
    return html`<${Page} title="Performance" subtitle=${subtitle}>
      <div class="stack gap-4">
        <${Notice} alerts=${alerts} onNavigate=${navigate} />
        <${Card}><${EmptyState} icon="performance" title="No return history yet"
          body=${(perf && perf.reason) || "Performance needs at least two priced days of holdings. Record a transaction and the history builds from the first price after it."}
          action=${html`<${Button} variant="primary" icon="transactions" onClick=${() => navigate("/transactions")}>Transactions<//>`} /><//>
      </div>
    <//>`;
  }

  return html`<${Page} title="Performance" subtitle=${subtitle}>
    <div class="stack gap-4">
      <${Notice} alerts=${alerts} onNavigate=${navigate}
        summary=${alerts.length ? `${fmt.count(alerts.length, "note")} on the history behind these figures` : undefined} />
      <${KpiStrip} s=${s} />
      <div class="grid">
        <${GrowthChart} perf=${perf} benchName=${benchName} base=${base} />
        <${TrailingTable} perf=${perf} benchName=${benchName} />
      </div>
      <div class="grid">
        <${DrawdownChart} perf=${perf} benchName=${benchName} s=${s} />
        <${ContributionChart} perf=${perf} base=${base} />
      </div>
      <div class="grid">
        <${MonthlyHeatmap} perf=${perf} benchName=${benchName} />
        <${YearsTable} perf=${perf} benchName=${benchName} />
      </div>
      <div class="grid">
        <${RollingChart} perf=${perf} benchName=${benchName} />
        <${PerHoldingChart} perf=${perf} holdings=${holdings} base=${base} />
      </div>
      <${SummaryCard} s=${s} benchName=${benchName} />
    </div>
  <//>`;
}
