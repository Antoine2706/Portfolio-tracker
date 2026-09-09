/* Risk — the product. "Where does the risk actually come from?" Props: { route, snapshot }.

   Layout order, and why:
   1. The headline: one sentence naming the holding whose share of the risk
      sits furthest from its share of the money, and the window it was
      measured over. Everything else on the page is supporting evidence.
   2. What qualifies the figures (window shortened, instruments excluded,
      correlated pairs, clusters) as one graded notice.
   3. The divergence chart with its table, beside the metrics — effective
      holdings first, because it is the only measure that sees a cluster of
      holdings all making the same bet.
   4. Standalone volatility beside the correlation grid: what each holding
      is on its own, and how they move together.
   5. Value at risk and the distribution of daily returns it is read from.
   6. The worst periods on record and the stress scenarios.
   Nothing is computed here beyond formatting, sorting and binning the daily
   returns for the histogram. */

import { html, useMemo } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { Chart } from "/static/components/Chart.js";
import { DataTable, colorCell } from "/static/components/DataTable.js";
import { Notice } from "/static/components/Notice.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Button } from "/static/components/Button.js";
import { Icon } from "/static/components/Icons.js";
import { Help } from "/static/components/Tooltip.js";
import { useStore } from "/static/lib/store.js";
import { navigate, href } from "/static/lib/router.js";
import { divergingBarOption, barOption, heatmapOption, histogramOption, categoryTable, matrixTable } from "/static/lib/charts.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- page-local helpers ---------------- */

const BROAD_INDEX_VOL = 0.15;

/** Everything that qualifies the figures, as graded notice items. */
function riskAlerts(risk) {
  const out = [];
  const w = risk.window;
  if (w) {
    if (w.effective < w.requested && w.binding_isin) {
      out.push({ code: "window", severity: "INFO", title: `Window shortened to ${fmt.int(w.effective)} returns from the ${fmt.int(w.requested)} requested`, detail: `Constrained by ${w.binding_name || w.binding_isin}, which has only ${fmt.int(w.binding_observations)} observations. The same window is used for every instrument — window lengths are never mixed.`, isins: [w.binding_isin], route: `#/holdings/${w.binding_isin}` });
    }
    for (const e of w.excluded || []) {
      out.push({ code: `excluded:${e.isin}`, severity: "WARNING", title: `${e.name || e.isin} is not in the risk model`, detail: e.reason, isins: [e.isin], route: `#/holdings/${e.isin}` });
    }
    for (const note of w.warnings || []) {
      if (/^Window shortened|excluded:/.test(note)) continue;   // already itemised above
      out.push({ code: `warn:${note.slice(0, 32)}`, severity: "INFO", title: note, detail: "", isins: [], route: "" });
    }
  }
  for (const c of risk.clusters || []) {
    if (c.members.length < 2) continue;
    out.push({ code: `cluster:${c.members.join(",")}`, severity: c.members.length >= 3 ? "SERIOUS" : "WARNING", title: `${c.members.length} holdings move as one bet: ${(c.short_names || c.names).join(", ")}`, detail: `Mean correlation ${fmt.num(c.mean_correlation)}, lowest pair ${fmt.num(c.min_correlation)}${c.combined_weight != null ? `, ${fmt.pct(c.combined_weight)} of capital between them` : ""}. Pairwise comparison cannot see a group; this is why effective holdings leads.`, isins: c.members, route: "#/simulator" });
  }
  for (const p of risk.pairs || []) {
    out.push({ code: `pair:${p.a}:${p.b}`, severity: "WARNING", title: `${p.a_short || p.a_name} and ${p.b_short || p.b_name} move together ${fmt.pct(p.correlation, { decimals: 0 })} of the time`, detail: p.sentence, isins: [p.a, p.b], route: `#/holdings/${p.a}` });
  }
  return out;
}

/** Equal-width bins over [min, max]; a value on the top edge lands in the last bin. */
function binReturns(values, count = 30) {
  const xs = (values || []).filter((v) => v != null && Number.isFinite(v));
  if (xs.length < 2) return [];
  let lo = Math.min(...xs), hi = Math.max(...xs);
  if (lo === hi) { lo -= 0.001; hi += 0.001; }
  const width = (hi - lo) / count;
  const bins = Array.from({ length: count }, (_, i) => ({ x0: lo + i * width, x1: lo + (i + 1) * width, count: 0 }));
  for (const v of xs) bins[Math.min(count - 1, Math.floor((v - lo) / width))].count += 1;
  return bins;
}

const METHOD_LABEL = { historical: "Historical", parametric: "Parametric", cornish_fisher: "Cornish-Fisher" };

function go(path) {
  return (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button === 1) return;
    e.preventDefault();
    navigate(path);
  };
}

/* ---------------- headline ---------------- */

function Headline({ risk }) {
  const head = risk.headline;
  const w = risk.window;
  const windowText = w
    ? `${fmt.int(w.effective)} daily returns, ${fmt.dateRange(w.first_date, w.last_date)}${w.effective < w.requested && w.binding_name ? ` — short of the ${fmt.int(w.requested)} requested, constrained by ${w.binding_name} (${fmt.int(w.binding_observations)} observations)` : ""}${w.excluded && w.excluded.length ? `; ${fmt.count(w.excluded.length, "instrument")} excluded` : ""}.`
    : "";
  return html`<${Card}>
    <div class="risk-headline">
      <div>
        <p class="risk-headline-sentence">${head ? head.sentence : "Risk and capital are in line across the book."}</p>
        <p class="risk-headline-window">${windowText} Every figure below is measured over this one window, so a volatility computed over 300 days and one computed over 252 are never mixed.</p>
      </div>
      ${head ? html`<div class="risk-headline-metric">
        <span class="caps">Largest gap between capital and risk</span>
        <span class=${["risk-headline-value", fmt.divergenceClass(head.divergence)].join(" ")}>${fmt.pp(head.divergence)}</span>
        <span class="risk-headline-sub" title=${head.name}>${head.short_name || head.name} · ${fmt.pct(head.weight)} of the money, ${fmt.pct(head.risk_share)} of the risk</span>
      </div>` : null}
    </div>
  <//>`;
}

/* ---------------- divergence ---------------- */

function DivergenceChart({ risk }) {
  const theme = useStore((s) => s.theme);
  const rows = risk.divergence || [];
  const names = rows.map((r) => r.short_name || r.name);
  const values = rows.map((r) => r.divergence);
  const table = useMemo(() => {
    // valueClass makes the chart's built-in table twin carry the same ink as
    // the bars. Without it this was a third rendering of the same quantity,
    // uncoloured, one keystroke away from the other two.
    const t = categoryTable({ categories: names, values, format: "pp", label: "Holding", valueLabel: "Divergence", valueClass: fmt.divergenceClass });
    t.rows.forEach((r, i) => { r.id = rows[i].isin; });
    return t;
  }, [rows]);
  return html`<${Chart} title="Capital share against risk share"
    caption="How far each holding's share of the risk sits from its share of the money. Warm bars carry more risk than capital, cool bars less; bars at zero are behaving as expected and have earned no attention."
    height=${Math.max(200, rows.length * 34 + 40)} deps=${[rows, theme]} table=${table} empty="No decomposition"
    onEvents=${{ click: (p) => { const r = rows[p.dataIndex]; if (r) navigate(`/holdings/${r.isin}`); } }}
    buildOption=${() => (rows.length ? divergingBarOption({ names, values, sentences: rows.map((r) => r.sentence) }) : null)} />`;
}

function DivergenceTable({ risk }) {
  const rows = useMemo(() => (risk.divergence || []).map((r) => ({ ...r, id: r.isin })), [risk]);
  const columns = useMemo(() => [
    // The label form, with the registered name on the tooltip. This column
    // was the worst offender: "iShares MSCI Europe Industrials Sector ..."
    // spent its width on the four characters every row shared.
    { key: "name", label: "Instrument", primary: true, render: (r, v) => html`<div class="truncate" style="max-width:240px" title=${v}>${r.short_name || v}<span class="cell-sub">${r.isin}</span></div>` },
    { key: "weight", label: "Capital weight", format: "pct", numeric: true, title: "Share of portfolio value" },
    { key: "risk_share", label: "Risk share", format: "pct", numeric: true, title: "Share of portfolio volatility this holding contributes" },
    { key: "divergence", label: "Divergence", numeric: true, title: "Risk share minus weight, in percentage points",
      render: (r, v) => html`<span class=${fmt.divergenceClass(v)}>${fmt.pp(v)}</span>` },
    { key: "marginal", label: "Marginal", format: "pct", numeric: true, title: "How much annualised portfolio volatility moves per extra unit of weight here (MCTR)" },
  ], []);
  return html`<${Card} title="Risk contribution" caption="Sorted by the size of the gap. Marginal is what one more euro here would do to portfolio volatility — the number the simulator works from." flush>
    <${DataTable} columns=${columns} rows=${rows} rowKey="id" density="compact" caption="Risk contribution by holding" onRowClick=${(r) => navigate(`/holdings/${r.isin}`)} empty="No decomposition" />
  <//>`;
}

/* ---------------- metrics ---------------- */

function MetricCard({ m, lead }) {
  return html`<div class=${["card", "risk-metric", lead ? "lead" : ""].filter(Boolean).join(" ")}>
    <span class="risk-metric-label">${m.label}</span>
    <span class="risk-metric-value">${m.display || fmt.DASH}</span>
    <span class="risk-metric-sentence">${m.sentence}</span>
    ${m.warning ? html`<span class="risk-metric-warning"><${Icon} name="alertTriangle" size=${12} stroke=${2} />${m.warning}</span>` : null}
  </div>`;
}

function Metrics({ risk }) {
  const metrics = risk.metrics || [];
  const lead = metrics.find((m) => m.key === "effective_holdings");
  const rest = metrics.filter((m) => m !== lead);
  return html`<div class="stack gap-3">
    <div class="risk-metrics">
      ${lead ? html`<${MetricCard} m=${lead} lead />` : null}
      ${rest.map((m) => html`<${MetricCard} key=${m.key} m=${m} />`)}
    </div>
    <p class="xs faint">Effective holdings leads because it is the only reading that sees a cluster of holdings all making the same bet. The correlation grid below reports one pair at a time and structurally cannot.</p>
  </div>`;
}

/* ---------------- standalone volatility ---------------- */

function StandaloneChart({ risk }) {
  const theme = useStore((s) => s.theme);
  const rows = risk.standalone || [];
  const names = rows.map((r) => r.short_name || r.name);
  const values = rows.map((r) => r.volatility);
  const table = useMemo(() => {
    const t = categoryTable({ categories: names, values, format: "pct", label: "Holding", valueLabel: "Annualised volatility",
      extra: [{ key: "multiple", label: "× broad index", numeric: true, format: "mult" }, { key: "weight", label: "Weight", numeric: true, format: "pct" }] });
    t.rows.forEach((r, i) => { r.id = rows[i].isin; r.multiple = rows[i].multiple; r.weight = rows[i].weight; });
    return t;
  }, [rows]);
  return html`<${Chart} class="col-6" title="How volatile each holding is on its own"
    caption=${`Annualised standalone volatility over the window. A broad European equity index typically sits near ${fmt.pct(BROAD_INDEX_VOL, { decimals: 0 })}; most thematic ETFs run two to three times that, and that is what the concentration costs.`}
    height=${Math.max(200, rows.length * 30 + 56)} deps=${[rows, theme]} table=${table} empty="No standalone volatilities"
    onEvents=${{ click: (p) => { const r = rows[p.dataIndex]; if (r) navigate(`/holdings/${r.isin}`); } }}
    buildOption=${() => (rows.length ? barOption({
      categories: names, values, horizontal: true, format: "pct", showLabels: true,
      markLines: [{ value: BROAD_INDEX_VOL, label: "broad index" }],
      tipRows: (p) => { const r = rows[p.dataIndex]; return r ? [{ name: "times a broad index", value: fmt.mult(r.multiple) }, { name: "weight", value: fmt.pct(r.weight) }] : []; },
    }) : null)} />`;
}

/* ---------------- correlation ---------------- */

function CorrelationCard({ risk, symbols }) {
  const theme = useStore((s) => s.theme);
  const corr = risk.correlation;
  const labels = corr ? corr.names : [];
  // Tickers on BOTH axes. A correlation matrix is symmetric, so labelling the
  // rows one way and the columns another invites the reader to look for a
  // difference that is not there; and a 44px cell cannot hold a fund name in
  // either direction. Full names live in the tooltip and in the table twin.
  const codes = corr ? corr.isins.map((isin, i) => {
    const s = symbols && symbols[isin];
    return s ? String(s).split(".")[0] : (corr.short_names || corr.names)[i];
  }) : [];

  // The colour domain is symmetric about zero but scaled to THIS grid, not to
  // the theoretical [-1, 1]. Real holdings sit in a band -- 0.15 to 0.81 is
  // typical for a diversified book -- and a scale stretched to the full
  // theoretical range spends almost all of its contrast on values that never
  // occur, leaving every real cell the same muted colour. Zero stays at the
  // neutral midpoint, so negative correlation still reads cool and its
  // absence stays visible; only the extent moves.
  //
  // The diagonal is excluded from the extent as well as from the display: it
  // is 1.00 by definition, and letting it set the domain would restore
  // exactly the flattening this is fixing.
  const extent = useMemo(() => {
    if (!corr) return 1;
    let m = 0;
    for (let y = 0; y < corr.matrix.length; y++) {
      for (let x = 0; x < corr.matrix[y].length; x++) {
        if (x === y) continue;
        const v = corr.matrix[y][x];
        if (v != null) m = Math.max(m, Math.abs(v));
      }
    }
    // A floor stops a grid of near-zero correlations from being amplified
    // into a dramatic-looking picture of nothing.
    return Math.max(0.1, m);
  }, [corr]);

  // The table twin has room the grid does not, so it names the rows properly
  // and keeps the tickers only where the columns are narrow.
  const table = useMemo(() => (corr ? matrixTable({ xLabels: codes, yLabels: (corr.short_names || corr.names), matrix: corr.matrix, cornerLabel: "" }) : null), [corr]);
  const pairs = risk.pairs || [];
  const clusters = (risk.clusters || []).filter((c) => c.members.length >= 2);
  const footer = html`<div class="risk-corr-foot">
    ${clusters.length ? clusters.map((c) => html`<div key=${c.members.join()} class="risk-cluster">
      <span class="caps">Cluster</span>
      ${c.names.map((n, i) => html`<a key=${c.members[i]} class="risk-chip" href=${href(`/holdings/${c.members[i]}`)} onClick=${go(`/holdings/${c.members[i]}`)} title=${n}>${(c.short_names || c.names)[i]}</a>`)}
      <span>mean ρ ${fmt.num(c.mean_correlation)}${c.combined_weight != null ? ` · ${fmt.pct(c.combined_weight)} of capital` : ""}</span>
    </div>`) : null}
    ${!pairs.length && !clusters.length ? html`<div class="risk-quiet"><${Icon} name="checkCircle" size=${14} />No pair above ${fmt.num(risk.threshold, { decimals: 2 })}. Nothing here is two tickers on one bet.</div>` : null}
    ${pairs.length && !clusters.length ? html`<div class="risk-quiet"><${Icon} name="alertTriangle" size=${14} />${fmt.count(pairs.length, "pair")} above ${fmt.num(risk.threshold, { decimals: 2 })} — listed under the notice at the top of the page.</div>` : null}
  </div>`;
  return html`<${Chart} class="col-6" title="Correlation"
    caption=${`Pairwise, over the window. The scale stays centred on zero — nothing on the cool side means nothing here hedges anything else, and that is itself the finding — but it runs to ±${fmt.num(extent, { decimals: 2 })}, the strongest pair in this grid, so the values you actually have get the full range of colour. The diagonal is greyed: it is 1.00 by definition and says nothing. Pairs above ${fmt.num(risk.threshold, { decimals: 2 })} are flagged.`}
    height=${corr ? Math.max(260, corr.isins.length * 34 + 90) : 260} deps=${[corr, theme, extent]} table=${table} empty="No correlation matrix" footer=${footer}
    buildOption=${() => (corr ? heatmapOption({ xLabels: codes, yLabels: codes, tipLabels: labels, tipYLabels: labels, matrix: corr.matrix, min: -extent, max: extent, mask: (x, y) => x === y, showValues: corr.isins.length <= 10 }) : null)} />`;
}

/* ---------------- value at risk ---------------- */

function VarCard({ risk, base, total }) {
  const rows = useMemo(() => [...(risk.var || [])].sort((a, b) => a.horizon_days - b.horizon_days || a.confidence - b.confidence || (a.method > b.method ? 1 : -1))
    .map((v, i) => ({ ...v, id: `${v.method}-${v.confidence}-${v.horizon_days}-${i}` })), [risk]);
  const columns = useMemo(() => [
    { key: "horizon_days", label: "Horizon", numeric: true, render: (r, v) => `${fmt.int(v)} ${v === 1 ? "day" : "days"}` },
    { key: "confidence", label: "Confidence", numeric: true, render: (r, v) => fmt.pct(v, { decimals: 0 }) },
    { key: "method", label: "Method", render: (r, v) => html`<span class="method">${METHOD_LABEL[v] || v}</span>` },
    // percentage on top, the amount in base currency underneath: seven
    // columns of figures do not fit a half-width card, five two-line ones do
    { key: "loss", label: "VaR", numeric: true, title: "The loss not exceeded with this confidence",
      render: (r, v) => html`<span class="neg">${fmt.pct(v)}</span><span class="cell-sub">${fmt.money(r.loss_amount, base)}</span>` },
    { key: "expected_shortfall", label: "Expected shortfall", numeric: true, title: "The average loss on the days beyond the VaR",
      render: (r, v) => html`<span class="neg">${fmt.pct(v)}</span><span class="cell-sub">${fmt.money(r.expected_shortfall_amount, base)}</span>` },
  ], [base]);
  const obs = rows.length ? rows[0].observations : null;
  return html`<${Card} class="col-6 risk-table" title="Value at risk" caption=${`Loss thresholds read from ${fmt.int(obs)} daily returns${total ? `, on ${fmt.money(total, base)} of priced holdings` : ""}. Ten-day figures scale the one-day ones by √10.`} flush>
    <${DataTable} columns=${columns} rows=${rows} rowKey="id" density="compact" caption="Value at risk by method, confidence and horizon" empty="No value at risk" />
    <div class="risk-note">
      <strong>Historical</strong> is what the worst days of the window actually were: no distribution assumed, so it cannot see a loss the window never contained.${" "}
      <strong>Parametric</strong> fits a normal distribution to the window's mean and volatility — smooth, and known to understate fat tails.${" "}
      <strong>Cornish-Fisher</strong> is parametric adjusted for the window's skew and kurtosis, so a fat left tail widens the loss. When the three disagree, the disagreement is the information.
    </div>
  <//>`;
}

function ReturnsHistogram({ risk }) {
  const theme = useStore((s) => s.theme);
  const values = (risk.portfolio_returns && risk.portfolio_returns.values) || [];
  const bins = useMemo(() => binReturns(values, 30), [values]);
  const marks = useMemo(() => (risk.var || []).filter((v) => v.method === "historical" && v.horizon_days === 1)
    .map((v) => ({ value: -v.loss, label: `VaR ${fmt.pct(v.confidence, { decimals: 0 })}` })), [risk]);
  const table = useMemo(() => ({
    columns: [
      { key: "x0", label: "From", format: "pct2", numeric: true },
      { key: "x1", label: "To", format: "pct2", numeric: true },
      { key: "count", label: "Days", format: "int", numeric: true },
    ],
    rows: bins.map((b, i) => ({ id: i, ...b })),
  }), [bins]);
  return html`<${Chart} class="col-6" title="Daily returns of the portfolio"
    caption=${`Distribution of the ${fmt.int(values.length)} daily returns over the window, in thirty bins. The rules mark the one-day historical VaR at 95% and 99%; everything left of a rule is the tail it summarises.`}
    height=${260} deps=${[bins, marks, theme]} table=${table} empty="No daily returns"
    buildOption=${() => (bins.length ? histogramOption({ bins, format: "pct", markLines: marks }) : null)} />`;
}

/* ---------------- worst periods, scenarios ---------------- */

function WorstPeriods({ risk, base }) {
  const rows = useMemo(() => [...(risk.worst_periods || [])].sort((a, b) => a.length_days - b.length_days || a.ret - b.ret)
    .map((p, i) => ({ ...p, id: `${p.length_days}-${p.start}-${i}` })), [risk]);
  const columns = useMemo(() => [
    { key: "length_days", label: "Length", numeric: true, render: (r, v) => `${fmt.int(v)} ${v === 1 ? "day" : "days"}` },
    { key: "start", label: "From", format: "date", width: 110 },
    { key: "end", label: "To", format: "date", width: 110 },
    { key: "ret", label: "Return", numeric: true, render: colorCell("pct-signed") },
    { key: "amount", label: base, numeric: true, title: "On today's holdings", render: colorCell("money-signed", { currency: base }) },
  ], [base]);
  return html`<${Card} class="col-6 risk-table" title="Worst periods on record" caption="The three worst days, weeks (5 trading days) and months (21 trading days) in the window, applied to today's portfolio value." flush>
    <${DataTable} columns=${columns} rows=${rows} rowKey="id" density="compact" caption="Worst periods" empty="No history" />
  <//>`;
}

function ScenarioBar({ value }) {
  const v = value == null ? 0 : Math.max(-1, Math.min(1, value / 0.25));
  const w = Math.round(Math.abs(v) * 50);
  return html`<span class="risk-bar" aria-hidden="true"><span class=${["risk-bar-fill", v < 0 ? "neg" : "pos"].join(" ")} style=${v < 0 ? `left:${50 - w}%;width:${w}%` : `left:50%;width:${w}%`}></span></span>`;
}

function Scenarios({ risk, base }) {
  const rows = useMemo(() => (risk.scenarios || []).map((s) => ({ ...s, id: s.key })), [risk]);
  const columns = useMemo(() => [
    { key: "label", label: "Scenario", sortable: false, class: "risk-scen-cell", render: (r, v) => html`<span class="risk-scen-label">${v}</span><span class="risk-scen-desc">${r.description}</span>` },
    { key: "portfolio_return", label: "Portfolio", numeric: true, class: "risk-scen-cell", render: (r, v) => html`<${ScenarioBar} value=${v} /><span class=${fmt.polarityClass(v)}>${fmt.pct(v, { signed: true })}</span>` },
    { key: "portfolio_amount", label: base, numeric: true, class: "risk-scen-cell", render: colorCell("money-signed", { currency: base }) },
  ], [base]);
  return html`<${Card} class="col-6 risk-table" title="Stress scenarios" caption="What today's book would do if the window's worst stretches repeated, and under a few hypothetical shocks. The bar is scaled to ±25%." flush>
    <${DataTable} columns=${columns} rows=${rows} rowKey="id" density="compact" caption="Stress scenarios" empty="No scenarios" />
  <//>`;
}

/* ---------------- page ---------------- */

export default function RiskPage({ snapshot }) {
  const risk = snapshot && snapshot.risk;
  const meta = snapshot && snapshot.meta;
  const base = (meta && meta.base_currency) || "EUR";
  const total = snapshot && snapshot.totals ? snapshot.totals.value : null;
  const alerts = useMemo(() => (risk && risk.available ? riskAlerts(risk) : []), [risk]);
  const symbols = useMemo(() => Object.fromEntries(((snapshot && snapshot.holdings) || []).map((h) => [h.isin, h.symbol])), [snapshot]);

  if (!snapshot) {
    return html`<${Page} title="Risk" subtitle="Where the risk actually comes from: contribution, concentration, correlation, value at risk.">
      <${Card}><${EmptyState} icon="clock" title="Loading the snapshot…" body="The risk model arrives with the snapshot." /><//>
    <//>`;
  }

  if (!risk || !risk.available) {
    return html`<${Page} title="Risk" subtitle="Where the risk actually comes from: contribution, concentration, correlation, value at risk.">
      <${Card}><${EmptyState} icon="risk" title="No risk model"
        body=${(risk && risk.reason) || "The risk model needs at least two priced holdings with overlapping price history."}
        action=${html`<${Button} variant="primary" icon="holdings" onClick=${() => navigate("/holdings")}>Holdings<//>`} /><//>
    <//>`;
  }

  const w = risk.window;
  const subtitle = `${fmt.int(risk.actual_holdings)} holdings in the model · ${w ? `${fmt.int(w.effective)}-day window to ${fmt.date(w.last_date)}` : ""} · beta vs ${risk.beta_benchmark || "no benchmark"} · pairs flagged above ${fmt.num(risk.threshold, { decimals: 2 })}`;

  return html`<${Page} title="Risk" subtitle=${subtitle}
    actions=${html`<${Button} variant="secondary" icon="sliders" onClick=${() => navigate("/simulator")}>Simulate a change<//>`}>
    <div class="stack gap-4">
      <${Headline} risk=${risk} />
      <${Notice} alerts=${alerts} visible=${3} onNavigate=${navigate}
        summary=${alerts.length ? `${fmt.count(alerts.length, "thing")} ${alerts.length === 1 ? "qualifies" : "qualify"} the figures below` : undefined} />
      <div class="grid">
        <div class="col-8 stack gap-4">
          <${DivergenceChart} risk=${risk} />
          <${DivergenceTable} risk=${risk} />
        </div>
        <div class="col-4">
          <${Metrics} risk=${risk} />
        </div>
      </div>
      <div class="grid">
        <${StandaloneChart} risk=${risk} />
        <${CorrelationCard} risk=${risk} symbols=${symbols} />
      </div>
      <div class="grid">
        <${VarCard} risk=${risk} base=${base} total=${total} />
        <${ReturnsHistogram} risk=${risk} />
      </div>
      <div class="grid">
        <${WorstPeriods} risk=${risk} base=${base} />
        <${Scenarios} risk=${risk} base=${base} />
      </div>
    </div>
  <//>`;
}
