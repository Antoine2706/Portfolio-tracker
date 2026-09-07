/* Simulator — the what-if workstation. Props: { route, snapshot }.

   Layout order and why:
   1. Header: the page's two verbs, Reset and Simulate, live in the page actions.
   2. Left column — the question. The current book as editable rows with a mode
      switch (add/remove cash · target weights · presets), the watchlist rows
      that may be admitted to the model, the minimum-trade input, and the inline
      error. Everything that changes the request sits here, nothing else does.
   3. Right column — the answer, in reading order: five before → after tiles
      (the headline), the two before/after charts (risk share, then weight),
      the per-holding table (the detail), and the trade list that would get
      you there (the action). While a request is pending the previous answer
      stays on screen at reduced opacity; nothing jumps.
   The page computes nothing: every after-number comes from POST /api/simulate.
   Client-side work is limited to building the request and formatting. */

import { html, useState, useEffect, useMemo, useRef, useCallback } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { Button } from "/static/components/Button.js";
import { Badge, TxTypeBadge } from "/static/components/Badge.js";
import { Banner } from "/static/components/Banner.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { KpiTile } from "/static/components/KpiTile.js";
import { Chart } from "/static/components/Chart.js";
import { DataTable } from "/static/components/DataTable.js";
import { Field, Input, Checkbox } from "/static/components/Field.js";
import { Segmented } from "/static/components/Segmented.js";
import { Help } from "/static/components/Tooltip.js";
import { toast } from "/static/components/Toast.js";
import { simulate, errorMessage } from "/static/lib/api.js";
import { barOption } from "/static/lib/charts.js";
import { tokens } from "/static/lib/theme.js";
import { navigate } from "/static/lib/router.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- page-local styles ---------------- */

const STYLE = `
.sim-book .table td, .sim-book .table th { height: 40px; }
.sim-book .table td.sim-input-cell { padding-top: 4px; padding-bottom: 4px; width: 112px; }
.sim-book .table td.num .cell-sub { text-align: right; }
.sim-book .input.num { height: 26px; font-size: var(--fs-sm); }
.sim-book .sim-group td { height: 28px; padding-top: 8px; font-size: var(--fs-xs); font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--color-text-3); border-bottom: 0; }
.sim-book tr.is-off td { color: var(--color-text-3); }
.sim-book tr.is-off .cell-sub { color: var(--color-text-3); }
.sim-results { transition: opacity var(--dur) var(--ease); }
.sim-results.is-pending { opacity: .55; pointer-events: none; }
.sim-tiles { display: grid; gap: var(--sp-3); grid-template-columns: repeat(5, minmax(0, 1fr)); }
@media (max-width: 1400px) { .sim-tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 900px) { .sim-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.sim-tiles .kpi { min-height: 88px; padding: 12px 14px; }
.sim-tiles .kpi-value.sm { font-size: 20px; }
.sim-presets { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--sp-2); }
.sim-presets .btn { height: auto; padding: 8px 10px; white-space: normal; text-align: left; align-items: flex-start; }
.sim-presets .btn-label { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; }
.sim-presets .preset-sub { font-weight: 400; font-size: var(--fs-xs); color: var(--color-text-3); }
.sim-presets .btn-primary .preset-sub { color: var(--color-accent-fg); opacity: .8; }
.sim-trades .table td { height: 38px; }
.sim-trades .weight-arrow { color: var(--color-text-3); padding: 0 4px; }
.sim-turnover { display: flex; flex-wrap: wrap; gap: var(--sp-4); }
.sim-turnover .stat-inline .stat-value { font-size: var(--fs-md); }
.sim-sum { display: flex; justify-content: space-between; align-items: center; gap: var(--sp-2); font-size: var(--fs-xs); color: var(--color-text-3); padding: 6px var(--sp-4) 0; }
.sim-sum .num { color: var(--color-text-2); }
`;

/* ---------------- helpers ---------------- */

const PRESETS = [
  { value: "equal", label: "Equal weight", sub: "1/N of the book in each modelled holding" },
  { value: "risk_parity", label: "Risk parity", sub: "every holding contributes the same share of risk" },
  { value: "min_variance", label: "Minimum variance", sub: "the long-only mix with the lowest volatility" },
];

const MODES = [
  { value: "cash", label: "Add / remove cash" },
  { value: "targets", label: "Target weights" },
  { value: "presets", label: "Presets" },
];

/** "1,250.5" | "1 250,5" | "-300" → number | null. Blank is null, not zero. */
function parseNum(s) {
  if (s == null) return null;
  let t = String(s).trim().replace(/\s/g, "").replace(/€|%/g, "");
  if (!t) return null;
  if (t.includes(",") && !t.includes(".")) t = t.replace(",", ".");
  else t = t.replace(/,/g, "");
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

function nearZero(v) { return v == null || Math.abs(v) < 1e-9; }

/** Rows of the book: held instruments first (largest first), then watchlist
    instruments the risk model can admit. Holdings excluded from the model are
    kept visible but disabled, with the reason in a tip. */
function bookRows(snapshot) {
  if (!snapshot) return [];
  const excluded = new Map();
  for (const e of (snapshot.risk && snapshot.risk.window && snapshot.risk.window.excluded) || []) excluded.set(e.isin, e.reason || "excluded from the risk model");
  const held = (snapshot.holdings || [])
    .slice()
    .sort((a, b) => (b.value || 0) - (a.value || 0))
    .map((h) => ({
      isin: h.isin, name: h.name, symbol: h.symbol, value: h.value, weight: h.weight, risk: h.risk_share,
      kind: "holding", modelled: !excluded.has(h.isin) && h.value != null, reason: excluded.get(h.isin) || (h.value == null ? "unpriced: no market value to simulate with" : null),
    }));
  const watch = (snapshot.watchlist || []).map((w) => ({
    isin: w.isin, name: w.name, symbol: w.symbol, value: null, weight: null, risk: null,
    kind: "watchlist", modelled: !!w.in_risk_model, reason: w.in_risk_model ? null : "not enough price history to enter the risk model",
  }));
  return [...held, ...watch];
}

function tradesText(result) {
  if (!result || !result.trades.length) return "Nothing to trade.";
  const lines = result.trades.map((t) => [
    t.action.padEnd(4), fmt.money(t.amount), t.units == null ? "" : `${fmt.qty(t.units)} units`,
    t.price == null ? "" : `@ ${fmt.money(t.price)}`, t.name, `${fmt.pct(t.weight_before)} → ${fmt.pct(t.weight_after)}`,
  ].filter(Boolean).join("  "));
  const turnover = result.trades.reduce((s, t) => s + (t.amount || 0), 0);
  lines.push(`Turnover ${fmt.money(turnover)} · total ${fmt.money(result.total_before)} → ${fmt.money(result.total_after)}`);
  return lines.join("\n");
}

async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) { await navigator.clipboard.writeText(text); return true; }
  } catch (_) { /* fall through */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch (_) { return false; }
}

/* ---------------- charts ---------------- */

/** Grouped horizontal bars, Before (quiet gray) vs After (accent), direct label
    on the After bar only. Legend is on because there are two series. */
function beforeAfterOption(rows, key) {
  const t = tokens();
  const opt = barOption({
    categories: rows.map((r) => fmt.shortName(r.name, 26)),
    series: [
      { name: "Before", values: rows.map((r) => r[`${key}_before`]), color: t.text3 },
      { name: "After", values: rows.map((r) => r[`${key}_after`]), color: t.series[0] },
    ],
    horizontal: true, format: "pct",
  });
  // barOption colours each data item, which leaves the legend on the theme
  // palette; a series-level colour keeps legend and bars in agreement.
  opt.series[0].itemStyle = { color: t.text3 };
  opt.series[1].itemStyle = { color: t.series[0] };
  opt.series[0].label = { show: false };
  opt.series[1].label = { show: true, position: "right", color: t.text2, fontSize: 11, formatter: (p) => fmt.pct(p.value) };
  opt.grid.right = 52;
  opt.xAxis.max = (v) => Math.min(1, Math.ceil((v.max + 0.02) * 10) / 10);
  return opt;
}

function beforeAfterTable(rows, key, label) {
  return {
    columns: [
      { key: "name", label: "Holding", sortable: false },
      { key: "before", label: `${label} before`, format: "pct", sortable: false },
      { key: "after", label: `${label} after`, format: "pct", sortable: false },
      { key: "delta", label: "Change", format: "pp", sortable: false },
    ],
    rows: rows.map((r) => ({ id: r.isin, name: r.name, before: r[`${key}_before`], after: r[`${key}_after`], delta: r[`${key}_after`] - r[`${key}_before`] })),
  };
}

/* ---------------- results ---------------- */

function Tiles({ result }) {
  const dVol = result.volatility_after - result.volatility_before;
  const dEff = result.effective_after - result.effective_before;
  const dDiv = result.diversification_after - result.diversification_before;
  const dMax = result.max_risk_share_after - result.max_risk_share_before;
  const dTot = result.total_after - result.total_before;
  const num2 = (v) => fmt.num(v, { decimals: 2, signed: true });
  const totalDelta = nearZero(dTot) ? "unchanged" : `${fmt.money(dTot, "EUR", { signed: true, compact: true })} ${dTot > 0 ? "added" : "withdrawn"}`;
  return html`<div class="sim-tiles">
    <${KpiTile} size="sm" label="Annualised volatility" value=${fmt.pct(result.volatility_after)}
      delta=${nearZero(dVol) ? 0 : dVol} deltaFormat="pp" upIsGood=${false} deltaLabel=${`from ${fmt.pct(result.volatility_before)}`} sub="lower is better"
      help="Portfolio standard deviation of daily returns, annualised, from the covariance of the risk window." />
    <${KpiTile} size="sm" label="Effective holdings" value=${fmt.num(result.effective_after, { decimals: 1 })}
      delta=${nearZero(dEff) ? 0 : dEff} deltaFormat=${num2} upIsGood=${true} deltaLabel=${`from ${fmt.num(result.effective_before, { decimals: 1 })}`} sub="higher is better"
      help="1 / Σ weight². Six equal holdings count as 6; one dominant holding pulls it toward 1." />
    <${KpiTile} size="sm" label="Diversification ratio" value=${fmt.mult(result.diversification_after)}
      delta=${nearZero(dDiv) ? 0 : dDiv} deltaFormat=${(v) => `${num2(v)}×`} upIsGood=${true} deltaLabel=${`from ${fmt.mult(result.diversification_before)}`} sub="higher is better"
      help="Weighted average of standalone volatilities divided by portfolio volatility. 1.0× means the holdings move as one." />
    <${KpiTile} size="sm" label="Max risk share" value=${fmt.pct(result.max_risk_share_after)}
      delta=${nearZero(dMax) ? 0 : dMax} deltaFormat="pp" upIsGood=${false} deltaLabel=${`from ${fmt.pct(result.max_risk_share_before)}`} sub="lower is better"
      help="The largest single holding's share of portfolio risk." />
    <${KpiTile} size="sm" label="Total value" value=${fmt.money(result.total_after, "EUR", { compact: true })}
      delta=${totalDelta} deltaLabel=${nearZero(dTot) ? undefined : `from ${fmt.money(result.total_before, "EUR", { compact: true })}`} sub="cash in or out; neither is good or bad" />
  </div>`;
}

function HoldingsTable({ rows }) {
  const money0 = (v) => fmt.money(v, "EUR", { decimals: 0 });
  const columns = useMemo(() => [
    { key: "name", label: "Holding", primary: true, render: (r, v) => html`<div class="truncate" style="max-width:240px" title=${v}>${v}<span class="cell-sub">${r.isin}</span></div>` },
    { key: "value_before", label: "Value before", format: money0 },
    { key: "value_after", label: "Value after", format: money0, sub: (r) => (nearZero(r.value_after - r.value_before) ? "" : fmt.money(r.value_after - r.value_before, "EUR", { signed: true, decimals: 0 })) },
    { key: "weight_before", label: "Weight before", format: "pct" },
    { key: "weight_after", label: "Weight after", format: "pct", sub: (r) => (nearZero(r.weight_after - r.weight_before) ? "" : fmt.pp(r.weight_after - r.weight_before)) },
    { key: "risk_before", label: "Risk before", format: "pct" },
    { key: "risk_after", label: "Risk after", format: "pct", sub: (r) => (nearZero(r.risk_after - r.risk_before) ? "" : fmt.pp(r.risk_after - r.risk_before)) },
    { key: "marginal_after", label: "Marginal after", format: "pct", title: "Marginal contribution to risk after the change, annualised" },
  ], []);
  return html`<${DataTable} columns=${columns} rows=${rows} rowKey="isin" density="compact" sort=${{ key: "value_after", dir: "desc" }} empty="No holdings in the model" />`;
}

function Trades({ result, currency = "EUR" }) {
  const trades = result.trades || [];
  const buys = trades.filter((t) => t.action === "BUY").reduce((s, t) => s + t.amount, 0);
  const sells = trades.filter((t) => t.action === "SELL").reduce((s, t) => s + t.amount, 0);
  const turnover = buys + sells;
  const onCopy = async () => {
    const ok = await copyText(tradesText(result));
    if (ok) toast.success("Trade list copied as text"); else toast.error("Could not copy to the clipboard");
  };
  return html`<${Card} class="sim-trades" title="Trades to get there" flush
    caption=${trades.length ? `${trades.length} trade${trades.length === 1 ? "" : "s"} · amounts in ${currency} at the latest price` : "The list of buys and sells that moves the book from before to after."}
    actions=${trades.length ? html`<${Button} size="sm" variant="ghost" icon="copy" onClick=${onCopy}>Copy as text<//>` : null}
    footer=${trades.length ? html`<div class="sim-turnover">
      <span class="stat-inline"><span class="stat-label">Turnover</span><span class="stat-value">${fmt.money(turnover, currency)}</span></span>
      <span class="stat-inline"><span class="stat-label">Buys</span><span class="stat-value">${fmt.money(buys, currency)}</span></span>
      <span class="stat-inline"><span class="stat-label">Sells</span><span class="stat-value">${fmt.money(sells, currency)}</span></span>
      <span class="stat-inline"><span class="stat-label">Turnover / book</span><span class="stat-value">${fmt.pct(result.total_before ? turnover / result.total_before : null)}</span></span>
    </div>` : null}>
    ${trades.length === 0 ? html`<${EmptyState} compact icon="check" title="Nothing to trade" body="The after-state matches the book within the minimum trade size." /> ` : html`
    <div class="table-wrap"><table class="table" data-density="compact">
      <thead><tr><th scope="col">Action</th><th scope="col">Instrument</th><th scope="col" class="num">Amount</th><th scope="col" class="num">Units</th><th scope="col" class="num">Price</th><th scope="col" class="num">Weight</th></tr></thead>
      <tbody>
        ${trades.map((t) => html`<tr key=${t.isin + t.action}>
          <td><${TxTypeBadge} type=${t.action} /></td>
          <td class="primary"><div class="truncate" style="max-width:260px" title=${t.name}>${t.name}<span class="cell-sub">${t.isin}</span></div></td>
          <td class="num">${fmt.money(t.amount, currency)}</td>
          <td class="num">${t.units == null ? html`<span class="faint" title="No price is known, so the amount cannot be turned into units">${fmt.DASH}</span>` : fmt.qty(t.units)}</td>
          <td class="num">${fmt.money(t.price, currency)}</td>
          <td class="num nowrap">${fmt.pct(t.weight_before)}<span class="weight-arrow">→</span><span class="strong">${fmt.pct(t.weight_after)}</span></td>
        </tr>`)}
      </tbody>
    </table></div>`}
  <//>`;
}

function Results({ result, pending, currency }) {
  const rows = useMemo(() => (result ? result.holdings.slice().sort((a, b) => b.value_before - a.value_before || b.value_after - a.value_after) : []), [result]);
  const riskOpt = useMemo(() => (rows.length ? beforeAfterOption(rows, "risk") : null), [rows]);
  const weightOpt = useMemo(() => (rows.length ? beforeAfterOption(rows, "weight") : null), [rows]);
  const riskTable = useMemo(() => beforeAfterTable(rows, "risk", "Risk share"), [rows]);
  const weightTable = useMemo(() => beforeAfterTable(rows, "weight", "Weight"), [rows]);
  const height = Math.max(180, rows.length * 34 + 56);
  return html`<div class=${["stack gap-4 sim-results", pending ? "is-pending" : ""].join(" ")} aria-busy=${pending ? "true" : "false"}>
    <${Tiles} result=${result} />
    <div class="grid">
      <div class="col-6"><${Chart} title="Risk share, before → after" caption="Share of portfolio risk each holding carries. The after bar is labelled." option=${riskOpt} height=${height} table=${riskTable} empty="No holdings" /></div>
      <div class="col-6"><${Chart} title="Weight, before → after" caption="Share of portfolio value. Compare with risk share to see who carries more risk than capital." option=${weightOpt} height=${height} table=${weightTable} empty="No holdings" /></div>
    </div>
    <${Card} title="Per holding" caption="Values in EUR; marginal is the contribution to annualised volatility of one more unit of weight." flush>
      <${HoldingsTable} rows=${rows} />
    <//>
    <${Trades} result=${result} currency=${currency} />
  </div>`;
}

/* ---------------- the book (left column) ---------------- */

function BookTable({ rows, mode, cash, targets, include, result, onCash, onTarget, onInclude, focusIsin }) {
  const held = rows.filter((r) => r.kind === "holding");
  const watch = rows.filter((r) => r.kind === "watchlist");
  const afterWeights = result && result.weights_after;
  const inputCol = mode === "cash" ? "Change" : mode === "targets" ? "Target" : "After";
  const renderInput = (r) => {
    const off = !r.modelled || (r.kind === "watchlist" && !include.has(r.isin));
    if (mode === "presets") {
      const w = afterWeights ? afterWeights[r.isin] : null;
      return html`<span class=${w == null ? "faint" : "num"}>${w == null ? fmt.DASH : fmt.pct(w)}</span>`;
    }
    if (mode === "cash") {
      return html`<${Input} numeric value=${cash[r.isin] ?? ""} placeholder="0" prefix="€" disabled=${off} data-isin=${r.isin}
        aria-label=${`Change in EUR for ${r.name}`} onInput=${(e) => onCash(r.isin, e.target.value)} />`;
    }
    return html`<${Input} numeric value=${targets[r.isin] ?? ""} placeholder="0" suffix="%" disabled=${off} data-isin=${r.isin}
      aria-label=${`Target weight for ${r.name}`} onInput=${(e) => onTarget(r.isin, e.target.value)} />`;
  };
  const row = (r) => {
    const off = !r.modelled;
    const nameCell = html`<div class="truncate" style="max-width:150px" title=${r.name}>
      <span>${r.name}</span>
      <span class="cell-sub">${[r.symbol, r.isin].filter(Boolean).join(" · ")}${off && r.reason ? html` · <span class="warn-mark" title=${r.reason}>not modelled</span>` : null}</span>
    </div>`;
    return html`<tr key=${r.isin} class=${[off ? "is-off" : "", focusIsin === r.isin ? "selected" : ""].filter(Boolean).join(" ")}>
      <td class="primary">${r.kind === "watchlist" ? html`<div class="row" style="gap:8px"><${Checkbox} checked=${include.has(r.isin)} disabled=${off} onChange=${(v) => onInclude(r.isin, v)} aria-label=${`Include ${r.name} in the model`} />${nameCell}</div>` : nameCell}</td>
      <td class="num">${r.kind === "watchlist"
        ? html`<span class="faint">${fmt.DASH}</span><span class="cell-sub">0.0%</span>`
        : html`${fmt.money(r.value, "EUR", { decimals: 0 })}<span class="cell-sub">${fmt.pct(r.weight)}</span>`}</td>
      <td class="num sim-input-cell">${renderInput(r)}</td>
    </tr>`;
  };
  return html`<div class="table-wrap sim-book"><table class="table" data-density="compact">
    <thead><tr>
      <th scope="col">Holding</th><th scope="col" class="num">Value · weight</th>
      <th scope="col" class="num">${inputCol}${mode === "cash" ? html` <span class="faint" style="text-transform:none;letter-spacing:0">EUR ±</span>` : mode === "targets" ? html` <span class="faint" style="text-transform:none;letter-spacing:0">%</span>` : null}</th>
    </tr></thead>
    <tbody>
      ${held.map(row)}
      ${watch.length ? html`<tr class="sim-group"><td colspan="3">Watchlist · tick to admit to the model</td></tr>` : null}
      ${watch.map(row)}
      ${!held.length && !watch.length ? html`<tr><td class="empty-cell" colspan="3">Nothing is held</td></tr>` : null}
    </tbody>
  </table></div>`;
}

/* ---------------- page ---------------- */

export default function SimulatorPage({ route, snapshot }) {
  const rows = useMemo(() => bookRows(snapshot), [snapshot]);
  const currency = (snapshot && snapshot.meta && snapshot.meta.base_currency) || "EUR";
  const risk = snapshot && snapshot.risk;

  const [mode, setMode] = useState("cash");
  const [cash, setCash] = useState({});
  const [targets, setTargets] = useState({});
  const [preset, setPreset] = useState(null);
  const [include, setInclude] = useState(() => new Set());
  const [minTrade, setMinTrade] = useState("");
  const [result, setResult] = useState(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [tick, setTick] = useState(0);
  const seq = useRef(0);
  const bump = () => setTick((t) => t + 1);

  // Targets start from the current weights, so leaving a row alone keeps it
  // (the server treats a missing target as 0%).
  const currentTargets = useCallback(() => {
    const t = {};
    for (const r of rows) if (r.kind === "holding" && r.modelled && r.weight != null) t[r.isin] = fmt.num(r.weight * 100, { decimals: 1 }).replace(/,/g, "");
    return t;
  }, [rows]);
  useEffect(() => { setTargets((prev) => (Object.keys(prev).length ? prev : currentTargets())); }, [currentTargets]);

  const buildRequest = () => {
    const body = { include: Array.from(include), min_trade: Math.max(0, parseNum(minTrade) || 0) };
    if (mode === "presets") {
      if (!preset) return null;
      body.preset = preset;
    } else if (mode === "targets") {
      const t = {};
      for (const r of rows) {
        if (!r.modelled) continue;
        if (r.kind === "watchlist" && !include.has(r.isin)) continue;
        const n = parseNum(targets[r.isin]);
        if (n != null && n > 0) t[r.isin] = n / 100;
      }
      if (!Object.keys(t).length) return null;
      body.targets = t;
    } else {
      const c = {};
      for (const r of rows) {
        if (!r.modelled) continue;
        if (r.kind === "watchlist" && !include.has(r.isin)) continue;
        const n = parseNum(cash[r.isin]);
        if (n != null && !nearZero(n)) c[r.isin] = n;
      }
      if (!Object.keys(c).length) return null;
      body.changes = c;
    }
    return body;
  };

  const run = async (body) => {
    const id = ++seq.current;
    setPending(true);
    try {
      const res = await simulate(body);
      if (id !== seq.current) return;
      setResult(res);
      setError(null);
    } catch (err) {
      if (id !== seq.current) return;
      const msg = errorMessage(err);
      setError(msg);
      toast.error(msg, { title: "Simulation failed" });
    } finally {
      if (id === seq.current) setPending(false);
    }
  };

  // Debounced run on input change.
  useEffect(() => {
    if (tick === 0) return undefined;
    const body = buildRequest();
    if (!body) return undefined;
    const h = setTimeout(() => run(body), 400);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  const onSimulate = () => {
    const body = buildRequest();
    if (!body) {
      toast.info(mode === "presets" ? "Pick a preset first." : mode === "targets" ? "Give at least one target weight." : "Enter a change in EUR for at least one holding.");
      return;
    }
    run(body);
  };

  const onReset = () => {
    seq.current++;
    setCash({}); setTargets(currentTargets()); setPreset(null); setInclude(new Set()); setMinTrade("");
    setResult(null); setError(null); setPending(false);
  };

  const onCash = (isin, v) => { setCash((c) => ({ ...c, [isin]: v })); bump(); };
  const onTarget = (isin, v) => { setTargets((t) => ({ ...t, [isin]: v })); bump(); };
  const onInclude = (isin, v) => {
    setInclude((s) => { const n = new Set(s); if (v) n.add(isin); else n.delete(isin); return n; });
    if (!v) { setCash((c) => ({ ...c, [isin]: "" })); setTargets((t) => ({ ...t, [isin]: "" })); }
    bump();
  };
  const onPreset = (p) => { setPreset(p); bump(); };
  const onMinTrade = (v) => { setMinTrade(v); bump(); };

  // ?isin= pre-focuses that holding's input.
  const focusIsin = route && route.params && route.params.isin ? String(route.params.isin).toUpperCase() : null;
  useEffect(() => {
    if (!focusIsin || !rows.length) return;
    const r = rows.find((x) => x.isin === focusIsin);
    if (r && r.kind === "watchlist" && r.modelled) setInclude((s) => (s.has(focusIsin) ? s : new Set([...s, focusIsin])));
    const h = setTimeout(() => {
      const el = document.querySelector(`input[data-isin="${focusIsin}"]`);
      if (el) { el.focus(); el.scrollIntoView({ block: "center" }); }
    }, 60);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusIsin, rows.length, mode]);

  const targetSum = useMemo(() => {
    if (mode !== "targets") return null;
    let s = 0;
    for (const r of rows) {
      if (!r.modelled || (r.kind === "watchlist" && !include.has(r.isin))) continue;
      const n = parseNum(targets[r.isin]); if (n != null && n > 0) s += n;
    }
    return s;
  }, [mode, rows, targets, include]);
  const cashSum = useMemo(() => {
    if (mode !== "cash") return null;
    let s = 0;
    for (const r of rows) { const n = parseNum(cash[r.isin]); if (n != null) s += n; }
    return s;
  }, [mode, rows, cash]);

  const modelled = rows.filter((r) => r.kind === "holding" && r.modelled).length;
  const admitted = rows.filter((r) => r.kind === "watchlist" && include.has(r.isin)).length;
  const bookValue = snapshot && snapshot.totals ? snapshot.totals.value : null;

  const actions = html`
    <${Button} variant="secondary" icon="refresh" onClick=${onReset} title="Clear every change and the result">Reset<//>
    <${Button} variant="primary" icon="zap" loading=${pending} onClick=${onSimulate}>Simulate<//>
  `;

  if (!snapshot) {
    return html`<${Page} title="Simulator" subtitle="What-if: cash changes, target weights and presets, with the before/after risk diff.">
      <${Card}><${EmptyState} icon="simulator" title="Loading the book…" body="The simulator needs the snapshot's holdings and risk model." /><//>
    <//>`;
  }
  if (!risk || !risk.available) {
    return html`<${Page} title="Simulator" subtitle="What-if: cash changes, target weights and presets, with the before/after risk diff.">
      <${Card}><${EmptyState} icon="risk" title="No risk model to simulate against"
        body=${(risk && risk.reason) || "The risk model is unavailable, so there is no covariance to project the book through."}
        action=${html`<${Button} variant="secondary" onClick=${() => navigate("/risk")}>Open the Risk page<//>`} /><//>
    <//>`;
  }

  const modeHelp = mode === "cash"
    ? "Positive adds money to a holding at the latest price, negative takes it out. Values stay in EUR; the rest of the book is untouched."
    : mode === "targets"
      ? "Weights are normalised to 100% by the server, so they only need to be in proportion. A blank row means 0%: it gets sold."
      : "A preset replaces every target weight in one go. Watchlist rows you have ticked take part.";

  return html`<${Page} title="Simulator" subtitle="What-if: cash changes, target weights and presets, with the before/after risk diff." actions=${actions}>
    <style>${STYLE}</style>
    <div class="grid">
      <div class="col-4">
        <${Card} title="The book" flush
          caption=${`${fmt.money(bookValue, currency, { compact: true })} · ${modelled} of ${rows.filter((r) => r.kind === "holding").length} holdings in the model${admitted ? ` · ${admitted} from the watchlist` : ""}`}>
          <div class="stack" style="padding: 12px var(--sp-4) 10px; gap: 10px">
            <${Segmented} ariaLabel="How to change the book" value=${mode} onChange=${setMode} options=${MODES} />
            <div class="small muted">${modeHelp}</div>
            ${mode === "presets" ? html`<div class="sim-presets">
              ${PRESETS.map((p) => html`<${Button} key=${p.value} variant=${preset === p.value ? "primary" : "secondary"} aria-pressed=${preset === p.value ? "true" : "false"} onClick=${() => onPreset(p.value)}>
                ${p.label}<span class="preset-sub">${p.sub}</span>
              <//>`)}
            </div>` : null}
          </div>
          <${BookTable} rows=${rows} mode=${mode} cash=${cash} targets=${targets} include=${include} result=${result}
            onCash=${onCash} onTarget=${onTarget} onInclude=${onInclude} focusIsin=${focusIsin} />
          ${mode === "targets" ? html`<div class="sim-sum"><span>Sum of targets</span><span class="num">${fmt.num(targetSum, { decimals: 1 })}% ${targetSum && Math.abs(targetSum - 100) > 0.05 ? html`<span class="faint">→ normalised to 100%</span>` : null}</span></div>` : null}
          ${mode === "cash" ? html`<div class="sim-sum"><span>Net cash</span><span class="num">${fmt.money(cashSum, currency, { signed: true })}</span></div>` : null}
          <div class="stack" style="padding: 12px var(--sp-4) var(--sp-4); gap: 12px">
            <${Field} label=${html`Minimum trade <${Help} text="Trades smaller than this are left out of the trade list. The after-state itself is unchanged." />`} id="sim-min-trade" hint="Drops dust trades from the list, in EUR.">
              <${Input} id="sim-min-trade" numeric value=${minTrade} placeholder="0" suffix="EUR" onInput=${(e) => onMinTrade(e.target.value)} />
            <//>
            <div class="row between">
              <span class="faint xs">${pending ? "Simulating…" : result ? "Re-runs 400 ms after you stop typing." : "Runs when you change a value."}</span>
              <${Button} variant="primary" size="sm" icon="zap" loading=${pending} onClick=${onSimulate}>Simulate<//>
            </div>
            ${error ? html`<${Banner} kind="error" title="The server refused the request.">${error}<//>` : null}
          </div>
        <//>
      </div>
      <div class="col-8">
        ${result ? html`<${Results} result=${result} pending=${pending} currency=${currency} />` : html`
          <${Card}><${EmptyState} icon="simulator" title="Change something to see the before → after"
            body=${mode === "presets" ? "Pick Equal weight, Risk parity or Minimum variance on the left." : mode === "targets" ? "Edit a target weight on the left; the result appears here after a short pause." : "Type a cash change (+ or −) next to a holding on the left; the result appears here after a short pause."}
            action=${mode === "presets" ? null : html`<${Button} variant="secondary" onClick=${() => { setMode("presets"); }}>Try a preset<//>`} /><//>`}
      </div>
    </div>
  <//>`;
}
