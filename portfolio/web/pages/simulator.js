/* Simulator — "what happens to my risk before I trade". Props: { route, snapshot }.

   Layout order and why:
   1. Left column — the question. Three modes as tabs: add or remove money per
      holding (changes), target weights (targets), or one of three presets
      (equal weight, risk parity, minimum variance). Below the held rows sits
      the watchlist ("add to the model"), then the minimum trade size and a
      live summary (net cash / total after, or the running sum of targets).
      Everything that changes the request lives here; nothing else does.
   2. Right column — the answer, in reading order: five before → after tiles
      (the headline), the paired bar chart per holding with a Segmented measure
      switch (risk share / weight / marginal), the per-holding table, and the
      trade list that would get you there. While a request is in flight the
      previous answer stays on screen at reduced opacity; nothing jumps.
   The page computes nothing: every after-number comes from POST /api/simulate,
   debounced 250 ms after the last edit. Client-side work is limited to input
   bookkeeping (running sums, normalising target inputs) and formatting. The
   last inputs are remembered for the session so returning to the page restores
   them. A 409 (no risk model) is shown with a link to Risk; a 400 inline. */

import { html, useState, useEffect, useMemo, useRef } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { Button } from "/static/components/Button.js";
import { TxTypeBadge } from "/static/components/Badge.js";
import { Banner } from "/static/components/Banner.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Chart } from "/static/components/Chart.js";
import { DataTable } from "/static/components/DataTable.js";
import { Field, Input, Checkbox } from "/static/components/Field.js";
import { Segmented } from "/static/components/Segmented.js";
import { Tabs } from "/static/components/Tabs.js";
import { Help } from "/static/components/Tooltip.js";
import { toast } from "/static/components/Toast.js";
import { api, errorMessage } from "/static/lib/api.js";
import { barOption, tipElement } from "/static/lib/charts.js";
import { tokens } from "/static/lib/theme.js";
import { useStore } from "/static/lib/store.js";
import { navigate } from "/static/lib/router.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- constants ---------------- */

const DEBOUNCE_MS = 250;
const SLIDER_MIN_MAX = 5000;
const SUBTITLE = "What happens to the risk picture before you trade: money in or out, target weights, or a preset, all answered before → after.";

const MODES = [
  { id: "changes", label: "Add or remove money" },
  { id: "targets", label: "Target weights" },
  { id: "presets", label: "Presets" },
];

const MODE_HELP = {
  changes: "Positive adds money to a holding at the latest price, negative takes it out; the rest of the book is untouched. Dragging past “sell everything” sells everything.",
  targets: "Give each holding the share of the book it should end up with. The total is held constant, so this is a rebalance, not a deposit. A blank row is sold out.",
  presets: "A preset replaces every weight in one go. Watchlist instruments you tick take part.",
};

const PRESETS = [
  { value: "equal", label: "Equal weight", body: "The same amount of money in each holding — the naive baseline the other two are measured against." },
  { value: "risk_parity", label: "Risk parity", body: "Each holding contributes the same share of portfolio volatility, so volatile holdings get less money and calm ones more. It is what the divergence table on the Risk page is implicitly asking for." },
  { value: "min_variance", label: "Minimum variance", body: "The long-only mix with the lowest volatility over the window. It usually concentrates in the calmest holdings, which is why it is shown and not recommended." },
];

const MEASURES = [
  { value: "risk", label: "Risk share" },
  { value: "weight", label: "Weight" },
  { value: "marginal", label: "Marginal" },
];

const MEASURE_CAPTION = {
  risk: "Share of portfolio volatility each holding carries, before and after. The after bar is labelled; the legend names the pair.",
  weight: "Share of portfolio value, before and after. Compare with risk share to see who carries more risk than capital.",
  marginal: "Marginal contribution to risk, annualised: how much portfolio volatility moves per extra euro in each holding. “Before” is the current figure from the Risk page; a watchlist instrument has none yet.",
};

/** The last inputs and answer, kept for the session so returning to the page restores them. */
let remembered = null;

/* ---------------- helpers ---------------- */

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

const nearZero = (v) => v == null || Math.abs(v) < 1e-9;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Fraction → input string in percent with one decimal and no thousands separators: 0.2587 → "25.9". */
const pctStr = (f) => (f == null ? "" : String(Math.round(f * 1000) / 10));

/** Currency affix for a numeric input: € as a prefix, any other code as a suffix. */
const moneyAffix = (ccy) => (ccy === "EUR" ? { prefix: "€" } : { suffix: ccy });

/** Book rows from the snapshot: held instruments in the model (largest first), the
    excluded ones with a reason, watchlist instruments the model can admit, and the
    current MCTR per ISIN from the Risk page (the "before" of the marginal view). */
function bookRows(snapshot) {
  const empty = { held: [], excluded: [], watch: [], marginalBefore: new Map() };
  if (!snapshot) return empty;
  const reasons = new Map();
  for (const e of (snapshot.risk && snapshot.risk.window && snapshot.risk.window.excluded) || []) reasons.set(e.isin, e.reason || "excluded from the risk model");
  const held = [], excluded = [];
  for (const h of (snapshot.holdings || []).slice().sort((a, b) => (b.value || 0) - (a.value || 0))) {
    const row = { isin: h.isin, name: h.name, symbol: h.symbol, value: h.value, weight: h.weight, risk: h.risk_share, price: h.price };
    if (h.risk_share != null && h.value != null) held.push(row);
    else excluded.push({ ...row, reason: reasons.get(h.isin) || (h.value == null ? "no price, so no market value to simulate with" : "not enough price history for the risk window") });
  }
  const watch = (snapshot.watchlist || []).filter((w) => w.in_risk_model)
    .map((w) => ({ isin: w.isin, name: w.name, symbol: w.symbol, price: w.price, currency: w.price_currency }));
  const marginalBefore = new Map();
  for (const d of (snapshot.risk && snapshot.risk.divergence) || []) marginalBefore.set(d.isin, d.marginal);
  return { held, excluded, watch, marginalBefore };
}

/** The request body for the current inputs, or null when there is nothing to ask. */
function buildRequest({ mode, changes, targets, preset, include, minTrade, book }) {
  const body = { include: Array.from(include), min_trade: Math.max(0, parseNum(minTrade) || 0) };
  if (mode === "presets") {
    if (!preset) return null;
    body.preset = preset;
    return body;
  }
  if (mode === "targets") {
    const t = {};
    for (const h of book.held) { const n = parseNum(targets[h.isin]); if (n != null && n > 0) t[h.isin] = n / 100; }
    for (const w of book.watch) if (include.has(w.isin)) { const n = parseNum(targets[w.isin]); if (n != null && n > 0) t[w.isin] = n / 100; }
    if (!Object.keys(t).length) return null;
    body.targets = t;
    return body;
  }
  const c = {};
  for (const h of book.held) { const n = parseNum(changes[h.isin]); if (n != null && !nearZero(n)) c[h.isin] = n; }
  // A ticked watchlist instrument enters the model even at 0: that shows what it
  // would do at the margin before any money goes in.
  for (const w of book.watch) if (include.has(w.isin)) { const n = parseNum(changes[w.isin]); c[w.isin] = n == null ? 0 : Math.max(n, 0); }
  if (!Object.keys(c).length) return null;
  body.changes = c;
  return body;
}

/** Input bookkeeping for the changes mode: net cash and the total after, clipped at zero per holding. */
function cashTotals(book, changes, include) {
  let total = 0, net = 0;
  for (const h of book.held) {
    const cur = h.value || 0;
    const after = Math.max(cur + (parseNum(changes[h.isin]) || 0), 0);
    total += after; net += after - cur;
  }
  for (const w of book.watch) if (include.has(w.isin)) { const n = Math.max(parseNum(changes[w.isin]) || 0, 0); total += n; net += n; }
  return { total, net };
}

/** Running sum of the target inputs that take part (held + ticked watchlist). */
function targetSum(book, targets, include) {
  let s = 0;
  for (const h of book.held) { const n = parseNum(targets[h.isin]); if (n != null && n > 0) s += n; }
  for (const w of book.watch) if (include.has(w.isin)) { const n = parseNum(targets[w.isin]); if (n != null && n > 0) s += n; }
  return s;
}

function tradesText(result, ccy) {
  if (!result || !result.trades.length) return "Nothing to trade.";
  const lines = result.trades.map((t) => [
    t.action.padEnd(4), fmt.money(t.amount, ccy), t.units == null ? "" : `${fmt.qty(t.units)} units`,
    t.price == null ? "" : `@ ${fmt.money(t.price, ccy)}`, t.name, `${fmt.pct(t.weight_before)} → ${fmt.pct(t.weight_after)}`,
  ].filter(Boolean).join("  "));
  const turnover = result.trades.reduce((s, t) => s + (t.amount || 0), 0);
  lines.push(`Turnover ${fmt.money(turnover, ccy)} · total ${fmt.money(result.total_before, ccy)} → ${fmt.money(result.total_after, ccy)}`);
  lines.push("Indicative: latest delayed prices, no fees, no rounding to whole units.");
  return lines.join("\n");
}

async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) { await navigator.clipboard.writeText(text); return true; }
  } catch (_) { /* fall through to the textarea path */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch (_) { return false; }
}

/* ---------------- controls (left column) ---------------- */

function Quick({ children, onClick, disabled, title }) {
  return html`<${Button} size="sm" variant="ghost" onClick=${onClick} disabled=${disabled} title=${title}>${children}<//>`;
}

/** Slider + numeric input pair. `zeroF` places a zero marker on a signed range. */
function RangeInput({ min, max, step, value, onValue, label, affix = {}, zeroF, isin, disabled }) {
  const n = parseNum(value);
  const v = clamp(n == null ? 0 : n, min, max);
  return html`<div class="sim-row-ctl">
    <div class="range-wrap" style=${zeroF != null ? `--zero-f:${zeroF}` : undefined}>
      <input type="range" class="range" min=${min} max=${max} step=${step} value=${v} disabled=${disabled}
        aria-label=${label} onInput=${(e) => onValue(String(e.target.value))} />
    </div>
    <${Input} numeric value=${value ?? ""} placeholder="0" ...${affix} disabled=${disabled} data-isin=${isin}
      aria-label=${label} onInput=${(e) => onValue(e.target.value)} />
  </div>`;
}

function RowIdentity({ row }) {
  return html`<div class="sim-row-id">
    <span class="sim-row-name" title=${row.name}>${row.name}</span>
    <span class="sim-row-sub">${[row.symbol, row.isin].filter(Boolean).join(" · ")}</span>
  </div>`;
}

function HoldingRow({ row, mode, value, onValue, ccy, weightAfter }) {
  const cur = row.value || 0;
  const head = html`<div class="sim-row-head">
    <${RowIdentity} row=${row} />
    <div class="sim-row-figs">${fmt.money(cur, ccy, { decimals: 0 })}<span class="w">${fmt.pct(row.weight)}</span></div>
  </div>`;

  if (mode === "presets") {
    return html`<div class="sim-row">
      <div class="sim-row-head">
        <${RowIdentity} row=${row} />
        <div class="sim-row-figs"><span class="sim-ba"><span class="b">${fmt.pct(row.weight)}</span><span class="arrow">→</span><span class="a">${weightAfter == null ? fmt.DASH : fmt.pct(weightAfter)}</span></span></div>
      </div>
    </div>`;
  }

  const n = parseNum(value);
  if (mode === "targets") {
    return html`<div class="sim-row">
      ${head}
      <${RangeInput} min=${0} max=${100} step=${0.5} value=${value} onValue=${onValue} isin=${row.isin} affix=${{ suffix: "%" }} label=${`Target weight for ${row.name}, percent`} />
      <div class="sim-quick">
        <${Quick} onClick=${() => onValue(pctStr(row.weight))} title="Back to the current weight">now<//>
        <${Quick} onClick=${() => onValue("0")} title="Sell out of this holding">0%<//>
        <span class="spacer"></span>
        <span class="sim-after">${n == null || nearZero(n - (row.weight || 0) * 100) ? "unchanged" : `from ${fmt.pct(row.weight)}`}</span>
      </div>
    </div>`;
  }

  const min = -Math.round(cur);
  const max = Math.max(Math.round(cur), SLIDER_MIN_MAX);
  const zeroF = (0 - min) / (max - min || 1);
  const after = Math.max(cur + (n || 0), 0);
  return html`<div class="sim-row">
    ${head}
    <${RangeInput} min=${min} max=${max} step=${10} value=${value} onValue=${onValue} isin=${row.isin} affix=${moneyAffix(ccy)} zeroF=${zeroF} label=${`Change in ${ccy} for ${row.name}`} />
    <div class="sim-quick">
      <${Quick} onClick=${() => onValue(String(-Math.round(cur * 0.25)))} title="Sell a quarter of the position">−25%<//>
      <${Quick} onClick=${() => onValue("1000")} title=${`Add ${fmt.money(1000, ccy, { decimals: 0 })}`}>+1,000<//>
      <${Quick} onClick=${() => onValue("5000")} title=${`Add ${fmt.money(5000, ccy, { decimals: 0 })}`}>+5,000<//>
      <${Quick} onClick=${() => onValue("")} disabled=${n == null} title="Clear this change">reset<//>
      <span class="spacer"></span>
      <span class="sim-after">${n == null || nearZero(n)
        ? "unchanged"
        : html`after ${fmt.money(after, ccy, { decimals: 0 })}${n < -cur ? html` · <span class="warn">sells everything</span>` : null}`}</span>
    </div>
  </div>`;
}

function WatchRow({ row, mode, checked, onCheck, value, onValue, ccy }) {
  const label = html`<span class="sim-row-name" title=${row.name}>${row.name}</span><span class="sim-row-sub">${[row.symbol, row.isin].filter(Boolean).join(" · ")}</span>`;
  const priceText = row.price == null ? fmt.DASH : fmt.money(row.price, row.currency || ccy);
  return html`<div class=${["sim-row", "sim-watch", checked ? "" : "is-off"].filter(Boolean).join(" ")}>
    <div class="sim-row-head">
      <${Checkbox} checked=${checked} onChange=${onCheck} label=${label} />
      <div class="sim-row-figs">${priceText}<span class="w">price</span></div>
    </div>
    ${checked && mode === "changes" ? html`<${RangeInput} min=${0} max=${SLIDER_MIN_MAX} step=${10} value=${value} onValue=${onValue} isin=${row.isin} affix=${moneyAffix(ccy)} label=${`Amount to buy of ${row.name}, in ${ccy}`} />` : null}
    ${checked && mode === "targets" ? html`<${RangeInput} min=${0} max=${100} step=${0.5} value=${value} onValue=${onValue} isin=${row.isin} affix=${{ suffix: "%" }} label=${`Target weight for ${row.name}, percent`} />` : null}
    ${checked && mode === "changes" && parseNum(value) == null ? html`<div class="sim-after" style="margin-top:4px">In the model at zero: the marginal figure shows what a first euro here would do.</div>` : null}
  </div>`;
}

function PresetPicker({ value, onChange }) {
  const onKey = (e, i) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp" && e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const dir = e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : -1;
    const next = PRESETS[(i + dir + PRESETS.length) % PRESETS.length];
    onChange(next.value);
    const btn = e.currentTarget.parentElement.querySelector(`[data-preset="${next.value}"]`);
    if (btn) btn.focus();
  };
  return html`<div class="sim-presets" role="radiogroup" aria-label="Preset">
    ${PRESETS.map((p, i) => html`<button key=${p.value} type="button" role="radio" class="sim-preset" data-preset=${p.value}
        aria-checked=${value === p.value ? "true" : "false"} tabindex=${value === p.value || (!value && i === 0) ? 0 : -1}
        onClick=${() => onChange(p.value)} onKeyDown=${(e) => onKey(e, i)}>
      <span class="sim-radio" aria-hidden="true"></span>
      <span class="stack" style="gap:0;min-width:0">
        <span class="sim-preset-title">${p.label}</span>
        <span class="sim-preset-body">${p.body}</span>
      </span>
    </button>`)}
  </div>`;
}

/* ---------------- results (right column) ---------------- */

function CompareTile({ label, help, before, after, deltaText, good, neutral = false, sub }) {
  const cls = deltaText == null ? "flat" : neutral || good == null ? "" : good ? "pos" : "neg";
  return html`<div class="card sim-kpi">
    <div class="sim-kpi-label">${label}${help ? html`<${Help} text=${help} />` : null}</div>
    <div class="sim-kpi-row">
      <span class="sim-kpi-before">${before}</span>
      <span class="sim-kpi-arrow" aria-hidden="true">→</span>
      <span class="sim-kpi-after">${after}</span>
    </div>
    <div class=${["sim-kpi-delta", cls].filter(Boolean).join(" ")}>
      <span>${deltaText == null ? "unchanged" : deltaText}</span>
      ${sub ? html`<span class="sim-kpi-sub">${sub}</span>` : null}
    </div>
  </div>`;
}

function Tiles({ result, ccy }) {
  const delta = (a, b) => (nearZero(a - b) ? null : a - b);
  const dVol = delta(result.volatility_after, result.volatility_before);
  const dEff = delta(result.effective_after, result.effective_before);
  const dDiv = delta(result.diversification_after, result.diversification_before);
  const dMax = delta(result.max_risk_share_after, result.max_risk_share_before);
  const dTot = delta(result.total_after, result.total_before);
  return html`<div class="sim-kpis" role="list">
    <${CompareTile} label="Volatility" help="Annualised standard deviation of daily portfolio returns, from the covariance of the risk window. Lower is better."
      before=${fmt.pct(result.volatility_before)} after=${fmt.pct(result.volatility_after)}
      deltaText=${dVol == null ? null : fmt.pp(dVol)} good=${dVol == null ? null : dVol < 0} sub="lower is better" />
    <${CompareTile} label="Effective holdings" help="1 / Σ weight². Six equal holdings count as 6; one dominant holding pulls it toward 1. Higher is better."
      before=${fmt.num(result.effective_before, { decimals: 1 })} after=${fmt.num(result.effective_after, { decimals: 1 })}
      deltaText=${dEff == null ? null : fmt.num(dEff, { decimals: 2, signed: true })} good=${dEff == null ? null : dEff > 0} sub="higher is better" />
    <${CompareTile} label="Diversification ratio" help="Weighted average of standalone volatilities divided by portfolio volatility. 1.0× means the holdings move as one. Higher is better."
      before=${fmt.mult(result.diversification_before)} after=${fmt.mult(result.diversification_after)}
      deltaText=${dDiv == null ? null : `${fmt.num(dDiv, { decimals: 2, signed: true })}×`} good=${dDiv == null ? null : dDiv > 0} sub="higher is better" />
    <${CompareTile} label="Largest risk share" help="The biggest single holding's share of portfolio volatility. Lower is better."
      before=${fmt.pct(result.max_risk_share_before)} after=${fmt.pct(result.max_risk_share_after)}
      deltaText=${dMax == null ? null : fmt.pp(dMax)} good=${dMax == null ? null : dMax < 0} sub="lower is better" />
    <${CompareTile} label="Total value" help="Market value of the modelled book. Money in or out is neither good nor bad." neutral
      before=${fmt.money(result.total_before, ccy, { compact: true })} after=${fmt.money(result.total_after, ccy, { compact: true })}
      deltaText=${dTot == null ? null : fmt.money(dTot, ccy, { signed: true, decimals: 0 })} sub=${dTot == null ? "rebalance only" : dTot > 0 ? "added" : "withdrawn"} />
  </div>`;
}

function measureSeries(rows, measure, marginalBefore) {
  if (measure === "weight") return { before: rows.map((r) => r.weight_before), after: rows.map((r) => r.weight_after) };
  if (measure === "marginal") return { before: rows.map((r) => (marginalBefore.has(r.isin) ? marginalBefore.get(r.isin) : null)), after: rows.map((r) => r.marginal_after) };
  return { before: rows.map((r) => r.risk_before), after: rows.map((r) => r.risk_after) };
}

/** Paired horizontal bars, Before (quiet gray) vs After (accent): 10px bars with a
    2px surface gap inside the pair and air between pairs; legend on (two series);
    the After bar carries the only direct label; per-mark tooltip lists both and the change. */
function pairedOption(rows, measure, marginalBefore) {
  const t = tokens();
  const { before, after } = measureSeries(rows, measure, marginalBefore);
  const opt = barOption({
    categories: rows.map((r) => fmt.shortName(r.name, 26)),
    horizontal: true, format: "pct",
    series: [{ name: "Before", values: before, color: t.text3 }, { name: "After", values: after, color: t.accent }],
  });
  const [sBefore, sAfter] = opt.series;
  for (const s of opt.series) { s.barWidth = 10; s.barMaxWidth = 10; s.barGap = "20%"; s.barCategoryGap = "40%"; }
  // barOption colours per data item; a series-level colour keeps the legend in agreement.
  sBefore.itemStyle = { color: t.text3 };
  sAfter.itemStyle = { color: t.accent };
  sAfter.label = { show: true, position: "right", color: t.text2, fontSize: 11, formatter: (p) => fmt.pct(p.value) };
  opt.grid.right = 52;
  opt.xAxis.max = (v) => { const m = Math.ceil((v.max + 0.02) * 20) / 20; return measure === "marginal" ? m : Math.min(1, m); };
  opt.tooltip.formatter = (p) => {
    const i = p.dataIndex;
    const r = rows[i];
    const b = before[i], a = after[i];
    const d = b == null || a == null ? null : a - b;
    const polarity = measure === "weight" || d == null || nearZero(d) ? "" : d > 0 ? "neg" : "pos";
    return tipElement(r ? r.name : p.name, [
      { name: "Before", value: fmt.pct(b), color: t.text3, kind: "swatch" },
      { name: "After", value: fmt.pct(a), color: t.accent, kind: "swatch" },
      { name: "Change", value: fmt.pp(d), polarity },
    ]);
  };
  return opt;
}

function pairedTable(rows, measure, marginalBefore) {
  const { before, after } = measureSeries(rows, measure, marginalBefore);
  const label = MEASURES.find((m) => m.value === measure).label;
  return {
    columns: [
      { key: "name", label: "Holding", sortable: true },
      { key: "before", label: `${label} before`, format: "pct" },
      { key: "after", label: `${label} after`, format: "pct" },
      { key: "delta", label: "Change", format: "pp" },
    ],
    rows: rows.map((r, i) => ({ id: r.isin, name: r.name, before: before[i], after: after[i], delta: before[i] == null || after[i] == null ? null : after[i] - before[i] })),
  };
}

function PairedChart({ rows, marginalBefore, measure, onMeasure }) {
  const theme = useStore((s) => s.theme);
  const height = Math.max(180, rows.length * 36 + 64);
  const table = useMemo(() => pairedTable(rows, measure, marginalBefore), [rows, measure, marginalBefore]);
  return html`<${Chart} title="Per holding, before → after" caption=${MEASURE_CAPTION[measure]} height=${height} table=${table}
    actions=${html`<${Segmented} size="sm" ariaLabel="Measure" value=${measure} onChange=${onMeasure} options=${MEASURES} />`}
    deps=${[rows, measure, marginalBefore, theme]} empty="No holdings in the model"
    buildOption=${() => (rows.length ? pairedOption(rows, measure, marginalBefore) : null)} />`;
}

/** "before → after" cell with an optional delta line beneath. */
function baCell(before, after, deltaText, deltaClass = "") {
  return html`<span class="sim-ba"><span class="b">${before}</span><span class="arrow" aria-hidden="true">→</span><span class="a">${after}</span></span>
    ${deltaText ? html`<span class=${["cell-sub", deltaClass].filter(Boolean).join(" ")}>${deltaText}</span>` : null}`;
}

function HoldingsTable({ rows, ccy }) {
  const columns = useMemo(() => {
    const money0 = (v) => fmt.money(v, ccy, { decimals: 0 });
    return [
      { key: "name", label: "Instrument", primary: true, render: (r, v) => html`<div class="truncate" style="max-width:240px" title=${v}>${v}<span class="cell-sub">${r.isin}</span></div>` },
      { key: "value_after", label: "Value", numeric: true, title: "Market value before → after",
        render: (r) => { const d = r.value_after - r.value_before; return baCell(money0(r.value_before), money0(r.value_after), nearZero(d) ? null : fmt.money(d, ccy, { signed: true, decimals: 0 }), "muted"); } },
      { key: "weight_after", label: "Weight", numeric: true, title: "Share of value before → after",
        render: (r) => { const d = r.weight_after - r.weight_before; return baCell(fmt.pct(r.weight_before), fmt.pct(r.weight_after), nearZero(d) ? null : fmt.pp(d), "muted"); } },
      { key: "risk_after", label: "Risk share", numeric: true, title: "Share of portfolio volatility before → after; a rising share is coloured as the direction worth a look",
        render: (r) => { const d = r.risk_after - r.risk_before; return baCell(fmt.pct(r.risk_before), fmt.pct(r.risk_after), nearZero(d) ? null : fmt.pp(d), nearZero(d) ? "" : d > 0 ? "neg" : "pos"); } },
      { key: "marginal_after", label: html`MCTR after <${Help} text="Marginal contribution to risk, annualised: how much portfolio volatility moves per extra euro here." />`, numeric: true, format: "pct" },
    ];
  }, [ccy]);
  return html`<${DataTable} class="sim-table" columns=${columns} rows=${rows} rowKey="isin" sort=${{ key: "value_after", dir: "desc" }} empty="No holdings in the model" caption="Per holding, before and after" />`;
}

function Trades({ result, ccy, minTrade }) {
  const trades = result.trades || [];
  const buys = trades.filter((t) => t.action === "BUY").reduce((s, t) => s + t.amount, 0);
  const sells = trades.filter((t) => t.action === "SELL").reduce((s, t) => s + t.amount, 0);
  const turnover = buys + sells;
  const minTradeN = Math.max(0, parseNum(minTrade) || 0);
  const columns = useMemo(() => [
    { key: "action", label: "Action", width: 76, render: (r, v) => html`<${TxTypeBadge} type=${v} />` },
    { key: "name", label: "Instrument", primary: true, render: (r, v) => html`<div class="truncate" style="max-width:260px" title=${v}>${v}<span class="cell-sub">${r.isin}</span></div>` },
    { key: "amount", label: "Amount", numeric: true, format: (v) => fmt.money(v, ccy) },
    { key: "units", label: "Units", numeric: true, title: "Amount at the latest price; blank when no price is known",
      render: (r, v) => (v == null
        ? html`<span class="faint" title="No price is known, so the amount cannot be turned into units">${fmt.DASH}</span>`
        : html`${fmt.qty(v)}<span class="cell-sub">@ ${fmt.money(r.price, ccy)}</span>`) },
    { key: "weight_after", label: "Weight", numeric: true, title: "Share of value before → after", render: (r) => baCell(fmt.pct(r.weight_before), fmt.pct(r.weight_after)) },
  ], [ccy]);
  const onCopy = async () => {
    const ok = await copyText(tradesText(result, ccy));
    if (ok) toast.success("Trade list copied as text"); else toast.error("Could not copy to the clipboard");
  };
  const footer = html`
    ${trades.length ? html`<div class="sim-turnover">
      <span class="stat-inline"><span class="stat-label">Turnover</span><span class="stat-value">${fmt.money(turnover, ccy)}</span></span>
      <span class="stat-inline"><span class="stat-label">Buys</span><span class="stat-value">${fmt.money(buys, ccy)}</span></span>
      <span class="stat-inline"><span class="stat-label">Sells</span><span class="stat-value">${fmt.money(sells, ccy)}</span></span>
      <span class="stat-inline"><span class="stat-label">Turnover / book</span><span class="stat-value">${fmt.pct(result.total_before ? turnover / result.total_before : null)}</span></span>
    </div>` : null}
    <div class="sim-note">Indicative only: amounts at the latest delayed price, no fees, no rounding to whole units${minTradeN > 0 ? `, trades under ${fmt.money(minTradeN, ccy, { decimals: 0 })} left out` : ""}. Nothing on this page is written to the ledger — the server never writes anything from the simulator.</div>`;
  return html`<${Card} class="sim-trades" title="Trades to get there" flush footer=${footer}
    caption=${trades.length ? `${fmt.count(trades.length, "trade")} · amounts in ${ccy}` : "The buys and sells that move the book from before to after."}
    actions=${trades.length ? html`<${Button} size="sm" variant="ghost" icon="copy" onClick=${onCopy}>Copy as text<//>` : null}>
    ${trades.length
      ? html`<${DataTable} class="sim-table" columns=${columns} rows=${trades} rowKey=${(r) => `${r.isin}:${r.action}`} sort=${{ key: "amount", dir: "desc" }} caption="Trades" />`
      : html`<${EmptyState} compact icon="check" title="Nothing to trade" body="The after-state matches the book within the minimum trade size." />`}
  <//>`;
}

function Results({ result, pending, ccy, marginalBefore, measure, onMeasure, minTrade }) {
  const rows = result.holdings || [];
  return html`<div class=${["sim-results", pending ? "is-pending" : ""].filter(Boolean).join(" ")} aria-busy=${pending ? "true" : "false"}>
    <${Tiles} result=${result} ccy=${ccy} />
    <${PairedChart} rows=${rows} marginalBefore=${marginalBefore} measure=${measure} onMeasure=${onMeasure} />
    <${Card} title="Per holding" caption=${`Values in ${ccy}. Risk share is the share of portfolio volatility; MCTR is its derivative per extra euro.`} flush>
      <${HoldingsTable} rows=${rows} ccy=${ccy} />
    <//>
    <${Trades} result=${result} ccy=${ccy} minTrade=${minTrade} />
  </div>`;
}

/* ---------------- page ---------------- */

export default function SimulatorPage({ route, snapshot }) {
  const book = useMemo(() => bookRows(snapshot), [snapshot]);
  const ccy = (snapshot && snapshot.meta && snapshot.meta.base_currency) || "EUR";
  const risk = snapshot && snapshot.risk;

  const mem = remembered || {};
  const [mode, setMode] = useState(mem.mode || "changes");
  const [changes, setChanges] = useState(mem.changes || {});
  const [targets, setTargets] = useState(mem.targets || null);   // null = start from the current weights
  const [preset, setPreset] = useState(mem.preset || null);
  const [include, setInclude] = useState(() => new Set(mem.include || []));
  const [minTrade, setMinTrade] = useState(mem.minTrade || "");
  const [measure, setMeasure] = useState(mem.measure || "risk");
  const [result, setResult] = useState(mem.result || null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const seq = useRef(0);
  const ctrl = useRef(null);

  // Targets start from the current weights so an untouched row keeps its money
  // (the server treats a missing target as 0%).
  const currentTargets = useMemo(() => {
    const t = {};
    for (const h of book.held) if (h.weight != null) t[h.isin] = pctStr(h.weight);
    return t;
  }, [book]);
  const targetsEff = targets || currentTargets;

  useEffect(() => {
    remembered = { mode, changes, targets, preset, include: Array.from(include), minTrade, measure, result };
  }, [mode, changes, targets, preset, include, minTrade, measure, result]);

  const body = useMemo(() => buildRequest({ mode, changes, targets: targetsEff, preset, include, minTrade, book }),
    [mode, changes, targetsEff, preset, include, minTrade, book]);
  const requestKey = body ? JSON.stringify(body) : "";

  const run = (payload) => {
    if (ctrl.current) ctrl.current.abort();
    const controller = new AbortController();
    ctrl.current = controller;
    const id = ++seq.current;
    setPending(true);
    api.post("/simulate", payload, { signal: controller.signal }).then((res) => {
      if (id !== seq.current) return;
      setResult(res); setError(null); setPending(false);
    }).catch((err) => {
      if (err && err.name === "AbortError") return;
      if (id !== seq.current) return;
      setError({ status: err && err.status ? err.status : 0, message: errorMessage(err) });
      setPending(false);
    });
  };

  // Debounced request on every change of the request body; nothing to ask clears the answer.
  useEffect(() => {
    if (!requestKey) {
      seq.current++;
      if (ctrl.current) ctrl.current.abort();
      setPending(false); setResult(null); setError(null);
      return undefined;
    }
    const h = setTimeout(() => run(JSON.parse(requestKey)), DEBOUNCE_MS);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey]);

  useEffect(() => () => { seq.current++; if (ctrl.current) ctrl.current.abort(); }, []);

  // ?isin= (from a holding's drawer) pre-focuses that instrument's input, admitting it first if it is on the watchlist.
  const focusIsin = route && route.params && route.params.isin ? String(route.params.isin).toUpperCase() : null;
  const focused = useRef(null);
  useEffect(() => {
    if (!focusIsin || !snapshot || focused.current === focusIsin) return undefined;
    focused.current = focusIsin;
    if (book.watch.some((w) => w.isin === focusIsin)) setInclude((s) => (s.has(focusIsin) ? s : new Set([...s, focusIsin])));
    if (mode === "presets") setMode("changes");
    const h = setTimeout(() => {
      const el = document.querySelector(`input[data-isin="${focusIsin}"]`);
      if (el) { el.focus(); el.scrollIntoView({ block: "center" }); }
    }, 80);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusIsin, snapshot]);

  const setChange = (isin, v) => setChanges((c) => ({ ...c, [isin]: v }));
  const setTarget = (isin, v) => setTargets((t) => ({ ...(t || currentTargets), [isin]: v }));
  const toggleInclude = (isin, on) => setInclude((s) => { const n = new Set(s); if (on) n.add(isin); else n.delete(isin); return n; });
  const reset = () => { setChanges({}); setTargets(null); setPreset(null); setInclude(new Set()); setMinTrade(""); setError(null); };
  const normalise = () => {
    const keys = [...book.held.map((h) => h.isin), ...book.watch.filter((w) => include.has(w.isin)).map((w) => w.isin)];
    const sum = targetSum(book, targetsEff, include);
    if (sum <= 0) return;
    const next = { ...targetsEff };
    for (const k of keys) { const n = parseNum(next[k]); next[k] = n != null && n > 0 ? pctStr(n / sum) : (n == null ? "" : "0"); }
    setTargets(next);
  };

  const cash = useMemo(() => (mode === "changes" ? cashTotals(book, changes, include) : null), [mode, book, changes, include]);
  const tSum = useMemo(() => (mode === "targets" ? targetSum(book, targetsEff, include) : null), [mode, book, targetsEff, include]);
  const sumOff = tSum != null && Math.abs(tSum - 100) > 0.05;
  const admitted = book.watch.filter((w) => include.has(w.isin)).length;
  const bookValue = snapshot && snapshot.totals ? snapshot.totals.value : null;

  if (!snapshot) {
    return html`<${Page} title="Simulator" subtitle=${SUBTITLE}>
      <${Card}><${EmptyState} icon="simulator" title="Loading the book…" body="The simulator needs the snapshot's holdings and risk model." /><//>
    <//>`;
  }
  if (!risk || !risk.available || !book.held.length) {
    return html`<${Page} title="Simulator" subtitle=${SUBTITLE}>
      <${Card}><${EmptyState} icon="risk" title="No risk model to simulate against"
        body=${html`<span>${(risk && risk.reason) || "The risk model is unavailable, so there is no covariance to project the book through."}</span><br/><span>The simulator needs at least two priced holdings with overlapping price history. Record the positions on the Holdings page and the model builds itself.</span>`}
        action=${html`<div class="row"><${Button} variant="primary" icon="holdings" href="#/holdings">Open Holdings<//><${Button} variant="ghost" href="#/risk">Risk page<//></div>`} /><//>
    <//>`;
  }

  const actions = html`<${Button} variant="secondary" icon="refresh" onClick=${reset} title="Clear every change and the answer">Reset<//>`;
  const errorBanner = !error ? null : error.status === 409
    ? html`<${Banner} kind="warn" title="No risk model right now." action=${html`<${Button} size="sm" variant="secondary" href="#/risk">Open Risk<//>`}>${error.message}<//>`
    : html`<${Banner} kind="error" title=${error.status === 400 ? "The server refused the request." : "The simulation failed."}
        action=${html`<${Button} size="sm" variant="secondary" onClick=${() => requestKey && run(JSON.parse(requestKey))}>Try again<//>`}>${error.message}<//>`;

  return html`<${Page} title="Simulator" subtitle=${SUBTITLE} actions=${actions}>
    <div class="sim-layout">
      <${Card} class="sim-controls" title="The book"
        caption=${`${fmt.money(bookValue, ccy, { compact: true })} · ${fmt.count(book.held.length, "holding")} in the model${admitted ? ` · ${admitted} from the watchlist` : ""}`}>
        <${Tabs} class="sim-tabs" tabs=${MODES} active=${mode} onChange=${setMode} />
        <div class="sim-mode-help">${MODE_HELP[mode]}</div>
        ${mode === "presets" ? html`<${PresetPicker} value=${preset} onChange=${setPreset} />` : null}

        <div class="sim-section"><span>Holdings</span><span class="sim-section-note">${mode === "presets" ? "weight now → after" : "value · weight now"}</span></div>
        <div class="sim-rows" role="group" aria-label="Held instruments">
          ${book.held.map((h) => html`<${HoldingRow} key=${h.isin} row=${h} mode=${mode} ccy=${ccy}
            value=${mode === "targets" ? targetsEff[h.isin] : changes[h.isin]}
            onValue=${(v) => (mode === "targets" ? setTarget(h.isin, v) : setChange(h.isin, v))}
            weightAfter=${result && result.weights_after ? result.weights_after[h.isin] : null} />`)}
        </div>
        ${book.excluded.length ? html`<div class="sim-excluded">Not in the model: ${book.excluded.map((e, i) => html`<span key=${e.isin}>${i ? "; " : ""}<span title=${e.reason} class="tip-inline">${fmt.shortName(e.name, 28)}</span> (${e.reason})</span>`)}.</div>` : null}

        ${book.watch.length ? html`
          <div class="sim-section"><span>Watchlist — add to the model</span><span class="sim-section-note">tick to admit</span></div>
          <div class="sim-rows" role="group" aria-label="Watchlist instruments">
            ${book.watch.map((w) => html`<${WatchRow} key=${w.isin} row=${w} mode=${mode} ccy=${ccy}
              checked=${include.has(w.isin)} onCheck=${(on) => toggleInclude(w.isin, on)}
              value=${mode === "targets" ? targetsEff[w.isin] : changes[w.isin]}
              onValue=${(v) => (mode === "targets" ? setTarget(w.isin, v) : setChange(w.isin, v))} />`)}
          </div>` : null}

        <div class="sim-foot">
          ${mode === "changes" && cash ? html`
            <div class="sim-sum"><span>Net cash</span><span class="num">${fmt.money(cash.net, ccy, { signed: true, decimals: 0 })}</span></div>
            <div class="sim-sum"><span>Total after</span><span class="num">${fmt.money(cash.total, ccy, { decimals: 0 })}<span class="faint">from ${fmt.money(bookValue, ccy, { decimals: 0 })}</span></span></div>` : null}
          ${mode === "targets" ? html`
            <div class=${["sim-sum", sumOff ? "is-warn" : ""].filter(Boolean).join(" ")}>
              <span>Sum of targets <${Help} text="Targets are rescaled to 100% by the server, so proportions are what count. Normalise rewrites the inputs so they read as they will be used." /></span>
              <span class="num">${fmt.num(tSum, { decimals: 1 })}%${sumOff ? html`<${Button} size="sm" variant="secondary" onClick=${normalise} disabled=${!(tSum > 0)}>Normalise<//>` : html`<span class="pos" title="Sums to 100%" aria-label="Sums to 100%">✓</span>`}</span>
            </div>
            <div class="sim-sum"><span>Total after</span><span class="num">${fmt.money(bookValue, ccy, { decimals: 0 })}<span class="faint">held constant</span></span></div>` : null}
          ${mode === "presets" ? html`<div class="sim-sum"><span>Total after</span><span class="num">${fmt.money(bookValue, ccy, { decimals: 0 })}<span class="faint">a preset only redistributes</span></span></div>` : null}
          <${Field} label=${html`Minimum trade <${Help} text="Trades smaller than this are left out of the trade list. The after-state itself is unchanged." />`} id="sim-min-trade" hint="Drops dust trades from the list.">
            <${Input} id="sim-min-trade" numeric value=${minTrade} placeholder="0" ...${moneyAffix(ccy)} onInput=${(e) => setMinTrade(e.target.value)} />
          <//>
          <div class="sim-status" aria-live="polite">
            ${pending ? html`<span class="spinner" aria-hidden="true"></span><span>Simulating…</span>` : requestKey ? html`<span>Re-runs ${DEBOUNCE_MS} ms after you stop typing.</span>` : html`<span>Runs as soon as you change something.</span>`}
          </div>
          ${errorBanner}
        </div>
      <//>

      <div>
        ${result ? html`<${Results} result=${result} pending=${pending} ccy=${ccy} marginalBefore=${book.marginalBefore} measure=${measure} onMeasure=${setMeasure} minTrade=${minTrade} />`
        : html`<${Card}><${EmptyState} icon="simulator"
            title=${pending ? "Simulating…" : error ? "No answer yet" : "Change something to see the before → after"}
            body=${error ? "The last request was refused; the message is on the left." : mode === "presets" ? "Pick Equal weight, Risk parity or Minimum variance on the left." : mode === "targets" ? "Move a target weight on the left; the answer appears here a moment later." : "Add or remove money next to a holding on the left; the answer appears here a moment later."}
            action=${mode === "presets" || error ? null : html`<${Button} variant="secondary" onClick=${() => { setMode("presets"); if (!preset) setPreset("risk_parity"); }}>Try risk parity<//>`} /><//>`}
      </div>
    </div>
  <//>`;
}
