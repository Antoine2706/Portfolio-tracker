/* Kitchen sink — #/kitchensink. Every component and chart builder rendered
   from the LIVE snapshot (no mock data). This is the team's visual regression
   page: open it in dark and light at 1440 and 1024 after touching anything
   under components/, lib/ or styles/. Keep it. */

import { html, useState, useMemo } from "/static/vendor/preact-htm.module.js";
import { Page, Section } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { Button } from "/static/components/Button.js";
import { Badge, SeverityBadge, VerdictBadge, TxTypeBadge, ModeBadge } from "/static/components/Badge.js";
import { Banner } from "/static/components/Banner.js";
import { Chart, ChartBody } from "/static/components/Chart.js";
import { DataTable, colorCell, nameCell } from "/static/components/DataTable.js";
import { confirm } from "/static/components/Modal.js";
import { toast } from "/static/components/Toast.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Field, Input, Select, Textarea, Checkbox } from "/static/components/Field.js";
import { KpiTile } from "/static/components/KpiTile.js";
import { Notice } from "/static/components/Notice.js";
import { Segmented, RangeSelector } from "/static/components/Segmented.js";
import { Sparkline } from "/static/components/Sparkline.js";
import { Tabs } from "/static/components/Tabs.js";
import { Tooltip, Help } from "/static/components/Tooltip.js";
import { Icon, ICON_NAMES } from "/static/components/Icons.js";
import { openDrawer, closeDrawer, openModal, closeModal, useStore } from "/static/lib/store.js";
import { navigate } from "/static/lib/router.js";
import * as charts from "/static/lib/charts.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- helpers ---------------- */

function Bomb({ armed }) {
  if (armed) throw new Error("Kitchen sink: deliberate render error to exercise the global error boundary.");
  return null;
}

function Swatch({ name }) {
  return html`<div class="ks-swatch" title=${name}>
    <span class="ks-swatch-chip" style=${`background:var(${name})`}></span>
    <span class="xs mono muted">${name.replace(/^--/, "")}</span>
  </div>`;
}

function HoldingDrawerBody({ h }) {
  return html`<div class="stack gap-4">
    <div class="row between">
      <div class="row">
        <${Badge} kind="neutral" outline>${h.asset_class}<//>
        ${h.price_is_stale ? html`<${Badge} kind="warn" icon="clock">stale<//>` : null}
      </div>
      <div style="width:120px;height:32px"><${Sparkline} values=${h.sparkline || []} /></div>
    </div>
    <dl class="definition-list">
      <dt>Price</dt><dd class="num">${fmt.money(h.price, h.price_currency)} <span class="faint small">${fmt.dateTime(h.price_as_of)}</span></dd>
      <dt>Quantity</dt><dd class="num">${fmt.qty(h.quantity)}</dd>
      <dt>Value</dt><dd class="num">${fmt.money(h.value)}</dd>
      <dt>Unrealised</dt><dd class=${"num " + fmt.polarityClass(h.unrealised)}>${fmt.money(h.unrealised, "EUR", { signed: true })} (${fmt.pct(h.unrealised_pct, { signed: true })})</dd>
      <dt>Weight</dt><dd class="num">${fmt.pct(h.weight)}</dd>
      <dt>Risk share</dt><dd class="num">${fmt.pct(h.risk_share)}</dd>
    </dl>
    <p class="small muted">${h.price_note}</p>
    <${Button} variant="secondary" size="sm" icon="external" onClick=${() => { closeDrawer(); navigate(`/holdings/${h.isin}`); }}>Open the real drawer route<//>
  </div>`;
}

/* ---------------- sections ---------------- */

function KpiRow({ snapshot }) {
  const t = snapshot.totals;
  const r = snapshot.risk;
  const ccy = snapshot.meta.base_currency;
  const spark = (snapshot.holdings[0] && snapshot.holdings[0].sparkline) || [];
  return html`<div class="kpi-grid">
    <${KpiTile} size="hero" label="Portfolio value" value=${fmt.money(t.value, ccy)} delta=${t.day_change_pct} deltaLabel="today" sub=${fmt.count(t.priced_holdings, "priced holding")} help="Market value of every open position at the last price, in the base currency." />
    <${KpiTile} label="Unrealised P&L" value=${fmt.money(t.unrealised, ccy, { signed: true })} polarity=${fmt.polarityClass(t.unrealised)} delta=${t.unrealised_pct} deltaLabel="on cost" />
    <${KpiTile} label="Day change" value=${fmt.money(t.day_change, ccy, { signed: true })} polarity=${fmt.polarityClass(t.day_change)} delta=${t.day_change_pct} deltaFormat="pct" deltaLabel="vs previous close" sparkline=${spark} />
    <${KpiTile} label="Invested" value=${fmt.money(t.invested, ccy)} sub="net external flow" />
    <${KpiTile} label="Volatility" value=${fmt.pct(r.volatility)} delta=${r.volatility_multiple} deltaFormat=${(v) => `${fmt.mult(v)} a broad index`} upIsGood=${false} sub=${r.window ? `${r.window.effective}-day window` : null} />
    <${KpiTile} size="sm" label="Not available" value=${null} delta=${null} sub="null renders as a dash, never NaN" />
  </div>`;
}

function HoldingsTable({ snapshot }) {
  const ccy = snapshot.meta.base_currency;
  const columns = useMemo(() => [
    { key: "name", label: "Holding", primary: true, render: nameCell(), width: 280 },
    { key: "quantity", label: "Units", format: "qty" },
    { key: "price", label: "Price", numeric: true, format: (v, r) => fmt.money(v, r.price_currency), sub: (r) => (r.price_is_stale ? "stale" : null) },
    { key: "value", label: "Value", format: "money" },
    { key: "weight", label: "Weight", format: "pct" },
    { key: "unrealised", label: "Unrealised", numeric: true, render: colorCell("money-signed") },
    { key: "unrealised_pct", label: "Return", numeric: true, render: colorCell("pct-signed") },
    { key: "day_change_pct", label: "Day", numeric: true, render: colorCell("pct-signed") },
    { key: "risk_share", label: "Risk share", format: "pct", title: "Share of portfolio variance" },
    { key: "divergence", label: "Divergence", format: "pp" },
    { key: "sparkline", label: "30 d", align: "center", sortable: false, render: (r) => (r.sparkline && r.sparkline.length > 1 ? html`<span class="cell-spark"><${Sparkline} values=${r.sparkline} /></span>` : fmt.DASH) },
  ], []);
  const rows = snapshot.holdings;
  const footer = { name: "Total", value: snapshot.totals.value, weight: 1, unrealised: snapshot.totals.unrealised, unrealised_pct: snapshot.totals.unrealised_pct, day_change_pct: snapshot.totals.day_change_pct, risk_share: 1 };
  const [selected, setSelected] = useState(null);
  const onRow = (h) => {
    setSelected(h.isin);
    openDrawer({ title: fmt.shortName(h.name, 40), subtitle: [h.symbol, h.isin].filter(Boolean).join(" · "), badge: html`<${Badge} kind="info">drawer<//>`,
      body: html`<${HoldingDrawerBody} h=${h} />`, footer: html`<${Button} variant="secondary" onClick=${closeDrawer}>Close<//>`, onClose: () => setSelected(null) });
  };
  return html`<${Card} title="Holdings" caption="Client-side sort, sticky header, tabular figures, row click opens a drawer, Enter/Space on a focused row." flush>
    <${DataTable} columns=${columns} rows=${rows} rowKey="isin" sort=${{ key: "value", dir: "desc" }} onRowClick=${onRow} selectedKey=${selected}
      densityToggle toolbar=${html`<span class="muted">${fmt.count(rows.length, "position")} · ${ccy}</span>`} footer=${footer} caption="Holdings" />
  <//>`;
}

function FormatTable() {
  const inputs = [null, undefined, NaN, 0, -0.0123, 0.4405, 3.518, 1234.5, -76363.0078, 1.5e6, "2026-09-04", "2026-09-04T17:30+00:00", "not a date"];
  const columns = [
    { key: "input", label: "Input", sortable: false, render: (r) => html`<span class="mono xs">${String(r.input)}</span>` },
    { key: "money", label: "money", format: "money" },
    { key: "money-compact", label: "compact", format: "money-compact" },
    { key: "money-signed", label: "signed", format: "money-signed" },
    { key: "pct", label: "pct", format: "pct-signed" },
    { key: "pp", label: "pp", format: "pp" },
    { key: "num", label: "num", format: "num" },
    { key: "int", label: "int", format: "int" },
    { key: "mult", label: "mult", format: "mult" },
    { key: "date", label: "date", format: "date" },
    { key: "dateTime", label: "dateTime", format: "dateTime" },
    { key: "relative", label: "relative", format: "relative" },
  ];
  const rows = inputs.map((v, i) => ({ id: i, input: v, money: v, "money-compact": v, "money-signed": v, pct: v, pp: v, num: v, int: v, mult: v, date: v, dateTime: v, relative: v }));
  return html`<${Card} title="lib/format.js" caption="Every helper is null-safe: null, undefined, NaN and unparsable input render an em dash, never NaN." flush>
    <${DataTable} columns=${columns} rows=${rows} />
  <//>`;
}

function ChartGrid({ snapshot }) {
  const p = snapshot.performance;
  const r = snapshot.risk;
  const ccy = snapshot.meta.base_currency;
  const bench = (snapshot.benchmarks || []).find((b) => b.symbol === snapshot.selected_benchmark);
  const benchName = bench ? bench.index : snapshot.selected_benchmark;
  const [range, setRange] = useState("ALL");

  // line: portfolio index vs benchmark index
  const lineTable = useMemo(() => charts.seriesTable({ dates: p.dates, series: [{ name: "Portfolio", values: p.index }, { name: benchName, values: p.benchmark_index }], format: "index" }), [p, benchName]);
  const lineOpt = () => (p.available ? charts.lineOption({
    dates: p.dates, yFormat: "index", currency: ccy, baseline: 100,
    series: [{ name: "Portfolio", values: p.index }, { name: benchName, values: p.benchmark_index, width: 1.5 }],
  }) : null);

  // diverging: risk share minus weight
  const div = r.divergence || [];
  const divNames = div.map((d) => fmt.shortName(d.name, 26));
  const divOpt = () => (div.length ? charts.divergingBarOption({ names: divNames, values: div.map((d) => d.divergence), sentences: div.map((d) => d.sentence) }) : null);
  const divTable = charts.categoryTable({ categories: divNames, values: div.map((d) => d.divergence), format: "pp", label: "Holding", valueLabel: "Risk − weight",
    extra: [{ key: "weight", label: "Weight", format: "pct" }, { key: "risk", label: "Risk share", format: "pct" }] });
  divTable.rows.forEach((row, i) => { row.weight = div[i].weight; row.risk = div[i].risk_share; });

  // heatmap: correlation matrix
  const corr = r.correlation;
  const corrX = corr ? corr.names.map((n) => fmt.shortName(n, 14)) : [];
  const corrY = corr ? corr.names.map((n) => fmt.shortName(n, 26)) : [];
  const heatOpt = () => (corr ? charts.heatmapOption({ xLabels: corrX, yLabels: corrY, matrix: corr.matrix, min: -1, max: 1 }) : null);
  const heatTable = corr ? charts.matrixTable({ xLabels: corrX, yLabels: corrY, matrix: corr.matrix, cornerLabel: "ρ" }) : null;

  // treemap: holdings by value
  const items = snapshot.holdings.filter((h) => h.value != null).map((h) => ({ name: fmt.shortName(h.name, 30), value: h.value, sub: `${h.symbol || ""} · ${h.isin}` }));
  const treeOpt = () => (items.length ? charts.treemapOption({ items: charts.foldOther(items), currency: ccy }) : null);
  const treeTable = charts.categoryTable({ categories: items.map((i) => i.name), values: items.map((i) => i.value), format: "money", currency: ccy, label: "Holding", valueLabel: "Value" });

  // drawdown
  const ddOpt = () => (p.available ? charts.drawdownOption(p.dates, p.drawdown, { benchmark: p.benchmark_drawdown, benchmarkName: benchName }) : null);
  const ddTable = charts.seriesTable({ dates: p.dates, series: [{ name: "Portfolio", values: p.drawdown }, { name: benchName, values: p.benchmark_drawdown }], format: "pct" });

  // monthly heatmap
  const years = [...new Set((p.monthly || []).map((m) => m.year))].sort();
  const monthly = years.map((y) => Array.from({ length: 12 }, (_, m) => { const row = p.monthly.find((x) => x.year === y && x.month === m + 1); return row ? row.ret : null; }));
  const monthOpt = () => (years.length ? charts.monthlyHeatmapOption(years, fmt.MONTHS_SHORT, monthly) : null);
  const monthTable = charts.matrixTable({ xLabels: fmt.MONTHS_SHORT, yLabels: years.map(String), matrix: monthly, format: (v) => fmt.pct(v, { signed: true }), cornerLabel: "Year" });

  // bars: exposure by issuer (one series) and yearly returns vs benchmark (two series)
  const issuers = (snapshot.exposure.issuer || []);
  const barOpt = () => (issuers.length ? charts.barOption({ categories: issuers.map((s) => s.label), values: issuers.map((s) => s.weight), horizontal: true, format: "pct", sort: "desc", showLabels: true }) : null);
  const barTable = charts.categoryTable({ categories: issuers.map((s) => s.label), values: issuers.map((s) => s.weight), format: "pct", label: "Issuer", valueLabel: "Weight" });
  const yearly = p.yearly || [];
  const yBarOpt = () => (yearly.length ? charts.barOption({ categories: yearly.map((y) => String(y.year)), format: "pct-signed",
    series: [{ name: "Portfolio", values: yearly.map((y) => y.ret) }, { name: benchName, values: yearly.map((y) => y.benchmark_ret) }] }) : null);
  const yBarTable = charts.seriesTable({ dates: yearly.map((y) => String(y.year)), series: [{ name: "Portfolio", values: yearly.map((y) => y.ret) }, { name: benchName, values: yearly.map((y) => y.benchmark_ret) }], format: "pct-signed", dateLabel: "Year" });

  const deps = [snapshot];
  return html`<div class="grid">
    <div class="col-8"><${Chart} title="Portfolio vs benchmark" caption=${`TWR index, 100 at ${fmt.date(p.dates[0])} · crosshair tooltip lists both series`} option=${lineOpt} deps=${deps} height=${280} table=${lineTable}
      actions=${html`<${RangeSelector} value=${range} onChange=${setRange} />`} empty=${p.reason || "No performance data"} /></div>
    <div class="col-4"><${Chart} title="Allocation by value" caption="Treemap · area is value; one sequential hue where no day change is supplied, diverging gain/loss where it is" option=${treeOpt} deps=${deps} height=${280} table=${treeTable} /></div>
    <div class="col-6"><${Chart} title="Risk share minus weight" caption="Diverging bars · warm = carries more risk than capital, neutral zero line" option=${divOpt} deps=${deps} height=${Math.max(180, div.length * 30 + 56)} table=${divTable} /></div>
    <div class="col-6"><${Chart} title="Correlation" caption=${`${r.window ? r.window.effective : "—"} trading days · scale centred on zero, extent set by the strongest pair · diagonal masked`} option=${heatOpt} deps=${deps} height=${Math.max(260, corrY.length * 40 + 90)} table=${heatTable} /></div>
    <div class="col-6"><${Chart} title="Drawdown" caption="Area below zero in the loss colour; benchmark as a quiet line" option=${ddOpt} deps=${deps} height=${220} table=${ddTable} /></div>
    <div class="col-6"><${Chart} title="Monthly returns" caption="Green = gain, red = loss, neutral at zero" option=${monthOpt} deps=${deps} height=${Math.max(120, years.length * 36 + 60)} table=${monthTable} /></div>
    <div class="col-6"><${Chart} title="Exposure by issuer" caption="One series → one colour, sorted, labels at the tip" option=${barOpt} deps=${deps} height=${Math.max(160, issuers.length * 32 + 48)} table=${barTable} /></div>
    <div class="col-6"><${Chart} title="Yearly returns" caption="Two series → fixed slots + legend; negative bars round at the tip" option=${yBarOpt} deps=${deps} height=${220} table=${yBarTable} /></div>
    <div class="col-4"><${Chart} title="Empty chart" caption="option = null renders the empty state" option=${null} height=${140} empty="No data for this window" /></div>
    <div class="col-8"><${Card} title="Sparklines" caption="Sparkline({ values, color, area, baseline, endDot }) — 30 closes per holding, quote currency">
      <div class="ks-sparks">
        ${snapshot.holdings.map((h) => html`<div key=${h.isin} class="ks-spark">
          <div class="row between small"><span class="truncate" title=${h.name}>${fmt.shortName(h.name, 22)}</span><span class=${"num " + fmt.polarityClass(h.day_change_pct)}>${fmt.pct(h.day_change_pct, { signed: true })}</span></div>
          <div style="height:36px"><${Sparkline} values=${h.sparkline} baseline=${h.sparkline[0]} /></div>
        </div>`)}
        <div class="ks-spark">
          <div class="row between small"><span>Bare ChartBody</span><span class="faint">no card</span></div>
          <${ChartBody} height=${36} option=${() => charts.sparklineOption(snapshot.holdings[0].sparkline, { color: "var(--color-accent)", area: false })} deps=${deps} />
        </div>
      </div>
    <//></div>
  </div>`;
}

function BadgeGallery({ snapshot }) {
  const kinds = ["neutral", "info", "good", "warn", "serious", "bad", "live", "demo"];
  return html`<${Card} title="Badges" caption="Status colours always ship with an icon or a word — never colour alone.">
    <div class="stack gap-3">
      <div class="row wrap">${kinds.map((k) => html`<${Badge} key=${k} kind=${k}>${k}<//>`)}</div>
      <div class="row wrap">${kinds.map((k) => html`<${Badge} key=${k} kind=${k} size="lg" dot>${k}<//>`)}</div>
      <div class="row wrap">${kinds.map((k) => html`<${Badge} key=${k} kind=${k} outline icon="info">${k}<//>`)}</div>
      <div class="row wrap"><span class="caps" style="width:80px">Severity</span>${["INFO", "WARNING", "SERIOUS", "CRITICAL"].map((s) => html`<${SeverityBadge} key=${s} severity=${s} />`)}</div>
      <div class="row wrap"><span class="caps" style="width:80px">Verdict</span>${["PASS", "THIN", "STALE", "FAILED", "REFUSED", "ABSENT"].map((v) => html`<${VerdictBadge} key=${v} verdict=${v} />`)}</div>
      <div class="row wrap"><span class="caps" style="width:80px">Tx type</span>${["BUY", "SELL", "DIVIDEND", "FEE"].map((t) => html`<${TxTypeBadge} key=${t} type=${t} />`)}</div>
      <div class="row wrap"><span class="caps" style="width:80px">Mode</span><${ModeBadge} mode="seed" /><${ModeBadge} mode="user" /><span class="faint small">live snapshot: ${snapshot.meta.mode}</span></div>
    </div>
  <//>`;
}

function NoticeGallery({ snapshot }) {
  const synthetic = [
    { code: "ks_critical", severity: "CRITICAL", title: "Synthetic: price feed down for 3 holdings", detail: "Nothing on this row is real — it is here so the CRITICAL treatment gets exercised.", isins: [], route: "#/settings" },
    { code: "ks_serious", severity: "SERIOUS", title: "Synthetic: cluster of 3 highly correlated holdings", detail: "Mean correlation 0.91 across 58% of the book.", isins: [], route: "#/risk" },
    { code: "ks_warning", severity: "WARNING", title: "Synthetic: effective holdings below half of actual", detail: "6 positions behave like 2.4.", isins: [], route: "#/risk" },
    { code: "ks_info", severity: "INFO", title: "Synthetic: risk window shortened to 180 days", detail: "The youngest holding binds the window.", isins: [], route: "" },
  ];
  const alerts = [...(snapshot.alerts || []), ...synthetic].slice(0, 5);
  while (alerts.length < 5) alerts.push({ code: `ks_pad${alerts.length}`, severity: "INFO", title: `Synthetic note ${alerts.length + 1}`, detail: "Padding to five items.", isins: [], route: "" });
  return html`<div class="stack gap-3">
    <${Notice} alerts=${alerts} visible=${3} onNavigate=${(route) => navigate(route)} />
    <${Notice} alerts=${alerts} compact />
    <${Notice} alerts=${[]} />
    <p class="faint small">Five items, three visible, the rest behind a disclosure; the compact variant is one line; an empty list renders nothing.</p>
  </div>`;
}

function BannerGallery() {
  const [dismissed, setDismissed] = useState(false);
  return html`<div class="stack">
    <${Banner} kind="info" title="Info.">Delayed prices are never shown as live.<//>
    <${Banner} kind="warn" title="Warning." action=${html`<${Button} size="sm" variant="secondary">Act<//>`}>Two holdings are priced from yesterday's close.<//>
    <${Banner} kind="error" title="Error." action=${html`<${Button} size="sm" variant="secondary">Retry<//>`}>The provider could not be reached. Showing the last good data.<//>
    <${Banner} kind="good" title="Done.">13 transactions imported.<//>
    ${dismissed ? html`<${Button} size="sm" variant="ghost" onClick=${() => setDismissed(false)}>Restore dismissed banner<//>` : html`<${Banner} kind="neutral" onDismiss=${() => setDismissed(true)}>Neutral, dismissible.<//>`}
  </div>`;
}

function OverlayGallery({ snapshot, onBomb }) {
  const [last, setLast] = useState("");
  const h = snapshot.holdings[0];
  const doConfirm = async (danger) => {
    const ok = await confirm({ title: danger ? "Void this transaction?" : "Refresh prices?", body: danger ? "Voided rows stay in the ledger with a reason; nothing is deleted." : "This bypasses the 15-minute quote cache.", confirmLabel: danger ? "Void" : "Refresh", danger });
    setLast(`confirm → ${ok}`);
  };
  const formModal = () => openModal({
    title: "Modal with a form", size: "lg",
    body: html`<div class="stack gap-3">
      <${Field} label="ISIN" hint="12 characters, checksum verified on save" required><${Input} placeholder="IE00B4L5Y983" autoFocus /><//>
      <${Field} label="Note"><${Textarea} rows=${3} placeholder="Optional" /><//>
    </div>`,
    footer: html`<${Button} variant="secondary" onClick=${closeModal}>Cancel<//><${Button} variant="primary" onClick=${() => { closeModal(); toast.success("Saved (not really)"); }}>Save<//>`,
  });
  return html`<div class="stack gap-3">
    <div class="row wrap">
      <${Button} variant="secondary" icon="panel" onClick=${() => openDrawer({ title: fmt.shortName(h.name, 40), subtitle: `${h.symbol} · ${h.isin}`, body: html`<${HoldingDrawerBody} h=${h} />`, footer: html`<${Button} variant="secondary" onClick=${closeDrawer}>Close<//>` })}>Open drawer<//>
      <${Button} variant="secondary" icon="panel" onClick=${() => openDrawer({ width: "wide", title: "Wide drawer", subtitle: "720px", flush: true, body: html`<${DataTable} columns=${[{ key: "name", label: "Name" }, { key: "value", label: "Value", format: "money" }]} rows=${snapshot.holdings.map((x) => ({ id: x.isin, name: fmt.shortName(x.name, 40), value: x.value }))} />` })}>Open wide drawer<//>
      <${Button} variant="secondary" icon="external" onClick=${() => navigate(`/holdings/${h.isin}`)}>Route drawer #/holdings/{isin}<//>
    </div>
    <div class="row wrap">
      <${Button} variant="secondary" onClick=${() => doConfirm(false)}>Confirm<//>
      <${Button} variant="danger" onClick=${() => doConfirm(true)}>Confirm (danger)<//>
      <${Button} variant="secondary" onClick=${formModal}>Modal with form<//>
      <span class="faint small">${last}</span>
    </div>
    <div class="row wrap">
      <${Button} variant="secondary" onClick=${() => toast.success("Prices refreshed")}>Toast success<//>
      <${Button} variant="secondary" onClick=${() => toast.info("Snapshot generated in 177 ms", { title: "Info" })}>Toast info<//>
      <${Button} variant="secondary" onClick=${() => toast.warn("Two holdings are stale", { title: "Warning" })}>Toast warn<//>
      <${Button} variant="secondary" onClick=${() => toast.error("The server could not be reached.", { title: "Refresh failed" })}>Toast error<//>
      <${Button} variant="secondary" onClick=${() => toast("Sticks until dismissed", { timeout: 0 })}>Toast sticky<//>
    </div>
    <div class="row wrap">
      <${Button} variant="danger" icon="zap" onClick=${onBomb}>Throw a render error<//>
      <span class="faint small">exercises the global error boundary — “Try again” recovers</span>
    </div>
  </div>`;
}

function ControlsGallery() {
  const [tab, setTab] = useState("positions");
  const [seg, setSeg] = useState("weight");
  const [range, setRange] = useState("1Y");
  const [text, setText] = useState("");
  const [amount, setAmount] = useState("2500");
  const [sel, setSel] = useState("XETR");
  const [chk, setChk] = useState(true);
  return html`<div class="grid">
    <div class="col-6"><${Card} title="Tabs" caption="Arrow keys move between tabs; counts and icons optional">
      <${Tabs} tabs=${[{ id: "positions", label: "Positions", count: 6, icon: "holdings" }, { id: "watchlist", label: "Watchlist", count: 3, icon: "eye" }, { id: "closed", label: "Closed", count: 0 }]} active=${tab} onChange=${setTab}>
        <p class="muted small">Active panel: <strong>${tab}</strong></p>
      <//>
    <//></div>
    <div class="col-6"><${Card} title="Segmented" caption="md / sm, a disabled option, the range preset">
      <div class="stack gap-3">
        <${Segmented} ariaLabel="Measure" value=${seg} onChange=${setSeg} options=${[{ value: "weight", label: "Weight" }, { value: "risk", label: "Risk share" }, { value: "vol", label: "Volatility", disabled: true, tip: "Needs 60 observations" }]} />
        <${Segmented} size="sm" ariaLabel="Density" value="comfortable" options=${["comfortable", "compact"]} />
        <div class="row"><${RangeSelector} value=${range} onChange=${setRange} /><span class="faint small">range = ${range}</span></div>
      </div>
    <//></div>
    <div class="col-12"><${Card} title="Fields" caption="Field({ label, hint, error, required, inline }) around Input / Select / Textarea / Checkbox">
      <div class="ks-fields">
        <${Field} label="Text" hint="A hint under the control"><${Input} value=${text} onInput=${(e) => setText(e.target.value)} placeholder="Type…" /><//>
        <${Field} label="Amount" required hint="numeric: right-aligned, decimal keyboard"><${Input} numeric prefix="€" suffix="EUR" value=${amount} onInput=${(e) => setAmount(e.target.value)} /><//>
        <${Field} label="With error" error="ISIN checksum does not match"><${Input} value="IE00B4L5Y98X" error /><//>
        <${Field} label="Disabled"><${Input} value="read only" disabled /><//>
        <${Field} label="Select"><${Select} value=${sel} onChange=${setSel} options=${[{ value: "XETR", label: "Xetra" }, { value: "XLON", label: "London" }, { value: "XAMS", label: "Amsterdam" }]} /><//>
        <${Field} label="Select, placeholder"><${Select} value="" placeholder="Choose a venue…" options=${["XETR", "XLON"]} /><//>
        <${Field} label="Date"><${Input} type="date" value="2026-09-04" /><//>
        <${Field} label="Inline" inline><${Checkbox} checked=${chk} onChange=${setChk} label="Include voided" /><//>
        <div style="grid-column: span 2"><${Field} label="Textarea" hint="Paste a CSV here"><${Textarea} rows=${3} placeholder="date,isin,type,quantity,price" /><//></div>
        <${Field} label="Compact select (top bar)"><${Select} compact value="MEUD.PA" options=${[{ value: "MEUD.PA", label: "STOXX Europe 600 (MEUD.PA)" }]} /><//>
      </div>
    <//></div>
  </div>`;
}

function ButtonGallery() {
  const variants = ["primary", "secondary", "ghost", "danger"];
  const sizes = ["sm", "md", "lg"];
  return html`<${Card} title="Buttons" caption="variant × size, then the states: icon, icon right, icon only, loading, disabled, link">
    <div class="stack gap-3">
      ${sizes.map((s) => html`<div key=${s} class="row wrap"><span class="caps" style="width:40px">${s}</span>${variants.map((v) => html`<${Button} key=${v} variant=${v} size=${s}>${v}<//>`)}</div>`)}
      <hr class="hairline" />
      ${variants.map((v) => html`<div key=${v} class="row wrap"><span class="caps" style="width:80px">${v}</span>
        <${Button} variant=${v} icon="plus">Icon<//>
        <${Button} variant=${v} iconRight="chevronDown">Icon right<//>
        <${Button} variant=${v} icon="refresh" iconOnly title="Icon only" />
        <${Button} variant=${v} loading>Loading<//>
        <${Button} variant=${v} disabled>Disabled<//>
        <${Button} variant=${v} href="#/kitchensink" icon="external">Link<//>
      </div>`)}
      <hr class="hairline" />
      <div class="row wrap">
        <${Tooltip} text="A CSS tooltip on hover or focus" pos="bottom"><${Button} variant="secondary">Tooltip<//><//>
        <span class="row small muted">Help glyph <${Help} text="Explains the figure next to it." /></span>
        <${Button} variant="secondary" tip="tip= on the button itself" tipPos="top">tip prop<//>
        <span class="tip-inline" data-tip="Dotted underline = has a definition">dotted term</span>
      </div>
    </div>
  <//>`;
}

function IconGallery() {
  return html`<${Card} title="Icons" caption=${`${ICON_NAMES.length} inline SVG icons, 1.5px stroke`}>
    <div class="ks-icons">${ICON_NAMES.map((n) => html`<div key=${n} class="ks-icon" title=${n}><${Icon} name=${n} /><span class="xs faint">${n}</span></div>`)}</div>
  <//>`;
}

function TokenGallery() {
  const groups = {
    surfaces: ["--color-page", "--color-surface", "--color-surface-2", "--color-surface-3", "--color-border", "--color-border-strong"],
    ink: ["--color-text", "--color-text-2", "--color-text-3", "--color-accent", "--color-good", "--color-warn", "--color-serious", "--color-bad"],
    series: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => `--series-${i}`).concat(["--series-other"]),
    sequential: [1, 2, 3, 4, 5, 6, 7].map((i) => `--seq-${i}`),
    diverging: ["--div-cool", "--div-neutral", "--div-warm"],
  };
  return html`<${Card} title="Tokens" caption="styles/tokens.css — the validated palette; dark and light are separate palettes, not inversions">
    <div class="stack gap-3">
      ${Object.entries(groups).map(([g, names]) => html`<div key=${g} class="row wrap"><span class="caps" style="width:84px">${g}</span>${names.map((n) => html`<${Swatch} key=${n} name=${n} />`)}</div>`)}
      <div class="row wrap gap-4 small">
        <span class="pos">positive</span><span class="neg">negative</span><span class="muted">muted</span><span class="faint">faint</span>
        <span class="num">tabular 1,234.56</span><span class="mono">mono IE00B4L5Y983</span><kbd>kbd</kbd><span class="caps">caps label</span>
      </div>
    </div>
  <//>`;
}

/* ---------------- page ---------------- */

const KS_CSS = `
.ks-sparks { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px 20px; }
.ks-spark { min-width: 0; }
.ks-fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }
.ks-icons { display: grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap: 8px; }
.ks-icon { display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: var(--radius); color: var(--color-text-2); }
.ks-icon:hover { background: var(--color-hover); color: var(--color-text); }
.ks-swatch { display: inline-flex; align-items: center; gap: 6px; }
.ks-swatch-chip { width: 18px; height: 18px; border-radius: 4px; border: 1px solid var(--color-border-strong); }
`;

export default function KitchenSinkPage() {
  const snapshot = useStore((s) => s.snapshot);
  const loading = useStore((s) => s.loading);
  const [armed, setArmed] = useState(false);
  if (!snapshot) {
    return html`<${Page} title="Kitchen sink" subtitle="Every component, rendered from the live snapshot.">
      <${Card}><${EmptyState} icon=${loading ? "clock" : "inbox"} title=${loading ? "Loading the snapshot…" : "No snapshot"} body=${loading ? "The gallery renders from GET /api/snapshot." : "The snapshot failed to load; retry from the banner above."} /><//>
    <//>`;
  }
  return html`<${Page} title="Kitchen sink" subtitle=${`Every component and chart builder, rendered from the live snapshot (${snapshot.meta.mode} · ${snapshot.meta.provider} · ${fmt.count(snapshot.holdings.length, "holding")}). Visual regression page — keep it.`}
      actions=${html`<${Button} variant="secondary" icon="refresh" onClick=${() => location.reload()}>Reload<//>`} wide>
    <style>${KS_CSS}</style>
    <${Bomb} armed=${armed} />
    <${Section} title="Figures" caption="KpiTile — proportional figures on the value, tabular on the delta"><${KpiRow} snapshot=${snapshot} /><//>
    <${Section} title="Charts" caption="lib/charts.js builders through <Chart> — each has a table twin (toggle top-right)"><${ChartGrid} snapshot=${snapshot} /><//>
    <${Section} title="Tables"><div class="stack gap-4"><${HoldingsTable} snapshot=${snapshot} /><${FormatTable} /></div><//>
    <${Section} title="Warnings" caption="Notice and Banner"><div class="grid"><div class="col-6"><${NoticeGallery} snapshot=${snapshot} /></div><div class="col-6"><${BannerGallery} /></div></div><//>
    <${Section} title="Overlays" caption="Drawer, Modal / confirm, toasts, error boundary"><${Card}><${OverlayGallery} snapshot=${snapshot} onBomb=${() => setArmed(true)} /><//><//>
    <${Section} title="Controls"><${ControlsGallery} /><//>
    <${Section} title="Buttons"><${ButtonGallery} /><//>
    <${Section} title="Badges"><${BadgeGallery} snapshot=${snapshot} /><//>
    <${Section} title="Empty states"><div class="grid">
      <div class="col-6"><${Card}><${EmptyState} icon="inbox" title="No transactions yet" body="Add one by hand or import a CSV from your broker." action=${html`<${Button} variant="primary" icon="plus">Add transaction<//>`} /><//></div>
      <div class="col-6"><${Card}><${EmptyState} compact icon="search" title="No matches" body="Try a different filter." /><//></div>
    </div><//>
    <${Section} title="Foundations"><div class="stack gap-4"><${TokenGallery} /><${IconGallery} /></div><//>
  <//>`;
}
