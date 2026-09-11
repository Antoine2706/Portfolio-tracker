/* Chart infrastructure on ECharts 5 (UMD global `echarts`).

   useChart(ref, buildOption, deps, opts) — init once (canvas, dirty-rect),
   setOption with lazyUpdate, shared debounced ResizeObserver, dispose on
   unmount, re-init on theme change.

   Option builders encode the dataviz rules: thin marks, 4px rounded data
   ends anchored at the baseline, 2px lines, hairline solid gridlines, a
   crosshair + one-tooltip-for-all-series on line/area, per-mark tooltips on
   bars/cells, legend only for >= 2 series, text in text tokens, no dual axes.
   Every builder has a table twin helper (seriesTable, categoryTable, matrixTable). */

import { html, useLayoutEffect, useEffect, useRef } from "/static/vendor/preact-htm.module.js";
import { tokens, alpha, inkOn } from "/static/lib/theme.js";
import { useStore } from "/static/lib/store.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- shared resize ---------------- */

const observed = new Map(); // element -> chart
let ro = null;
let resizeTimer = 0;
const dirty = new Set();

function flushResize() {
  resizeTimer = 0;
  for (const chart of dirty) {
    if (chart && !chart.isDisposed()) chart.resize();
  }
  dirty.clear();
}

function scheduleResize(chart) {
  dirty.add(chart);
  if (resizeTimer) return;
  resizeTimer = setTimeout(flushResize, 80);
}

function observe(el, chart) {
  if (!ro) {
    ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const c = observed.get(e.target);
        if (c) scheduleResize(c);
      }
    });
    window.addEventListener("resize", () => { for (const c of observed.values()) scheduleResize(c); });
  }
  observed.set(el, chart);
  ro.observe(el);
}

function unobserve(el) {
  if (ro && el) ro.unobserve(el);
  observed.delete(el);
}

/* ---------------- hook ---------------- */

/**
 * useChart(ref, buildOption, deps, { theme, onEvents, notMerge })
 *   ref         — a useRef() attached to the container div (needs a height)
 *   buildOption — () => ECharts option | null (null clears the chart)
 *   deps        — dependency array; the option is rebuilt when they change
 *   onEvents    — { click: fn, ... } bound once per chart instance
 * Returns the ref to the chart instance ({ current }).
 */
export function useChart(ref, buildOption, deps = [], { theme, onEvents, notMerge = false } = {}) {
  const storeTheme = useStore((s) => s.theme);
  const themeName = theme || storeTheme;
  const chartRef = useRef(null);
  const build = useRef(buildOption);
  build.current = buildOption;
  const events = useRef(onEvents);
  events.current = onEvents;

  // init / re-init on theme change
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !window.echarts) return undefined;
    const chart = window.echarts.init(el, themeName, { renderer: "canvas", useDirtyRect: true });
    chartRef.current = chart;
    const opt = build.current && build.current();
    if (opt) chart.setOption(opt, { notMerge: true, lazyUpdate: true });
    const bound = events.current || {};
    for (const [name, fn] of Object.entries(bound)) chart.on(name, fn);
    observe(el, chart);
    return () => {
      unobserve(el);
      dirty.delete(chart);
      chart.dispose();
      if (chartRef.current === chart) chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [themeName]);

  // update on deps change (skip the very first run; init already set it)
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    const chart = chartRef.current;
    if (!chart || chart.isDisposed()) return;
    const opt = build.current && build.current();
    if (!opt) { chart.clear(); return; }
    chart.setOption(opt, { notMerge, lazyUpdate: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return chartRef;
}

/* ---------------- tooltip DOM (safe: textContent only) ---------------- */

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = String(text);
  return n;
}

/** Build a tooltip element: title + rows [{ name, value, color, kind: "line"|"swatch", polarity }]. */
export function tipElement(title, rows, foot) {
  const root = el("div", "echarts-tip");
  if (title != null && title !== "") root.appendChild(el("div", "tip-title", title));
  for (const r of rows) {
    const row = el("div", "tip-row");
    if (r.color) {
      const key = el("span", r.kind === "swatch" ? "swatch" : "linekey");
      key.style.background = r.color;
      row.appendChild(key);
    }
    if (r.name != null) row.appendChild(el("span", "tip-name", r.name));
    const v = el("span", "tip-value" + (r.polarity ? " " + r.polarity : ""), r.value);
    row.appendChild(v);
    root.appendChild(row);
  }
  if (foot) root.appendChild(el("div", "tip-foot", foot));
  return root;
}

function tooltipBase(t) {
  return {
    show: true,
    confine: true,
    appendToBody: true,
    transitionDuration: 0.12,
    backgroundColor: t.surface2,
    borderColor: t.borderStrong,
    borderWidth: 1,
    padding: [8, 10],
    textStyle: { color: t.text, fontFamily: t.fontSans, fontSize: 12 },
    extraCssText: "box-shadow: 0 8px 24px rgba(0,0,0,.25); border-radius: 6px;",
  };
}

/* ---------------- axis helpers ---------------- */

function valueFormatter(yFormat, ccy) {
  if (typeof yFormat === "function") return yFormat;
  switch (yFormat) {
    case "money": return (v) => fmt.money(v, ccy);
    case "money-compact": return (v) => fmt.money(v, ccy, { compact: true });
    case "pct": return (v) => fmt.pct(v, { decimals: 1 });
    case "pct2": return (v) => fmt.pct(v, { decimals: 2 });
    case "pct-signed": return (v) => fmt.pct(v, { signed: true, decimals: 1 });
    case "index": return (v) => fmt.num(v, { decimals: 1 });
    case "num": return (v) => fmt.num(v, { decimals: 2 });
    default: return (v) => fmt.num(v, { decimals: 2 });
  }
}

function tickFormatter(yFormat, ccy) {
  if (typeof yFormat === "function") return yFormat;
  switch (yFormat) {
    case "money": case "money-compact": return (v) => fmt.money(v, ccy, { compact: true });
    case "pct": case "pct-signed": case "pct2": return (v) => fmt.pct(v, { decimals: Math.abs(v) < 0.01 && v !== 0 ? 1 : 0 });
    case "index": return (v) => fmt.num(v, { decimals: 0 });
    case "num": return (v) => fmt.num(v, { decimals: 2, compact: true });
    default: return (v) => String(v);
  }
}

/**
 * Ticks for a category axis of ISO dates: one label per month boundary (every
 * month / quarter / half-year / year depending on the span; the year is spelled
 * out on the first tick and on January), or every 1–2 weeks for short spans.
 * Returns { interval, formatter } for axisLabel — never the same label twice.
 */
export function dateAxisTicks(dates = []) {
  const n = dates.length;
  const ticks = new Map();
  if (!n) return { interval: 0, formatter: (v) => fmt.date(v), ticks };
  const parts = dates.map((d) => {
    const [y, m, day] = String(d).slice(0, 10).split("-").map(Number);
    return { y, m, d: day, t: Date.UTC(y, (m || 1) - 1, day || 1) / 86400000 };
  });
  const span = parts[n - 1].t - parts[0].t;
  if (span > 120) {
    const step = span > 1500 ? 12 : span > 730 ? 6 : span > 400 ? 3 : 1; // months between ticks
    for (let i = 1; i < n; i++) {
      const p = parts[i], q = parts[i - 1];
      if (p.y === q.y && p.m === q.m) continue;
      if ((p.m - 1) % step !== 0) continue;
      ticks.set(i, p.m === 1 || ticks.size === 0 ? `${fmt.MONTHS_SHORT[p.m - 1]} ${p.y}` : fmt.MONTHS_SHORT[p.m - 1]);
    }
  } else {
    const stepDays = span > 45 ? 14 : span > 12 ? 7 : 1;
    let last = -Infinity;
    for (let i = 0; i < n; i++) {
      if (parts[i].t - last < stepDays) continue;
      ticks.set(i, `${parts[i].d} ${fmt.MONTHS_SHORT[parts[i].m - 1]}`);
      last = parts[i].t;
    }
  }
  return { interval: (i) => ticks.has(i), formatter: (v, i) => ticks.get(i) ?? "", ticks };
}

function baseGrid(t, { legend = false, left = 8, right = 12, top, bottom = 4 } = {}) {
  return { left, right, top: top ?? (legend ? 32 : 12), bottom, containLabel: true };
}

function xTimeAxis(t, dates) {
  const ticks = dateAxisTicks(dates);
  return {
    type: "category",
    data: dates,
    boundaryGap: false,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: t.axisText, fontSize: 11, hideOverlap: true, interval: ticks.interval, formatter: ticks.formatter, margin: 10, showMinLabel: false, showMaxLabel: false },
    splitLine: { show: false },
    axisPointer: { label: { show: false } },
  };
}

function yValueAxis(t, yFormat, ccy, extra = {}) {
  return {
    type: "value",
    scale: true,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: t.axisText, fontSize: 11, formatter: tickFormatter(yFormat, ccy), margin: 8 },
    splitLine: { show: true, lineStyle: { color: t.gridline, width: 1, type: "solid" } },
    splitNumber: 4,
    ...extra,
  };
}

function legendFor(t, count, names) {
  if (count < 2) return { show: false };
  return {
    show: true,
    data: names,
    top: 0, right: 8,
    icon: "roundRect", itemWidth: 12, itemHeight: 8, itemGap: 14,
    textStyle: { color: t.text2, fontSize: 11 },
    inactiveColor: t.text3,
  };
}

/** Series colour by fixed slot; >8 folds to Other. */
export function seriesColor(i, t = tokens()) {
  return i < 8 ? t.series[i] : t.seriesOther;
}

/** Fold a list of items past the first 8 into "Other" (by value sum). */
export function foldOther(items, { limit = 8, key = "value" } = {}) {
  if (items.length <= limit) return items;
  const sorted = [...items].sort((a, b) => (b[key] || 0) - (a[key] || 0));
  const head = sorted.slice(0, limit - 1);
  const tail = sorted.slice(limit - 1);
  const other = { name: "Other", [key]: tail.reduce((s, x) => s + (x[key] || 0), 0), members: tail, isOther: true };
  return [...head, other];
}

/* ---------------- builders ---------------- */

/**
 * lineOption({ series: [{ name, dates?, values, color?, area?, dashed?, width? }], dates?,
 *              yFormat, currency, area, markLines: [{ value, label }], baseline, colors, height,
 *              endLabels, stacked })
 * dates may be given once at the top level or per series (they must be aligned).
 * endLabels: direct-label each series at its last point (name + value), legend kept.
 * stacked:   stacked area — bands filled at .55 with a 2px surface gap between them.
 */
export function lineOption({ series = [], dates, yFormat = "num", currency = "EUR", area = false, markLines = [], baseline, colors, theme, min, max, emphasisIndex, endLabels = false, stacked = false } = {}) {
  const t = tokens(theme);
  const xs = dates || (series[0] && series[0].dates) || [];
  const names = series.map((s) => s.name);
  const format = valueFormatter(yFormat, currency);
  const s = series.map((ser, i) => {
    const color = ser.color || (colors && colors[i]) || seriesColor(i, t);
    const isEm = emphasisIndex == null || emphasisIndex === i;
    const out = {
      name: ser.name,
      type: "line",
      data: ser.values.map((v) => (v == null ? null : v)),
      showSymbol: false,
      symbol: "circle",
      symbolSize: 8,
      connectNulls: false,
      itemStyle: { color, borderColor: t.chartSurface, borderWidth: 2 },
      lineStyle: { color, width: ser.width ?? (isEm ? 2 : 1.5), type: ser.dashed ? "dashed" : "solid", join: "round", cap: "round", opacity: isEm ? 1 : .55 },
      emphasis: { focus: "none", lineStyle: { width: ser.width ?? 2 } },
      z: isEm ? 3 : 2,
      sampling: xs.length > 2000 ? "lttb" : undefined,
    };
    if (area || ser.area) {
      out.areaStyle = { color: alpha(color, .10) };
    }
    if (stacked) {
      out.stack = "total";
      out.areaStyle = { color: alpha(color, .55) };
      out.lineStyle = { color: t.chartSurface, width: 2, join: "round", cap: "round" };
      out.emphasis = { focus: "none" };
    }
    if (endLabels) {
      out.endLabel = { show: true, color: t.text2, fontSize: 11, offset: [6, 0], valueAnimation: false, formatter: (p) => `${p.seriesName} ${format(p.value)}` };
      out.labelLayout = { moveOverlap: "shiftY" };
    }
    if (i === 0 && markLines.length) {
      out.markLine = {
        silent: true,
        symbol: "none",
        animation: false,
        lineStyle: { color: t.borderStrong, width: 1, type: "solid" },
        label: { color: t.text3, fontSize: 11, position: "insideEndTop", formatter: (p) => p.name || "" },
        data: markLines.map((m) => ({ yAxis: m.value, name: m.label || "" })),
      };
    }
    return out;
  });
  if (baseline != null && !markLines.length && s[0]) {
    s[0].markLine = {
      silent: true, symbol: "none", animation: false,
      lineStyle: { color: t.borderStrong, width: 1, type: "solid" },
      label: { show: false },
      data: [{ yAxis: baseline }],
    };
  }
  // room for the end labels: the longest "name value" at ~6.4px per character
  const labelRoom = endLabels ? Math.min(220, 16 + Math.round((Math.max(0, ...names.map((n) => String(n).length)) + 7) * 6.4)) : 12;
  return {
    animationDuration: 200,
    animationDurationUpdate: 160,
    animationEasing: "cubicOut",
    grid: baseGrid(t, { legend: series.length > 1, right: labelRoom }),
    legend: legendFor(t, series.length, names),
    tooltip: {
      ...tooltipBase(t),
      trigger: "axis",
      axisPointer: { type: "line", snap: true, lineStyle: { color: t.borderStrong, width: 1, type: "solid" }, label: { show: false } },
      formatter: (params) => {
        const list = Array.isArray(params) ? params : [params];
        if (!list.length) return "";
        const title = fmt.date(list[0].axisValue);
        const rows = list.map((p) => ({
          name: p.seriesName, color: p.color, value: format(p.value), kind: "line",
        }));
        return tipElement(title, rows);
      },
    },
    xAxis: xTimeAxis(t, xs),
    yAxis: yValueAxis(t, yFormat, currency, { min, max }),
    series: s,
  };
}

/**
 * barOption({ categories, values, horizontal, format, color, series: [{name, values}], stacked, currency,
 *             markLines: [{ value, label }], tipRows: (params) => [{ name, value }] })
 * One series → one colour (slot 1). Several → fixed slots, legend shown.
 * markLines draw a reference rule on the value axis (the axis is widened to
 * include it); tipRows appends rows to the per-bar tooltip.
 */
export function barOption({ categories = [], values, series, horizontal = false, format = "num", currency = "EUR", color, colors, stacked = false, theme, showLabels = false, sort, markLines = [], tipRows } = {}) {
  const t = tokens(theme);
  const fmtV = valueFormatter(format, currency);
  let cats = categories;
  let sers = series || [{ name: "", values: values || [] }];
  if (sort && sers.length === 1) {
    const idx = cats.map((_, i) => i).sort((a, b) => (sort === "asc" ? 1 : -1) * ((sers[0].values[a] || 0) - (sers[0].values[b] || 0)));
    cats = idx.map((i) => cats[i]);
    sers = [{ ...sers[0], values: idx.map((i) => sers[0].values[i]) }];
  }
  const multi = sers.length > 1;
  const radius = (v) => {
    const neg = v != null && v < 0;
    if (horizontal) return neg ? [4, 0, 0, 4] : [0, 4, 4, 0];
    return neg ? [0, 0, 4, 4] : [4, 4, 0, 0];
  };
  const s = sers.map((ser, i) => {
    const c = ser.color || color || (colors && colors[i]) || seriesColor(i, t);
    return {
      name: ser.name,
      type: "bar",
      stack: stacked ? "total" : undefined,
      barMaxWidth: 18,
      barGap: "30%",
      barCategoryGap: "45%",
      data: ser.values.map((v) => ({
        value: v,
        itemStyle: { color: c, borderRadius: stacked ? 0 : radius(v), borderColor: t.chartSurface, borderWidth: stacked ? 1 : 0 },
      })),
      emphasis: { focus: "none", itemStyle: { opacity: .85 } },
      label: showLabels ? {
        show: true,
        position: horizontal ? "right" : "top",
        color: t.text2, fontSize: 11,
        formatter: (p) => fmtV(p.value),
      } : { show: false },
    };
  });
  const catAxis = {
    type: "category",
    data: cats,
    inverse: horizontal,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: t.axisText, fontSize: 11, hideOverlap: true, width: horizontal ? 160 : undefined, overflow: horizontal ? "truncate" : undefined, interval: horizontal ? 0 : "auto" },
    splitLine: { show: false },
  };
  const ruleMax = markLines.length ? Math.max(...markLines.map((m) => m.value)) : null;
  const valAxis = yValueAxis(t, format, currency, { scale: false, max: ruleMax == null ? undefined : (v) => Math.max(v.max, ruleMax * 1.08) });
  if (markLines.length && s[0]) {
    s[0].markLine = {
      silent: true, symbol: "none", animation: false,
      lineStyle: { color: t.text3, width: 1, type: "solid" },
      // a vertical rule labels above the plot (the grid keeps 24px for it);
      // "inside" positions would set the text along the line
      label: { color: t.text3, fontSize: 11, position: horizontal ? "start" : "insideEndTop", formatter: (p) => p.name || "" },
      data: markLines.map((m) => (horizontal ? { xAxis: m.value, name: m.label || "" } : { yAxis: m.value, name: m.label || "" })),
    };
  }
  return {
    animationDuration: 200,
    animationDurationUpdate: 160,
    animationEasing: "cubicOut",
    grid: baseGrid(t, { legend: multi, right: showLabels ? 40 : 12, top: horizontal && markLines.length ? 24 : undefined }),
    legend: legendFor(t, sers.length, sers.map((x) => x.name)),
    tooltip: {
      ...tooltipBase(t),
      trigger: "item",
      formatter: (p) => tipElement(p.name, [{ name: multi ? p.seriesName : null, color: p.color, kind: "swatch", value: fmtV(p.value) }, ...(tipRows ? tipRows(p) : [])]),
    },
    xAxis: horizontal ? valAxis : catAxis,
    yAxis: horizontal ? catAxis : valAxis,
    series: s,
  };
}

/**
 * divergingBarOption({ names, values, extent, format, sentences, labels: { pos, neg, zero }, palette })
 * Horizontal bars coloured by sign. palette "risk" (default): warm for positive
 * = carries MORE risk than capital, cool (accent) for negative. palette "pnl":
 * gain / loss status colours for signed money or return. Neutral zero rule;
 * labels name the sign in the tooltip; sentences[i] is the tooltip footer.
 */
export function divergingBarOption({ names = [], values = [], extent, format = "pp", currency = "EUR", theme, sentences, labels, palette = "risk" } = {}) {
  const t = tokens(theme);
  const fmtV = format === "pp" ? (v) => fmt.pp(v, { signed: true }) : valueFormatter(format, currency);
  const maxAbs = extent ?? Math.max(0.01, ...values.map((v) => Math.abs(v || 0)));
  const posColor = palette === "pnl" ? t.good : t.divWarm;
  const negColor = palette === "pnl" ? t.bad : t.divCool;
  const lab = { pos: "more risk than capital", neg: "less risk than capital", zero: "balanced", ...(labels || {}) };
  const data = values.map((v) => ({
    value: v,
    itemStyle: {
      color: v == null ? t.divNeutral : v > 0 ? posColor : v < 0 ? negColor : t.divNeutral,
      borderRadius: v != null && v < 0 ? [4, 0, 0, 4] : [0, 4, 4, 0],
    },
  }));
  // Headroom on both sides of the symmetric domain so the outside label of the
  // largest bar never runs past the plot edge.
  const span = maxAbs * 1.18;
  return {
    animationDuration: 200,
    animationDurationUpdate: 160,
    grid: { left: 8, right: 16, top: 8, bottom: 4, containLabel: true },
    legend: { show: false },
    tooltip: {
      ...tooltipBase(t),
      trigger: "item",
      formatter: (p) => tipElement(p.name, [{ color: p.color, kind: "swatch", value: fmtV(p.value), name: p.value > 0 ? lab.pos : p.value < 0 ? lab.neg : lab.zero }], sentences ? sentences[p.dataIndex] : undefined),
    },
    xAxis: {
      type: "value",
      min: -span, max: span,
      axisLine: { show: false },
      axisTick: { show: false },
      // the domain edges are not round numbers (headroom), so only the
      // interval ticks are labelled — otherwise "6%" and "7%" collide
      axisLabel: { color: t.axisText, fontSize: 11, showMinLabel: false, showMaxLabel: false, formatter: (v) => (format === "pp" ? fmt.pp(v, { signed: true, decimals: 0 }) : tickFormatter(format, currency)(v)) },
      splitLine: { show: true, lineStyle: { color: t.gridline, width: 1, type: "solid" } },
      splitNumber: 4,
    },
    yAxis: {
      type: "category",
      data: names,
      inverse: true,
      axisLine: { show: true, onZero: true, lineStyle: { color: t.borderStrong, width: 1 } },
      axisTick: { show: false },
      axisLabel: { color: t.text2, fontSize: 11, width: 170, overflow: "truncate", interval: 0 },
      splitLine: { show: false },
      z: 5,
    },
    series: [{
      type: "bar",
      data,
      barMaxWidth: 16,
      barCategoryGap: "45%",
      emphasis: { focus: "none", itemStyle: { opacity: .85 } },
      label: {
        show: true,
        position: "outside",
        color: t.text2,
        fontSize: 11,
        formatter: (p) => fmtV(p.value),
      },
      labelLayout: (p) => ({ x: p.rect.x + (values[p.dataIndex] >= 0 ? p.rect.width + 6 : -6), align: values[p.dataIndex] >= 0 ? "left" : "right" }),
    }],
  };
}

/**
 * heatmapOption({ xLabels, yLabels, matrix, min:-1, max:1, format, showValues, mask })
 * Diverging cool ↔ neutral ↔ warm. Cells carry their own tooltip.
 *
 * `mask(x, y)` marks cells that are structurally uninformative — the diagonal
 * of a correlation grid, where every value is 1.00 by definition. Masked cells
 * are drawn flat and unlabelled in a second series that the visualMap does not
 * touch, so they cannot be the brightest thing on screen while carrying no
 * information, and they do not stretch the colour domain either.
 */
export function heatmapOption({ xLabels = [], yLabels = [], matrix = [], min = -1, max = 1, format, showValues = true, theme, cellLabel, tipLabels, tipYLabels, mask } = {}) {
  const t = tokens(theme);
  const fmtV = format || ((v) => fmt.num(v, { decimals: 2 }));
  // the axes may carry short codes (tickers); the tooltip names in full
  const tipX = tipLabels || xLabels;
  const tipY = tipYLabels || yLabels;
  const data = [];
  const masked = [];
  for (let y = 0; y < yLabels.length; y++) {
    for (let x = 0; x < xLabels.length; x++) {
      const v = matrix[y] ? matrix[y][x] : null;
      if (v == null) continue;
      (mask && mask(x, y) ? masked : data).push([x, y, v]);
    }
  }
  const mid = (min + max) / 2;
  const colorAt = (v) => {
    if (v == null) return t.divNeutral;
    const r = v >= mid ? (v - mid) / (max - mid || 1) : (mid - v) / (mid - min || 1);
    return mixHex(t.divNeutral, v >= mid ? t.divWarm : t.divCool, Math.min(1, Math.max(0, r)));
  };
  return {
    animation: false,
    grid: { left: 8, right: 12, top: 8, bottom: 36, containLabel: true },
    tooltip: {
      ...tooltipBase(t),
      trigger: "item",
      position: "top",
      formatter: (p) => tipElement(`${tipY[p.value[1]]} × ${tipX[p.value[0]]}`, [{ color: colorAt(p.value[2]), kind: "swatch", value: fmtV(p.value[2]) }]),
    },
    xAxis: {
      type: "category", data: xLabels, position: "top",
      axisLine: { show: false }, axisTick: { show: false },
      // long labels lean at 45° so neighbours never overprint; short codes stay flat
      axisLabel: { color: t.text2, fontSize: 11, interval: 0, width: 90, overflow: "truncate", rotate: xLabels.some((l) => String(l).length > 8) ? 45 : 0 },
      splitLine: { show: false }, splitArea: { show: false },
    },
    yAxis: {
      type: "category", data: yLabels, inverse: true,
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: t.text2, fontSize: 11, interval: 0, width: 150, overflow: "truncate" },
      splitLine: { show: false }, splitArea: { show: false },
    },
    visualMap: {
      min, max,
      show: true,
      orient: "horizontal",
      left: "center", bottom: 0,
      itemWidth: 8, itemHeight: 120,
      calculable: false,
      // series 0 only: the masked cells in series 1 keep their flat fill
      seriesIndex: 0,
      inRange: { color: [t.divCool, t.divNeutral, t.divWarm] },
      text: [fmtV(max), fmtV(min)],
      textStyle: { color: t.text3, fontSize: 10 },
      precision: 2,
    },
    series: [{
      type: "heatmap",
      data: data.map(([x, y, v]) => ({ value: [x, y, v], label: { color: inkOn(colorAt(v)) } })),
      itemStyle: { borderColor: t.chartSurface, borderWidth: 2, borderRadius: 3 },
      label: { show: showValues, fontSize: 11, formatter: (p) => (cellLabel ? cellLabel(p.value[2], p.value[0], p.value[1]) : fmtV(p.value[2])) },
      emphasis: { itemStyle: { borderColor: t.text, borderWidth: 1.5 } },
    }, {
      // Masked cells: present so the grid keeps its shape, recessive so they
      // never compete with a real reading. Not covered by the visualMap.
      type: "heatmap",
      silent: true,
      data: masked.map(([x, y, v]) => ({ value: [x, y, v] })),
      itemStyle: { color: t.gridline, borderColor: t.chartSurface, borderWidth: 2, borderRadius: 3 },
      label: { show: false },
    }],
  };
}

/**
 * treemapOption({ items: [{ name, value, sub, change }], currency, total, format, changeExtent })
 *
 * Area is value. That is the encoding a treemap exists for, and it is already
 * unambiguous, so tinting the same variable a second time by lightness spends
 * the colour channel on information the reader has. Where an item carries a
 * signed `change`, colour is given to that instead: gain green, loss red,
 * neutral at zero, on the app-wide polarity convention. Two channels, two
 * variables — what each holding is worth, and what it did today.
 *
 * Without any `change` values it falls back to the single sequential hue,
 * which is honest for a plain allocation picture: one variable, one channel,
 * and lightness carrying no second claim.
 */
export function treemapOption({ items = [], currency = "EUR", format, theme, total, changeExtent, changeFormat } = {}) {
  const t = tokens(theme);
  const values = items.map((i) => i.value || 0);
  const sum = total ?? values.reduce((a, b) => a + b, 0);
  const maxV = Math.max(...values, 1e-9);
  const minV = Math.min(...values, maxV);
  const fmtV = format || ((v) => fmt.money(v, currency, { compact: true }));
  const fmtC = changeFormat || ((v) => fmt.pct(v, { signed: true }));
  const steps = t.sequential;
  const stepFor = (v) => {
    // magnitude relative to the largest item, on a 7-step ramp (index 1..6 keeps extremes readable)
    const r = maxV === minV ? 1 : (v - minV) / (maxV - minV);
    return steps[Math.min(steps.length - 1, 1 + Math.round(r * (steps.length - 2)))];
  };
  // Scaled to the day actually being shown, with a floor so a quiet session is
  // not amplified into a dramatic one. A fixed extent would leave every tile
  // neutral on most days; an unfloored one would make ±0.05% look like a rout.
  const changes = items.map((i) => i.change).filter((v) => v != null);
  const byChange = changes.length > 0;
  const cMax = Math.max(0.005, changeExtent ?? Math.max(0, ...changes.map(Math.abs)));
  const changeColor = (v) => {
    if (v == null) return t.divNeutral;
    return mixHex(t.divNeutral, v >= 0 ? t.good : t.bad, Math.min(1, Math.abs(v) / cMax));
  };
  const data = items.map((i) => {
    const color = i.color || (byChange ? changeColor(i.change) : stepFor(i.value || 0));
    return {
      name: i.name,
      value: i.value,
      sub: i.sub,
      change: i.change,
      weight: sum ? (i.value || 0) / sum : null,
      itemStyle: { color, borderColor: t.chartSurface, borderWidth: 0, gapWidth: 2 },
      label: { color: inkOn(color) },
    };
  });
  return {
    animationDurationUpdate: 160,
    tooltip: {
      ...tooltipBase(t),
      trigger: "item",
      formatter: (p) => tipElement(p.name, [
        { name: "value", value: fmtV(p.value), color: p.color, kind: "swatch" },
        { name: "weight", value: fmt.pct(p.data.weight) },
        ...(byChange && p.data.change != null
          ? [{ name: "today", value: fmtC(p.data.change), polarity: fmt.polarityClass(p.data.change) }]
          : []),
      ], p.data.sub),
    },
    series: [{
      type: "treemap",
      data,
      roam: false,
      nodeClick: false,
      breadcrumb: { show: false },
      left: 0, right: 0, top: 0, bottom: 0,
      squareRatio: 0.7,
      itemStyle: { borderColor: t.chartSurface, borderWidth: 0, gapWidth: 2, borderRadius: 4 },
      label: {
        show: true,
        position: "insideTopLeft",
        padding: [8, 10],
        fontSize: 11,
        lineHeight: 15,
        formatter: (p) => {
          const w = p.data.weight;
          return `{n|${fmt.shortName(p.name, 26)}}\n{v|${w == null ? "" : fmt.pct(w)}}`;
        },
        rich: {
          n: { fontSize: 11, fontWeight: 600, lineHeight: 15 },
          v: { fontSize: 11, lineHeight: 15, opacity: .85 },
        },
        overflow: "truncate",
      },
      upperLabel: { show: false },
      emphasis: { focus: "none", itemStyle: { opacity: .9 }, label: { show: true } },
      levels: [{ itemStyle: { gapWidth: 2, borderWidth: 0, borderColor: t.chartSurface } }],
    }],
  };
}

/**
 * sparklineOption(values, { color, area, baseline }) — no axes, no tooltip, no grid.
 * De-emphasis hue by default (text-3); the end dot in accent.
 */
export function sparklineOption(values = [], { color, area = true, theme, endDot = true, baseline } = {}) {
  const t = tokens(theme);
  const c = color || t.text3;
  const clean = values.map((v) => (v == null ? null : v));
  const last = clean.length - 1;
  return {
    animation: false,
    grid: { left: 1, right: endDot ? 4 : 1, top: 3, bottom: 3 },
    xAxis: { type: "category", show: false, boundaryGap: false, data: clean.map((_, i) => i) },
    yAxis: { type: "value", show: false, scale: true, min: baseline != null ? (v) => Math.min(v.min, baseline) : undefined },
    tooltip: { show: false },
    series: [{
      type: "line",
      data: clean.map((v, i) => (i === last && endDot ? { value: v, symbol: "circle", symbolSize: 6, itemStyle: { color: t.accent, borderColor: t.chartSurface, borderWidth: 1.5 } } : v)),
      showSymbol: endDot,
      symbol: "none",
      symbolSize: 0,
      lineStyle: { color: c, width: 1.5, join: "round", cap: "round" },
      areaStyle: area ? { color: alpha(c, .10) } : undefined,
      silent: true,
      markLine: baseline != null ? { silent: true, symbol: "none", animation: false, label: { show: false }, lineStyle: { color: t.borderStrong, width: 1, type: "solid" }, data: [{ yAxis: baseline }] } : undefined,
    }],
  };
}

/**
 * drawdownOption(dates, values, { benchmark }) — area below zero in the loss
 * colour at low alpha; optional benchmark drawdown as a second, quieter line.
 */
export function drawdownOption(dates = [], values = [], { benchmark, benchmarkName = "Benchmark", name = "Portfolio", theme } = {}) {
  const t = tokens(theme);
  const series = [{
    name,
    type: "line",
    data: values,
    showSymbol: false,
    lineStyle: { color: t.bad, width: 1.5, join: "round", cap: "round" },
    itemStyle: { color: t.bad },
    areaStyle: { color: alpha(t.bad, .14), origin: "start" },
    emphasis: { focus: "none" },
    z: 3,
  }];
  if (benchmark) {
    series.push({
      name: benchmarkName,
      type: "line",
      data: benchmark,
      showSymbol: false,
      lineStyle: { color: t.text3, width: 1.5, join: "round", cap: "round" },
      itemStyle: { color: t.text3 },
      emphasis: { focus: "none" },
      z: 2,
    });
  }
  return {
    animationDuration: 200,
    grid: baseGrid(t, { legend: !!benchmark }),
    legend: legendFor(t, series.length, series.map((s) => s.name)),
    tooltip: {
      ...tooltipBase(t),
      trigger: "axis",
      axisPointer: { type: "line", snap: true, lineStyle: { color: t.borderStrong, width: 1, type: "solid" }, label: { show: false } },
      formatter: (params) => {
        const list = Array.isArray(params) ? params : [params];
        return tipElement(fmt.date(list[0].axisValue), list.map((p) => ({ name: p.seriesName, color: p.color, kind: "line", value: fmt.pct(p.value, { decimals: 1 }), polarity: fmt.polarityClass(p.value) })));
      },
    },
    xAxis: xTimeAxis(t, dates),
    yAxis: yValueAxis(t, "pct", "EUR", { max: 0, scale: false }),
    series,
  };
}

/**
 * monthlyHeatmapOption(years, months, matrix) — matrix[yearIndex][monthIndex] = return or null.
 * Diverging by polarity: green = gain, red = loss, neutral at zero.
 */
export function monthlyHeatmapOption(years = [], months = fmt.MONTHS_SHORT, matrix = [], { theme, extent, showValues = true } = {}) {
  const t = tokens(theme);
  const flat = matrix.flat().filter((v) => v != null);
  const maxAbs = extent ?? Math.max(0.01, ...flat.map((v) => Math.abs(v)));
  const colorAt = (v) => {
    if (v == null) return t.divNeutral;
    const r = Math.min(1, Math.abs(v) / maxAbs);
    return mixHex(t.divNeutral, v >= 0 ? t.good : t.bad, r);
  };
  const data = [];
  for (let y = 0; y < years.length; y++) {
    for (let m = 0; m < months.length; m++) {
      const v = matrix[y] ? matrix[y][m] : null;
      if (v == null) continue;
      data.push({ value: [m, y, v], itemStyle: { color: colorAt(v) }, label: { color: inkOn(colorAt(v)) } });
    }
  }
  return {
    animation: false,
    grid: { left: 8, right: 8, top: 8, bottom: 8, containLabel: true },
    tooltip: {
      ...tooltipBase(t),
      trigger: "item",
      position: "top",
      formatter: (p) => tipElement(`${months[p.value[0]]} ${years[p.value[1]]}`, [{ color: p.color, kind: "swatch", value: fmt.pct(p.value[2], { signed: true, decimals: 1 }), polarity: fmt.polarityClass(p.value[2]) }]),
    },
    xAxis: {
      type: "category", data: months, position: "top",
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: t.text2, fontSize: 11, interval: 0 },
      splitLine: { show: false },
    },
    yAxis: {
      type: "category", data: years.map(String), inverse: true,
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: t.text2, fontSize: 11, interval: 0 },
      splitLine: { show: false },
    },
    visualMap: { show: false, min: -maxAbs, max: maxAbs, inRange: { color: [t.bad, t.divNeutral, t.good] } },
    series: [{
      type: "heatmap",
      data,
      itemStyle: { borderColor: t.chartSurface, borderWidth: 2, borderRadius: 3 },
      label: { show: showValues, fontSize: 11, formatter: (p) => fmt.pct(p.value[2], { decimals: 1 }) },
      emphasis: { itemStyle: { borderColor: t.text, borderWidth: 1.5 } },
    }],
  };
}

/**
 * histogramOption({ bins: [{ x0, x1, count }], format, currency, markLines: [{ value, label }], countLabel })
 * One hue. Bars sit on a value axis (bin centres) so reference rules land at
 * exact values; each bar carries its own tooltip (range + count).
 */
export function histogramOption({ bins = [], format = "pct", currency = "EUR", markLines = [], theme, countLabel = "days" } = {}) {
  const t = tokens(theme);
  const fmtV = valueFormatter(format, currency);
  const data = bins.map((b) => ({ value: [(b.x0 + b.x1) / 2, b.count], x0: b.x0, x1: b.x1 }));
  const lo = bins.length ? bins[0].x0 : undefined;
  const hi = bins.length ? bins[bins.length - 1].x1 : undefined;
  return {
    animationDuration: 200,
    animationDurationUpdate: 160,
    grid: baseGrid(t, { top: markLines.length ? 26 : 12 }),
    legend: { show: false },
    tooltip: {
      ...tooltipBase(t),
      trigger: "item",
      formatter: (p) => tipElement(`${fmtV(p.data.x0)} to ${fmtV(p.data.x1)}`, [{ color: p.color, kind: "swatch", value: `${fmt.int(p.value[1])} ${countLabel}` }]),
    },
    xAxis: {
      type: "value",
      min: lo, max: hi,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: t.axisText, fontSize: 11, formatter: tickFormatter(format, currency), margin: 10, hideOverlap: true },
      splitLine: { show: false },
      splitNumber: 6,
    },
    yAxis: yValueAxis(t, (v) => fmt.int(v), currency, { scale: false, minInterval: 1 }),
    series: [{
      type: "bar",
      data,
      barCategoryGap: "12%",
      itemStyle: { color: seriesColor(0, t), borderRadius: [3, 3, 0, 0] },
      emphasis: { focus: "none", itemStyle: { opacity: .85 } },
      markLine: markLines.length ? {
        silent: true, symbol: "none", animation: false,
        lineStyle: { color: t.text2, width: 1, type: "solid" },
        label: { color: t.text2, fontSize: 11, position: "end", formatter: (p) => p.name || "" },
        data: markLines.map((m) => ({ xAxis: m.value, name: m.label || "" })),
      } : undefined,
    }],
  };
}

/**
 * smallMultiplesOption({ panels: [{ name, values, yFormat, markLines, color, area, min, max }], dates, height, currency })
 * Several measures on one canvas, one panel each, stacked vertically with a
 * shared x axis and a linked crosshair — the answer to "two scales" that never
 * becomes a dual axis. `height` is the container height in px (panels split it).
 */
export function smallMultiplesOption({ panels = [], dates, height = 320, theme, currency = "EUR", left = 48 } = {}) {
  const t = tokens(theme);
  const xs = dates || (panels[0] && panels[0].dates) || [];
  const n = Math.max(1, panels.length);
  const titleH = 22, gapH = 12, axisH = 24;
  const slot = (height - axisH) / n;
  const grids = [], xAxes = [], yAxes = [], series = [], titles = [], formats = [];
  panels.forEach((p, i) => {
    const top = Math.round(i * slot + titleH);
    const h = Math.max(40, Math.round(slot - titleH - gapH));
    const last = i === n - 1;
    const color = p.color || seriesColor(i, t);
    grids.push({ left, right: 12, top, height: h, containLabel: false });
    const xa = xTimeAxis(t, xs);
    xa.gridIndex = i;
    if (!last) xa.axisLabel = { ...xa.axisLabel, show: false };
    xAxes.push(xa);
    const ya = yValueAxis(t, p.yFormat || "num", currency, { gridIndex: i, min: p.min, max: p.max, splitNumber: 3 });
    yAxes.push(ya);
    titles.push({ text: p.name, left: left - 2, top: Math.round(i * slot) - 2, textStyle: { color: t.text2, fontSize: 11, fontWeight: 500, fontFamily: t.fontSans } });
    formats.push(valueFormatter(p.yFormat || "num", currency));
    series.push({
      name: p.name,
      type: "line",
      xAxisIndex: i, yAxisIndex: i,
      data: (p.values || []).map((v) => (v == null ? null : v)),
      showSymbol: false, symbol: "circle", symbolSize: 8, connectNulls: false,
      itemStyle: { color, borderColor: t.chartSurface, borderWidth: 2 },
      lineStyle: { color, width: 2, join: "round", cap: "round" },
      areaStyle: p.area ? { color: alpha(color, .10) } : undefined,
      emphasis: { focus: "none" },
      sampling: xs.length > 2000 ? "lttb" : undefined,
      markLine: p.markLines && p.markLines.length ? {
        silent: true, symbol: "none", animation: false,
        lineStyle: { color: t.text3, width: 1, type: "solid" },
        label: { color: t.text3, fontSize: 11, position: "insideEndTop", formatter: (m) => m.name || "" },
        data: p.markLines.map((m) => ({ yAxis: m.value, name: m.label || "" })),
      } : undefined,
    });
  });
  return {
    animationDuration: 200,
    animationDurationUpdate: 160,
    title: titles,
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    legend: { show: false },
    axisPointer: { link: [{ xAxisIndex: "all" }], lineStyle: { color: t.borderStrong, width: 1, type: "solid" }, label: { show: false } },
    tooltip: {
      ...tooltipBase(t),
      trigger: "axis",
      axisPointer: { type: "line", snap: true, lineStyle: { color: t.borderStrong, width: 1, type: "solid" }, label: { show: false } },
      formatter: (params) => {
        const list = Array.isArray(params) ? params : [params];
        if (!list.length) return "";
        return tipElement(fmt.date(list[0].axisValue), list.map((p) => ({ name: p.seriesName, color: p.color, kind: "line", value: formats[p.seriesIndex](p.value) })));
      },
    },
    series,
  };
}

/* ---------------- table twins ---------------- */

/** seriesTable({ dates, series:[{name, values}], format }) → { columns, rows } for DataTable */
export function seriesTable({ dates = [], series = [], format = "num", currency = "EUR", dateLabel = "Date" } = {}) {
  const f = valueFormatter(format, currency);
  const columns = [
    { key: "date", label: dateLabel, format: (v) => fmt.date(v), width: 120 },
    ...series.map((s, i) => ({ key: `s${i}`, label: s.name, align: "right", format: f, sortable: true })),
  ];
  const rows = dates.map((d, r) => {
    const row = { date: d, id: d };
    series.forEach((s, i) => { row[`s${i}`] = s.values[r] ?? null; });
    return row;
  });
  return { columns, rows };
}

/** categoryTable({ categories, values, format, label, valueLabel, valueClass })
 *
 * `valueClass(v)` returns a CSS class for the value cell, so a chart's table
 * twin can carry the same colour encoding as its marks. A table that shows
 * the same numbers in a different colour from the chart it toggles with is
 * two claims about one quantity.
 */
export function categoryTable({ categories = [], values = [], format = "num", currency = "EUR", label = "Category", valueLabel = "Value", extra = [], valueClass } = {}) {
  const f = format === "pp" ? (v) => fmt.pp(v, { signed: true }) : valueFormatter(format, currency);
  const columns = [
    { key: "name", label, sortable: true },
    valueClass
      ? { key: "value", label: valueLabel, align: "right", sortable: true,
          render: (row, v) => html`<span class=${valueClass(v)}>${f(v, row)}</span>` }
      : { key: "value", label: valueLabel, align: "right", format: f, sortable: true },
    ...extra,
  ];
  const rows = categories.map((c, i) => ({ id: c, name: c, value: values[i] ?? null }));
  return { columns, rows };
}

/** matrixTable({ xLabels, yLabels, matrix, format, cornerLabel }) */
export function matrixTable({ xLabels = [], yLabels = [], matrix = [], format, cornerLabel = "" } = {}) {
  const f = format || ((v) => fmt.num(v, { decimals: 2 }));
  const columns = [
    { key: "name", label: cornerLabel, sortable: false },
    ...xLabels.map((x, i) => ({ key: `c${i}`, label: x, align: "right", format: f })),
  ];
  const rows = yLabels.map((y, r) => {
    const row = { id: y, name: y };
    xLabels.forEach((_, c) => { row[`c${c}`] = matrix[r] ? matrix[r][c] ?? null : null; });
    return row;
  });
  return { columns, rows };
}

/* ---------------- colour math ---------------- */

function toRgb(c) {
  const s = String(c).trim();
  if (s.startsWith("#")) {
    const hex = s.length === 4 ? s.slice(1).split("").map((ch) => ch + ch).join("") : s.slice(1, 7);
    const n = parseInt(hex, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const m = s.match(/rgba?\(([^)]+)\)/);
  if (m) return m[1].split(",").slice(0, 3).map((x) => parseFloat(x));
  return [0, 0, 0];
}

/** Linear mix of two colours in sRGB, r ∈ [0,1] toward `b`. */
export function mixHex(a, b, r) {
  const A = toRgb(a), B = toRgb(b);
  const c = A.map((v, i) => Math.round(v + (B[i] - v) * r));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
