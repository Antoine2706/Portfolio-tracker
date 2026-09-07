/* Theme: dark by default, light selected. Persists to localStorage("pt.theme").
   index.html applies the stored/preferred theme before first paint; this
   module keeps the store, the <html> attribute and ECharts in sync.

   tokens(name?) reads computed CSS custom properties (styles/tokens.css is
   the single source of truth). Reading a non-active theme uses a probe
   element carrying data-theme, so both ECharts themes can be registered at
   boot without flipping the page. */

import { app } from "/static/lib/store.js";

const STORAGE_KEY = "pt.theme";

export const TOKEN_NAMES = {
  page: "--color-page",
  surface: "--color-surface",
  surface2: "--color-surface-2",
  surface3: "--color-surface-3",
  border: "--color-border",
  borderStrong: "--color-border-strong",
  text: "--color-text",
  text2: "--color-text-2",
  text3: "--color-text-3",
  accent: "--color-accent",
  accentFg: "--color-accent-fg",
  accentSoft: "--color-accent-soft",
  good: "--color-good",
  warn: "--color-warn",
  serious: "--color-serious",
  bad: "--color-bad",
  gridline: "--color-gridline",
  chartSurface: "--color-chart-surface",
  axisText: "--color-axis-text",
  seriesOther: "--series-other",
  divCool: "--div-cool",
  divNeutral: "--div-neutral",
  divWarm: "--div-warm",
  fontSans: "--font-sans",
};

const cache = new Map();

/** Read the design tokens of a theme as a plain object. */
export function tokens(name) {
  const current = getTheme();
  const which = name || current;
  if (cache.has(which)) return cache.get(which);
  let el = document.documentElement;
  let probe = null;
  if (which !== current) {
    probe = document.createElement("div");
    probe.setAttribute("data-theme", which);
    probe.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;visibility:hidden";
    document.body.appendChild(probe);
    el = probe;
  }
  const cs = getComputedStyle(el);
  const read = (v) => cs.getPropertyValue(v).trim();
  const t = {};
  for (const [k, v] of Object.entries(TOKEN_NAMES)) t[k] = read(v);
  t.series = [];
  for (let i = 1; i <= 8; i++) t.series.push(read(`--series-${i}`));
  t.sequential = [];
  for (let i = 1; i <= 7; i++) t.sequential.push(read(`--seq-${i}`));
  t.fontSans = t.fontSans || "Inter, ui-sans-serif, system-ui, sans-serif";
  t.name = which;
  if (probe) probe.remove();
  cache.set(which, t);
  return t;
}

/** Clear the token cache (call if tokens.css is swapped at runtime). */
export function invalidateTokens() { cache.clear(); }

export function getTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

export function setTheme(name) {
  const theme = name === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem(STORAGE_KEY, theme); } catch (_) { /* private mode */ }
  app.patch({ theme });
  return theme;
}

export function toggleTheme() {
  return setTheme(getTheme() === "dark" ? "light" : "dark");
}

/** Follow the OS if the user has never chosen. */
export function watchSystemTheme() {
  if (!window.matchMedia) return;
  const mq = window.matchMedia("(prefers-color-scheme: light)");
  const onChange = () => {
    let stored = null;
    try { stored = localStorage.getItem(STORAGE_KEY); } catch (_) { /* ignore */ }
    if (!stored) {
      document.documentElement.setAttribute("data-theme", mq.matches ? "light" : "dark");
      app.patch({ theme: mq.matches ? "light" : "dark" });
    }
  };
  if (mq.addEventListener) mq.addEventListener("change", onChange);
  else if (mq.addListener) mq.addListener(onChange);
}

/** Hex/rgb colour with alpha applied, for washes. */
export function alpha(color, a) {
  const c = String(color).trim();
  if (c.startsWith("#")) {
    const hex = c.length === 4 ? c.slice(1).split("").map((ch) => ch + ch).join("") : c.slice(1, 7);
    const n = parseInt(hex, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const [r, g, b] = m[1].split(",").map((s) => parseFloat(s));
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  return c;
}

/** Relative luminance 0..1 of a hex/rgb colour (for label ink over fills). */
export function luminance(color) {
  const c = String(color).trim();
  let r = 0, g = 0, b = 0;
  if (c.startsWith("#")) {
    const hex = c.length === 4 ? c.slice(1).split("").map((ch) => ch + ch).join("") : c.slice(1, 7);
    const n = parseInt(hex, 16);
    r = (n >> 16) & 255; g = (n >> 8) & 255; b = n & 255;
  } else {
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (m) [r, g, b] = m[1].split(",").map((s) => parseFloat(s));
  }
  const lin = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/** Ink colour that clears contrast on a given fill. */
export function inkOn(fill) {
  return luminance(fill) > 0.4 ? "#121211" : "#f2f1ed";
}

/** Build an ECharts theme object from the tokens of `name`. */
export function chartTheme(name) {
  const t = tokens(name);
  const axisCommon = {
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: t.axisText, fontFamily: t.fontSans, fontSize: 11 },
    splitLine: { show: true, lineStyle: { color: t.gridline, width: 1, type: "solid" } },
    splitArea: { show: false },
    nameTextStyle: { color: t.text3, fontFamily: t.fontSans, fontSize: 11 },
  };
  return {
    color: t.series,
    backgroundColor: "transparent",
    textStyle: { color: t.text2, fontFamily: t.fontSans, fontSize: 12 },
    title: { textStyle: { color: t.text, fontFamily: t.fontSans, fontWeight: 600, fontSize: 13 } },
    line: {
      lineStyle: { width: 2, join: "round", cap: "round" },
      symbol: "circle", symbolSize: 8, showSymbol: false, smooth: false,
      emphasis: { lineStyle: { width: 2 } },
    },
    bar: { barMaxWidth: 18, itemStyle: { borderRadius: [4, 4, 0, 0] } },
    categoryAxis: { ...axisCommon, splitLine: { show: false }, boundaryGap: true },
    valueAxis: axisCommon,
    timeAxis: { ...axisCommon, splitLine: { show: false } },
    logAxis: axisCommon,
    legend: {
      textStyle: { color: t.text2, fontFamily: t.fontSans, fontSize: 11 },
      inactiveColor: t.text3,
      itemWidth: 12, itemHeight: 8, itemGap: 14, icon: "roundRect",
      pageTextStyle: { color: t.text2 },
      pageIconColor: t.text2, pageIconInactiveColor: t.text3,
    },
    tooltip: {
      backgroundColor: t.surface2,
      borderColor: t.borderStrong,
      borderWidth: 1,
      padding: [8, 10],
      textStyle: { color: t.text, fontFamily: t.fontSans, fontSize: 12 },
      extraCssText: "box-shadow: 0 8px 24px rgba(0,0,0,.25); border-radius: 6px;",
      axisPointer: {
        type: "line",
        lineStyle: { color: t.borderStrong, width: 1, type: "solid" },
        crossStyle: { color: t.borderStrong, width: 1, type: "solid" },
        shadowStyle: { color: t.borderStrong, opacity: .3 },
        label: { backgroundColor: t.surface3, color: t.text, fontFamily: t.fontSans, fontSize: 11, borderRadius: 4, padding: [3, 6] },
      },
    },
    visualMap: { textStyle: { color: t.text3, fontFamily: t.fontSans, fontSize: 11 } },
    markLine: { lineStyle: { color: t.borderStrong, width: 1, type: "solid" }, label: { color: t.text3, fontFamily: t.fontSans, fontSize: 11 } },
    markPoint: { label: { color: t.text, fontFamily: t.fontSans } },
    dataZoom: { textStyle: { color: t.text3 } },
    heatmap: { itemStyle: { borderColor: t.chartSurface, borderWidth: 2 } },
    treemap: { itemStyle: { borderColor: t.chartSurface, gapWidth: 2 }, breadcrumb: { show: false } },
  };
}

/** Register both ECharts themes ("dark" and "light"). Idempotent. */
export function registerChartThemes() {
  if (!window.echarts) return;
  window.echarts.registerTheme("dark", chartTheme("dark"));
  window.echarts.registerTheme("light", chartTheme("light"));
}
