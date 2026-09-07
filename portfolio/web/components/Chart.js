/* Chart — a Card-wrapped ECharts instance with a table-view twin.
   Chart({ title, caption, option | buildOption, deps, height = 260, table: { columns, rows },
           tableDefault = false, actions, footer, empty = "No data", loading, onEvents, class, legend })
   - option: an ECharts option built by lib/charts.js (recomputed when `deps` change;
     if deps is omitted the option object identity is the dependency).
   - table: the same data as rows for DataTable; when given, a table/chart toggle appears.
   - While the store is loading, the previous render stays at reduced opacity (no skeleton).
   Also exports ChartBody({ option, deps, height, onEvents }) — the bare canvas without the card. */

import { html, useRef, useState } from "/static/vendor/preact-htm.module.js";
import { Card } from "/static/components/Card.js";
import { Button } from "/static/components/Button.js";
import { DataTable } from "/static/components/DataTable.js";
import { useChart } from "/static/lib/charts.js";
import { useStore } from "/static/lib/store.js";

export function ChartBody({ option, buildOption, deps, height = 260, onEvents, class: cls = "", empty = "No data", loading, style }) {
  const storeLoading = useStore((s) => s.loading);
  const isLoading = loading ?? storeLoading;
  const ref = useRef(null);
  const build = buildOption || (() => option || null);
  const hasData = !!(option || buildOption);
  useChart(ref, build, deps || [option], { onEvents });
  return html`<div class=${["chart-body", isLoading ? "is-loading" : "", cls].filter(Boolean).join(" ")} style=${`height:${typeof height === "number" ? height + "px" : height};${style || ""}`}>
    <div ref=${ref} class="chart-canvas" role="img"></div>
    ${!hasData ? html`<div class="chart-empty">${empty}</div>` : null}
  </div>`;
}

export function Chart({ title, caption, option, buildOption, deps, height = 260, table, tableDefault = false, actions, footer, empty = "No data", loading, onEvents, class: cls = "", legend, bordered = false }) {
  const [view, setView] = useState(tableDefault ? "table" : "chart");
  const toggle = table ? html`<${Button} variant="ghost" size="sm" icon=${view === "chart" ? "table" : "chart"} iconOnly
    title=${view === "chart" ? "Show as table" : "Show as chart"} tipPos="bottom-right" onClick=${() => setView(view === "chart" ? "table" : "chart")}
    aria-pressed=${view === "table" ? "true" : "false"} />` : null;
  const acts = actions || toggle ? html`${actions}${toggle}` : null;
  return html`<${Card} class=${["chart-card", cls].filter(Boolean).join(" ")} title=${title} caption=${caption} actions=${acts} footer=${footer} bordered=${bordered} flush=${view === "table"}>
    ${view === "table" && table
      ? html`<div class="chart-table"><${DataTable} columns=${table.columns} rows=${table.rows} rowKey=${table.rowKey || "id"} maxHeight=${Math.max(height, 240)} empty=${empty} loading=${loading} /></div>`
      : html`${legend ? html`<div class="chart-legend">${legend}</div>` : null}
             <${ChartBody} option=${option} buildOption=${buildOption} deps=${deps} height=${height} onEvents=${onEvents} empty=${empty} loading=${loading} />`}
  <//>`;
}

export default Chart;
