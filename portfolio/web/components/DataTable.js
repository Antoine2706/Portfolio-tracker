/* DataTable({
     columns: [{ key, label, align: "left"|"right"|"center", format: fn(value,row) | string, width,
                 sortable (default true), render: fn(row, value) → vnode, sub: fn(row) → string,
                 numeric (implies right + tabular), primary, class, title, sortValue: fn(row) }],
     rows, rowKey = "id" | fn(row), sort: { key, dir } (initial), onSort,
     onRowClick(row, event), selectedKey, empty = "No rows", footer: row | fn(rows) → vnode,
     density: "comfortable"|"compact", densityToggle, toolbar (left) / toolbarRight, maxHeight,
     scroll (the wrap scrolls horizontally; implied by maxHeight), rowClass: fn(row) → class,
     loading (defaults to store.loading), class, stickyHeader = true, caption
   })
   Client-side sort with an indicator; sticky header (sticks to the page scroll
   by default, to the wrap when maxHeight/scroll is set); hover rows; numeric
   columns right-aligned with tabular figures; keyboard-focusable clickable rows
   (Enter/Space). Footer cells use the column's render/format like body cells. */

import { html, useState, useMemo, useCallback, useRef, useEffect } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";
import { Segmented } from "/static/components/Segmented.js";
import { useStore } from "/static/lib/store.js";
import * as fmt from "/static/lib/format.js";

function resolveFormat(col) {
  if (typeof col.format === "function") return col.format;
  if (typeof col.format === "string") return fmt.fmtCell(col.format, col.formatOptions);
  return (v) => (v == null || v === "" ? fmt.DASH : String(v));
}

function isNumericCol(col) {
  if (col.numeric != null) return col.numeric;
  if (col.align === "right") return true;
  const f = col.format;
  return typeof f === "string" && /money|pct|pp|num|int|qty|mult|days/.test(f);
}

function compare(a, b) {
  const an = a == null, bn = b == null;
  if (an && bn) return 0;
  if (an) return 1;   // nulls last
  if (bn) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "boolean" && typeof b === "boolean") return (a === b ? 0 : a ? -1 : 1);
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
}

export function DataTable({
  columns = [], rows = [], rowKey = "id", sort: initialSort, onSort, onRowClick, selectedKey,
  empty = "No rows", footer, density: densityProp, densityToggle = false, toolbar, toolbarRight,
  maxHeight, loading, class: cls = "", stickyHeader = true, caption, rowClass, scroll = false,
}) {
  const storeLoading = useStore((s) => s.loading);
  const isLoading = loading ?? storeLoading;
  const [sort, setSort] = useState(initialSort || null);
  const [density, setDensity] = useState(densityProp || "comfortable");
  const keyOf = useCallback((row, i) => (typeof rowKey === "function" ? rowKey(row, i) : row[rowKey] ?? i), [rowKey]);

  const cols = useMemo(() => columns.map((c) => ({
    ...c,
    numeric: isNumericCol(c),
    fmt: resolveFormat(c),
    sortable: c.sortable !== false,
    align: c.align || (isNumericCol(c) ? "right" : "left"),
  })), [columns]);

  const sorted = useMemo(() => {
    if (!sort || !sort.key) return rows;
    const col = cols.find((c) => c.key === sort.key);
    if (!col) return rows;
    const val = (row) => (col.sortValue ? col.sortValue(row) : row[col.key]);
    const dir = sort.dir === "desc" ? -1 : 1;
    return rows.map((r, i) => [r, i]).sort((a, b) => dir * compare(val(a[0]), val(b[0])) || a[1] - b[1]).map((x) => x[0]);
  }, [rows, sort, cols]);

  const toggleSort = (col) => {
    if (!col.sortable) return;
    let next;
    if (!sort || sort.key !== col.key) next = { key: col.key, dir: col.numeric ? "desc" : "asc" };
    else if (sort.dir === (col.numeric ? "desc" : "asc")) next = { key: col.key, dir: col.numeric ? "asc" : "desc" };
    else next = null;
    setSort(next);
    if (onSort) onSort(next);
  };

  const onRowKey = (e, row) => {
    if (!onRowClick) return;
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onRowClick(row, e); }
  };

  const footRow = typeof footer === "function" ? footer(sorted) : footer;
  const showToolbar = toolbar || toolbarRight || densityToggle;

  // A table wider than its card must scroll sideways rather than lose its
  // last columns to the clip; a table that fits keeps the page-sticky header.
  // CSS cannot tell the two apart, so the wrap measures itself.
  const wrapRef = useRef(null);
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap || typeof ResizeObserver === "undefined") return undefined;
    const check = () => {
      const table = wrap.querySelector("table.table");
      if (!table) return;
      const over = table.offsetWidth > wrap.clientWidth + 1;
      if ((wrap.dataset.overflow === "true") !== over) wrap.dataset.overflow = over ? "true" : "false";
    };
    check();
    const ro = new ResizeObserver(check);
    ro.observe(wrap);
    const table = wrap.querySelector("table.table");
    if (table) ro.observe(table);
    return () => ro.disconnect();
  }, [cols, sorted, density]);

  return html`<div ref=${wrapRef} class=${["table-wrap", maxHeight || scroll ? "scroll" : "", isLoading ? "is-loading" : "", cls].filter(Boolean).join(" ")} style=${maxHeight ? `max-height:${typeof maxHeight === "number" ? maxHeight + "px" : maxHeight}` : undefined}>
    ${showToolbar ? html`<div class="table-toolbar">
      <div class="row">${toolbar}</div>
      <div class="row">
        ${toolbarRight}
        ${densityToggle ? html`<${Segmented} size="sm" ariaLabel="Density" value=${density} onChange=${setDensity}
          options=${[{ value: "comfortable", label: "Comfortable" }, { value: "compact", label: "Compact" }]} />` : null}
      </div>
    </div>` : null}
    <table class="table" data-density=${density}>
      ${caption ? html`<caption class="sr-only">${caption}</caption>` : null}
      <thead style=${stickyHeader ? undefined : "position:static"}>
        <tr>
          ${cols.map((c) => {
            const active = sort && sort.key === c.key;
            const ariaSort = active ? (sort.dir === "asc" ? "ascending" : "descending") : undefined;
            return html`<th key=${c.key} scope="col" class=${[c.numeric ? "num" : "", c.align === "center" ? "center" : "", c.sortable ? "sortable" : "", c.class || ""].filter(Boolean).join(" ")}
              style=${c.width ? `width:${typeof c.width === "number" ? c.width + "px" : c.width}` : undefined}
              aria-sort=${ariaSort} tabindex=${c.sortable ? 0 : undefined} title=${c.title}
              onClick=${() => toggleSort(c)} onKeyDown=${(e) => { if (c.sortable && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggleSort(c); } }}>
              <span class="th-inner">${c.label}${c.sortable ? html`<${Icon} class="sort-icon" name=${active ? (sort.dir === "asc" ? "sortAsc" : "sortDesc") : "sort"} size=${12} stroke=${2} />` : null}</span>
            </th>`;
          })}
        </tr>
      </thead>
      <tbody>
        ${sorted.length === 0 ? html`<tr><td class="empty-cell" colspan=${cols.length}>${empty}</td></tr>` : sorted.map((row, i) => {
          const k = keyOf(row, i);
          const clickable = !!onRowClick;
          const extra = typeof rowClass === "function" ? rowClass(row) : "";
          return html`<tr key=${k} class=${[clickable ? "clickable" : "", selectedKey != null && selectedKey === k ? "selected" : "", extra].filter(Boolean).join(" ")}
            tabindex=${clickable ? 0 : undefined} onClick=${clickable ? (e) => onRowClick(row, e) : undefined} onKeyDown=${clickable ? (e) => onRowKey(e, row) : undefined}
            aria-selected=${selectedKey != null && selectedKey === k ? "true" : undefined}>
            ${cols.map((c) => {
              const v = row[c.key];
              const content = c.render ? c.render(row, v) : c.fmt(v, row);
              const sub = c.sub ? c.sub(row) : null;
              return html`<td key=${c.key} class=${[c.numeric ? "num" : "", c.align === "center" ? "center" : "", c.primary ? "primary" : "", c.class || ""].filter(Boolean).join(" ")}>
                ${content}${sub ? html`<span class="cell-sub">${sub}</span>` : null}
              </td>`;
            })}
          </tr>`;
        })}
      </tbody>
      ${footRow ? html`<tfoot><tr>
        ${cols.map((c) => {
          const v = footRow[c.key];
          const content = c.renderFooter ? c.renderFooter(footRow, v) : v == null ? "" : typeof v === "string" ? v : c.render ? c.render(footRow, v) : c.fmt(v, footRow);
          return html`<td key=${c.key} class=${[c.numeric ? "num" : "", c.align === "center" ? "center" : ""].filter(Boolean).join(" ")}>${content}</td>`;
        })}
      </tr></tfoot>` : null}
    </table>
  </div>`;
}

/** A polarity-coloured numeric cell renderer: colorCell("pct-signed") → render fn */
export function colorCell(kind, opts) {
  const f = fmt.fmtCell(kind, opts);
  return (row, v) => html`<span class=${fmt.polarityClass(v)}>${f(v, row)}</span>`;
}

/** Name cell with a secondary line (symbol · ISIN), truncated. */
export function nameCell({ sub = (r) => [r.symbol, r.isin].filter(Boolean).join(" · "), maxWidth = 260 } = {}) {
  return (row, v) => html`<div class="truncate" style=${`max-width:${maxWidth}px`} title=${v}>
    <span>${v ?? fmt.DASH}</span>
    ${sub(row) ? html`<span class="cell-sub">${sub(row)}</span>` : null}
  </div>`;
}

export default DataTable;
