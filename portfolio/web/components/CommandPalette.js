/* CommandPalette — Ctrl/Cmd+K. Fuzzy-matches navigation, actions and holdings.
   Mounted once by app.js as <CommandPalette actions=${[...]} />; extra actions
   can be registered at runtime with registerCommands(list) (returns an unregister fn).
   A command: { id, label, sub, icon, group, hint: ["g","o"], keywords, run() } */

import { html, useState, useEffect, useMemo, useRef } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";
import { useStore, closePalette } from "/static/lib/store.js";
import { ROUTES, navigate } from "/static/lib/router.js";
import { shortName } from "/static/lib/format.js";

const ICONS = { overview: "overview", holdings: "holdings", performance: "performance", risk: "risk", simulator: "simulator", instruments: "instruments", transactions: "transactions", settings: "settings" };

const extra = new Set();
export function registerCommands(list) {
  const arr = Array.isArray(list) ? list : [list];
  arr.forEach((c) => extra.add(c));
  return () => arr.forEach((c) => extra.delete(c));
}

/** Subsequence fuzzy score. Higher is better; -1 = no match. Returns [score, matchedIndexes]. */
export function fuzzyScore(query, text) {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  if (!q) return [0, []];
  const idx = t.indexOf(q);
  if (idx >= 0) return [100 - idx + (idx === 0 || /\s/.test(t[idx - 1]) ? 20 : 0), Array.from({ length: q.length }, (_, i) => idx + i)];
  let ti = 0, score = 0, prev = -2;
  const matched = [];
  for (let qi = 0; qi < q.length; qi++) {
    const ch = q[qi];
    let found = -1;
    while (ti < t.length) { if (t[ti] === ch) { found = ti; ti++; break; } ti++; }
    if (found < 0) return [-1, []];
    matched.push(found);
    score += (found === prev + 1 ? 6 : 1) + (found === 0 || /[\s\-_.]/.test(t[found - 1]) ? 5 : 0);
    prev = found;
  }
  // A subsequence scattered over more than 3× the query length is noise, not a match.
  if (matched[matched.length - 1] - matched[0] + 1 > q.length * 3) return [-1, []];
  return [score, matched];
}

function Highlight({ text, matched }) {
  if (!matched || !matched.length) return html`${text}`;
  const set = new Set(matched);
  const out = [];
  let buf = "", mode = false;
  for (let i = 0; i < text.length; i++) {
    const m = set.has(i);
    if (m !== mode) { if (buf) out.push(mode ? html`<mark>${buf}</mark>` : buf); buf = ""; mode = m; }
    buf += text[i];
  }
  if (buf) out.push(mode ? html`<mark>${buf}</mark>` : buf);
  return html`${out}`;
}

function isMac() { return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent); }
export const MOD_KEY = isMac() ? "⌘" : "Ctrl";

export function CommandPalette({ actions = [] }) {
  const open = useStore((s) => s.paletteOpen);
  const snapshot = useStore((s) => s.snapshot);
  const [q, setQ] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  useEffect(() => {
    if (open) { setQ(""); setCursor(0); setTimeout(() => inputRef.current && inputRef.current.focus(), 0); }
  }, [open]);

  const commands = useMemo(() => {
    const nav = ROUTES.map((r) => ({
      id: `nav:${r.page}`, label: r.label, group: "Navigate", icon: ICONS[r.page] || "circle",
      hint: ["g", r.key], keywords: `go ${r.page}`, run: () => navigate(r.path),
    }));
    const hold = [];
    if (snapshot) {
      for (const h of snapshot.holdings || []) {
        hold.push({ id: `h:${h.isin}`, label: h.name, sub: [h.symbol, h.isin].filter(Boolean).join(" · "), group: "Holdings", icon: "holdings",
          keywords: `${h.isin} ${h.symbol || ""} ${shortName(h.name, 40)}`, run: () => navigate(`/holdings/${h.isin}`) });
      }
      for (const w of snapshot.watchlist || []) {
        hold.push({ id: `w:${w.isin}`, label: w.name, sub: `${[w.symbol, w.isin].filter(Boolean).join(" · ")} · watchlist`, group: "Holdings", icon: "eye",
          keywords: `${w.isin} ${w.symbol || ""} watchlist`, run: () => navigate(`/holdings/${w.isin}`) });
      }
    }
    return [...nav, ...actions, ...extra, ...hold];
  }, [snapshot, actions]);

  const results = useMemo(() => {
    if (!q.trim()) return commands.filter((c) => c.group !== "Holdings" || false).concat(commands.filter((c) => c.group === "Holdings").slice(0, 6));
    const out = [];
    for (const c of commands) {
      const [s1, m1] = fuzzyScore(q, c.label);
      const [s2] = c.keywords ? fuzzyScore(q, c.keywords) : [-1, []];
      const [s3] = c.sub ? fuzzyScore(q, c.sub) : [-1, []];
      const s = Math.max(s1, s2 - 2, s3 - 1);
      if (s < 0) continue;
      out.push({ c, s, m: s1 >= s ? m1 : [] });
    }
    out.sort((a, b) => b.s - a.s);
    return out.slice(0, 40).map((x) => ({ ...x.c, matched: x.m }));
  }, [q, commands]);

  useEffect(() => { setCursor(0); }, [q]);
  useEffect(() => {
    const el = listRef.current && listRef.current.querySelector('[aria-selected="true"]');
    if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  if (!open) return null;

  const run = (c) => { closePalette(); if (c && c.run) setTimeout(() => c.run(), 0); };
  const onKey = (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setCursor((i) => Math.min(results.length - 1, i + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setCursor((i) => Math.max(0, i - 1)); }
    else if (e.key === "Enter") { e.preventDefault(); run(results[cursor]); }
    else if (e.key === "Escape") { e.preventDefault(); closePalette(); }
    else if (e.key === "Tab") { e.preventDefault(); }
  };

  let lastGroup = null;
  return html`<div class="palette-root" onKeyDown=${onKey}>
    <div class="palette-backdrop" onClick=${closePalette}></div>
    <div class="palette" role="dialog" aria-modal="true" aria-label="Command palette">
      <div class="palette-input-row">
        <${Icon} name="search" size=${16} />
        <input ref=${inputRef} class="palette-input" type="text" placeholder="Go to page, run an action, or find a holding…"
          value=${q} onInput=${(e) => setQ(e.target.value)} role="combobox" aria-expanded="true" aria-controls="palette-list" aria-activedescendant=${results[cursor] ? `pi-${results[cursor].id}` : undefined} autocomplete="off" spellcheck="false" />
        <kbd>esc</kbd>
      </div>
      <div ref=${listRef} id="palette-list" class="palette-list" role="listbox">
        ${results.length === 0 ? html`<div class="palette-empty">No matches for “${q}”</div>` : results.map((c, i) => {
          const header = c.group !== lastGroup ? html`<div class="palette-group">${c.group}</div>` : null;
          lastGroup = c.group;
          return html`${header}<button type="button" id=${`pi-${c.id}`} class="palette-item" role="option" aria-selected=${i === cursor ? "true" : "false"}
            onMouseEnter=${() => setCursor(i)} onClick=${() => run(c)}>
            <span class="pi-icon"><${Icon} name=${c.icon || "circle"} size=${15} /></span>
            <span class="pi-label"><${Highlight} text=${c.label} matched=${c.matched} /></span>
            ${c.sub ? html`<span class="pi-sub">${c.sub}</span>` : null}
            ${c.hint ? html`<span class="pi-hint">${c.hint.map((k) => html`<kbd>${k}</kbd>`)}</span>` : null}
          </button>`;
        })}
      </div>
      <div class="palette-foot">
        <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
        <span><kbd>↵</kbd> select</span>
        <span><kbd>esc</kbd> close</span>
      </div>
    </div>
  </div>`;
}

export default CommandPalette;
