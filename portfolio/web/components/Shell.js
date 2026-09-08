/* Shell({ children }) — sidebar + top bar around the routed page.
   Top bar: mode badge, prices-as-of note, refresh, benchmark select, theme toggle, palette trigger.
   Sidebar: nav with icons and "g then key" hints, collapsible to icons only. */

import { html, useState, useEffect } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";
import { Button } from "/static/components/Button.js";
import { ModeBadge } from "/static/components/Badge.js";
import { Select } from "/static/components/Field.js";
import { MOD_KEY } from "/static/components/CommandPalette.js";
import { toast } from "/static/components/Toast.js";
import { app, useStore, setBenchmark, setSidebarCollapsed, openPalette } from "/static/lib/store.js";
import { ROUTES, href, useRoute, navigate } from "/static/lib/router.js";
import { toggleTheme } from "/static/lib/theme.js";
import { refreshPrices, errorMessage } from "/static/lib/api.js";
import * as fmt from "/static/lib/format.js";

const ICONS = { overview: "overview", holdings: "holdings", performance: "performance", risk: "risk", simulator: "simulator", allocate: "wallet", instruments: "instruments", transactions: "transactions", settings: "settings" };

function BrandMark() {
  return html`<svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <rect x="2" y="9" width="3" height="5" rx="1" fill="currentColor" opacity=".55"/>
    <rect x="6.5" y="5" width="3" height="9" rx="1" fill="currentColor" opacity=".8"/>
    <rect x="11" y="2" width="3" height="12" rx="1" fill="currentColor"/>
  </svg>`;
}

/** True while the media query matches; re-renders on change. */
export function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => !!(window.matchMedia && window.matchMedia(query).matches));
  useEffect(() => {
    if (!window.matchMedia) return undefined;
    const mq = window.matchMedia(query);
    const onChange = () => setMatches(mq.matches);
    onChange();
    if (mq.addEventListener) mq.addEventListener("change", onChange); else mq.addListener(onChange);
    return () => { if (mq.removeEventListener) mq.removeEventListener("change", onChange); else mq.removeListener(onChange); };
  }, [query]);
  return matches;
}

/** Below this width the sidebar is icons-only regardless of the stored choice. */
export const NARROW_QUERY = "(max-width: 1100px)";

function Sidebar({ collapsed, forced }) {
  const route = useRoute();
  const mode = useStore((s) => (s.snapshot && s.snapshot.meta.mode) || (s.settings && s.settings.mode) || null);
  return html`<nav class="sidebar" aria-label="Primary">
    <div class="sidebar-brand">
      <div class="brand-mark"><${BrandMark} /></div>
      <div class="brand-text stack" style="gap:0">
        <span class="brand-name">Portfolio</span>
        <span class="brand-sub">${mode === "user" ? "Your data" : mode === "seed" ? "Demo data" : "Risk workstation"}</span>
      </div>
    </div>
    <div class="sidebar-nav">
      ${ROUTES.map((r) => html`<a key=${r.page} class="nav-item" href=${href(r.path)} aria-current=${route.page === r.page ? "page" : undefined}
          title=${collapsed ? `${r.label}  (g ${r.key})` : undefined} data-tip=${collapsed ? r.label : undefined} data-tip-pos=${collapsed ? "right" : undefined}>
        <span class="nav-icon"><${Icon} name=${ICONS[r.page]} /></span>
        <span class="nav-label">${r.label}</span>
        <span class="nav-hint" aria-hidden="true"><kbd>g</kbd><kbd>${r.key}</kbd></span>
      </a>`)}
    </div>
    <div class="sidebar-foot">
      <button type="button" class="nav-item" onClick=${() => openPalette()} title="Command palette">
        <span class="nav-icon"><${Icon} name="search" /></span>
        <span class="nav-label">Search</span>
        <span class="nav-hint" style="opacity:1"><kbd>${MOD_KEY}</kbd><kbd>K</kbd></span>
      </button>
      ${forced ? null : html`<button type="button" class="nav-item" onClick=${() => setSidebarCollapsed(!collapsed)} title=${collapsed ? "Expand sidebar  ([)" : "Collapse sidebar  ([)"} aria-expanded=${collapsed ? "false" : "true"}>
        <span class="nav-icon"><${Icon} name=${collapsed ? "chevronsRight" : "chevronsLeft"} /></span>
        <span class="nav-label">Collapse</span>
        <span class="nav-hint" aria-hidden="true"><kbd>[</kbd></span>
      </button>`}
    </div>
  </nav>`;
}

function PricesNote() {
  const meta = useStore((s) => s.snapshot && s.snapshot.meta);
  const loadedAt = useStore((s) => s.loadedAt);
  const error = useStore((s) => s.error);
  if (error && !meta) return html`<span class="topbar-status"><span class="neg">${errorMessage(error)}</span></span>`;
  if (!meta) return html`<span class="topbar-status"><span class="faint">loading…</span></span>`;
  const failures = meta.failures && meta.failures.length;
  return html`<span class="topbar-status">
    <span class="truncate" title=${`Snapshot generated ${fmt.dateTime(meta.generated_at)} in ${meta.elapsed_ms} ms · provider ${meta.provider} · cache ${meta.cache_hits} hits / ${meta.cache_misses} misses`}>
      ${fmt.delayNote(meta.prices_as_of, 15)}
    </span>
    ${failures ? html`<span class="sep">·</span><span class="badge badge-serious" title=${meta.failures.map((f) => `${f.name}: ${f.message}`).join("\n")}>${failures} failed</span>` : null}
    ${loadedAt ? html`<span class="sep">·</span><span class="faint" title=${fmt.dateTime(loadedAt, { seconds: true })}>fetched ${fmt.relative(loadedAt)}</span>` : null}
  </span>`;
}

function BenchmarkSelect() {
  const snapshot = useStore((s) => s.snapshot);
  const benchmark = useStore((s) => s.benchmark);
  const list = (snapshot && snapshot.benchmarks) || [];
  if (!list.length) return null;
  // The server's answer wins over a stale stored choice that is not in the list.
  const known = benchmark && list.some((b) => b.symbol === benchmark) ? benchmark : null;
  const value = known || (snapshot && snapshot.selected_benchmark) || (list.find((b) => b.is_default) || list[0]).symbol;
  return html`<label class="row" style="gap:6px" title="Benchmark used for beta, alpha and the comparison lines">
    <span class="faint small nowrap">vs</span>
    <${Select} compact value=${value} onChange=${(v) => setBenchmark(v)} aria-label="Benchmark"
      options=${list.map((b) => ({ value: b.symbol, label: b.label || b.name }))} />
  </label>`;
}

function TopBar() {
  const loading = useStore((s) => s.loading);
  const theme = useStore((s) => s.theme);
  const mode = useStore((s) => (s.snapshot && s.snapshot.meta.mode) || (s.settings && s.settings.mode) || "seed");
  const [refreshing, setRefreshing] = useState(false);
  const onRefresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    try {
      await refreshPrices();
      toast.success("Prices refreshed");
    } catch (err) {
      toast.error(errorMessage(err), { title: "Refresh failed" });
    } finally { setRefreshing(false); }
  };
  return html`<header class="topbar">
    <div class="topbar-group">
      <${ModeBadge} mode=${mode} />
      <${PricesNote} />
    </div>
    <div class="spacer"></div>
    <div class="topbar-group">
      <${BenchmarkSelect} />
      <${Button} variant="ghost" icon="refresh" iconOnly loading=${refreshing} title="Refresh prices (bypasses the 15-minute quote cache)" tipPos="bottom-right" onClick=${onRefresh} ariaLabel="Refresh prices" />
      <${Button} variant="ghost" icon=${theme === "dark" ? "sun" : "moon"} iconOnly title=${theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} tipPos="bottom-right" onClick=${() => toggleTheme()} ariaLabel="Toggle theme" />
      <${Button} variant="secondary" icon="search" onClick=${() => openPalette()} title="Command palette">
        <span class="muted">Search</span><kbd>${MOD_KEY}</kbd><kbd>K</kbd>
      <//>
    </div>
    <div class="topbar-progress" data-active=${loading ? "true" : "false"} aria-hidden="true"></div>
  </header>`;
}

export function Shell({ children }) {
  const stored = useStore((s) => s.sidebarCollapsed);
  const narrow = useMediaQuery(NARROW_QUERY);
  const collapsed = stored || narrow;
  const route = useRoute();
  // scroll the main pane to top on page change (not on drawer param changes)
  useEffect(() => {
    const main = document.querySelector(".main");
    if (main) main.scrollTo({ top: 0 });
  }, [route.page]);
  return html`<div class="shell" data-collapsed=${collapsed ? "true" : "false"} data-narrow=${narrow ? "true" : "false"}>
    <${Sidebar} collapsed=${collapsed} forced=${narrow} />
    <${TopBar} />
    <main class="main" id="main" tabindex="-1">${children}</main>
  </div>`;
}

export { navigate, app };
export default Shell;
