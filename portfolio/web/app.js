/* Application entry: registers chart themes, starts the router, mounts the
   shell + routed page + drawer + modal + command palette + toasts, loads the
   snapshot, and wires global keyboard shortcuts. A global error boundary
   renders a readable card — never a blank screen. */

import { html, render, useEffect, useErrorBoundary, useState } from "/static/vendor/preact-htm.module.js";
import { app, useStore, openDrawer, closeDrawer, openPalette, closePalette, openModal, closeModal, setSidebarCollapsed } from "/static/lib/store.js";
import { registerChartThemes, watchSystemTheme, toggleTheme } from "/static/lib/theme.js";
import { startRouter, useRoute, navigate, ROUTES } from "/static/lib/router.js";
import { loadSnapshot, getSettings, putSettings, refreshPrices, errorMessage } from "/static/lib/api.js";
import { Shell } from "/static/components/Shell.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { Button } from "/static/components/Button.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Banner } from "/static/components/Banner.js";
import { DrawerHost } from "/static/components/Drawer.js";
import { ModalHost } from "/static/components/Modal.js";
import { CommandPalette, MOD_KEY } from "/static/components/CommandPalette.js";
import { Toasts, toast } from "/static/components/Toast.js";
import { pages } from "/static/pages/index.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- error card ---------------- */

function ErrorCard({ error, onRetry }) {
  const message = errorMessage(error);
  const stack = error && error.stack ? String(error.stack) : "";
  return html`<div class="page"><div class="error-card" role="alert">
    <h2>Something went wrong</h2>
    <p class="muted">${message}</p>
    ${stack ? html`<pre>${stack}</pre>` : null}
    <div class="row" style="margin-top:16px">
      <${Button} variant="primary" onClick=${onRetry}>Try again<//>
      <${Button} variant="secondary" onClick=${() => location.reload()}>Reload<//>
      <${Button} variant="ghost" onClick=${() => navigate("/overview")}>Go to overview<//>
    </div>
  </div></div>`;
}

function NotFound({ route }) {
  return html`<${Page} title="Not found">
    <${Card}><${EmptyState} icon="search" title="No such page" body=${`Nothing lives at #${route.path}.`}
      action=${html`<${Button} variant="primary" onClick=${() => navigate("/overview")}>Go to overview<//>`} /><//>
  <//>`;
}

/* ---------------- shortcuts help ---------------- */

function ShortcutsHelp() {
  return html`<div class="stack gap-3 small">
    <div class="caps">Navigate</div>
    <dl class="definition-list" style="grid-template-columns: 1fr max-content">
      ${ROUTES.map((r) => html`<dt style="color:var(--color-text)">${r.label}</dt><dd><kbd>g</kbd> <kbd>${r.key}</kbd></dd>`)}
    </dl>
    <div class="caps">Global</div>
    <dl class="definition-list" style="grid-template-columns: 1fr max-content">
      <dt style="color:var(--color-text)">Command palette</dt><dd><kbd>${MOD_KEY}</kbd> <kbd>K</kbd> or <kbd>/</kbd></dd>
      <dt style="color:var(--color-text)">Refresh prices</dt><dd><kbd>${MOD_KEY}</kbd> <kbd>⇧</kbd> <kbd>R</kbd></dd>
      <dt style="color:var(--color-text)">Toggle theme</dt><dd><kbd>${MOD_KEY}</kbd> <kbd>⇧</kbd> <kbd>L</kbd></dd>
      <dt style="color:var(--color-text)">Collapse sidebar</dt><dd><kbd>[</kbd></dd>
      <dt style="color:var(--color-text)">Close drawer / dialog</dt><dd><kbd>esc</kbd></dd>
      <dt style="color:var(--color-text)">This help</dt><dd><kbd>?</kbd></dd>
    </dl>
  </div>`;
}

function showShortcuts() {
  openModal({ title: "Keyboard shortcuts", body: html`<${ShortcutsHelp} />`, footer: html`<${Button} variant="primary" onClick=${closeModal} data-autofocus>Done<//>` });
}

/* ---------------- palette actions ---------------- */

async function doRefresh() {
  try { await refreshPrices(); toast.success("Prices refreshed"); }
  catch (err) { toast.error(errorMessage(err), { title: "Refresh failed" }); }
}

async function switchMode() {
  const cur = app.get().settings?.mode || app.get().snapshot?.meta.mode || "seed";
  const next = cur === "seed" ? "user" : "seed";
  try {
    await putSettings({ mode: next });
    toast.success(`Switched to ${next === "seed" ? "demo" : "live"} data`);
    await loadSnapshot();
  } catch (err) { toast.error(errorMessage(err), { title: "Could not switch mode" }); }
}

const PALETTE_ACTIONS = [
  { id: "act:refresh", label: "Refresh prices", sub: "bypass the 15-minute quote cache", group: "Actions", icon: "refresh", keywords: "reload fetch quotes", run: doRefresh },
  { id: "act:theme", label: "Toggle theme", sub: "dark / light", group: "Actions", icon: "sun", keywords: "dark light mode appearance", run: () => toggleTheme() },
  { id: "act:mode", label: "Switch to demo / live data", sub: "seed ↔ user", group: "Actions", icon: "layers", keywords: "seed user demo live mode data", run: switchMode },
  { id: "act:allocate", label: "Direct new money", sub: "where should a purchase go", group: "Actions", icon: "wallet", keywords: "allocate buy purchase contribute cash new money", run: () => navigate("/allocate") },
  { id: "act:add-tx", label: "Add transaction", group: "Actions", icon: "plus", keywords: "buy sell dividend fee ledger new", run: () => navigate("/transactions?new=1") },
  { id: "act:add-instr", label: "Add instrument", group: "Actions", icon: "plus", keywords: "isin resolve new universe", run: () => navigate("/instruments?new=1") },
  { id: "act:sidebar", label: "Toggle sidebar", group: "Actions", icon: "sidebar", hint: ["["], keywords: "collapse expand nav", run: () => setSidebarCollapsed(!app.get().sidebarCollapsed) },
  { id: "act:help", label: "Keyboard shortcuts", group: "Actions", icon: "keyboard", hint: ["?"], keywords: "help keys", run: showShortcuts },
  { id: "act:gallery", label: "Kitchen sink", sub: "every component, from the live snapshot", group: "Actions", icon: "layers", keywords: "gallery dev components visual regression", run: () => navigate("/kitchensink") },
];

/* ---------------- keyboard ---------------- */

function isTyping(e) {
  const el = e.target;
  if (!el || !el.tagName) return false;
  const tag = el.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
}

function installShortcuts() {
  let pendingG = 0;
  window.addEventListener("keydown", (e) => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && !e.altKey && e.key.toLowerCase() === "k") {
      e.preventDefault();
      if (app.get().paletteOpen) closePalette(); else openPalette();
      return;
    }
    if (mod && e.shiftKey && e.key.toLowerCase() === "r") { e.preventDefault(); doRefresh(); return; }
    if (mod && e.shiftKey && e.key.toLowerCase() === "l") { e.preventDefault(); toggleTheme(); return; }
    if (mod || e.altKey || isTyping(e)) return;
    if (app.get().paletteOpen || app.get().modal) return;
    const now = Date.now();
    if (e.key === "g") { pendingG = now; return; }
    if (pendingG && now - pendingG < 1500) {
      const r = ROUTES.find((x) => x.key === e.key);
      pendingG = 0;
      if (r) { e.preventDefault(); navigate(r.path); return; }
    }
    if (e.key === "/") { e.preventDefault(); openPalette(); return; }
    if (e.key === "?") { e.preventDefault(); showShortcuts(); return; }
    if (e.key === "[") { e.preventDefault(); setSidebarCollapsed(!app.get().sidebarCollapsed); return; }
  });
}

/* ---------------- root ---------------- */

function RoutedPage({ route }) {
  const snapshot = useStore((s) => s.snapshot);
  const PageComponent = pages[route.page];
  if (!PageComponent) return html`<${NotFound} route=${route} />`;
  return html`<${PageComponent} key=${route.page} route=${route} snapshot=${snapshot} />`;
}

function LoadErrorBanner() {
  const error = useStore((s) => s.error);
  const snapshot = useStore((s) => s.snapshot);
  const loading = useStore((s) => s.loading);
  if (!error) return null;
  return html`<div style="padding: 16px var(--gutter) 0">
    <${Banner} kind="error" title=${snapshot ? "Could not refresh the snapshot." : "Could not load the portfolio."}
      action=${html`<${Button} size="sm" variant="secondary" loading=${loading} onClick=${() => loadSnapshot().catch(() => {})}>Retry<//>`}>
      ${errorMessage(error)}${snapshot ? " Showing the last good data." : ""}
    <//>
  </div>`;
}

function Root() {
  const [error, resetError] = useErrorBoundary((err) => console.error("[portfolio] render error", err));
  const route = useRoute();
  const benchmark = useStore((s) => s.benchmark);
  const lookback = useStore((s) => s.lookback);
  const [booted, setBooted] = useState(false);

  // load on start and whenever benchmark / lookback change
  useEffect(() => {
    loadSnapshot().catch((err) => { if (!booted) toast.error(errorMessage(err), { title: "Snapshot failed" }); });
    setBooted(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [benchmark, lookback]);

  // route-driven drawer: #/holdings/:isin
  useEffect(() => {
    const isin = route.page === "holdings" ? route.params.isin : null;
    const cur = app.get().drawer;
    if (isin) {
      const Detail = pages.holdingDetail;
      const snap = app.get().snapshot;
      const h = snap && ((snap.holdings || []).find((x) => x.isin === isin) || (snap.watchlist || []).find((x) => x.isin === isin));
      openDrawer({
        routeDriven: true,
        title: h ? fmt.shortName(h.name, 40) : isin,
        subtitle: h ? [h.symbol, h.isin].filter(Boolean).join(" · ") : "Holding",
        body: () => html`<${Detail} isin=${isin} />`,
        onAfterClose: () => { if (app.get().route.params.isin === isin) navigate("/holdings"); },
      });
    } else if (cur && cur.routeDriven) {
      app.patch({ drawer: null });
    }
  }, [route.page, route.params.isin]);

  // refresh the drawer title once the snapshot arrives
  const snapshot = useStore((s) => s.snapshot);
  useEffect(() => {
    const cur = app.get().drawer;
    if (!cur || !cur.routeDriven || !snapshot) return;
    const isin = app.get().route.params.isin;
    const h = isin && ((snapshot.holdings || []).find((x) => x.isin === isin) || (snapshot.watchlist || []).find((x) => x.isin === isin));
    if (h && cur.title !== fmt.shortName(h.name, 40)) app.patch({ drawer: { ...cur, title: fmt.shortName(h.name, 40), subtitle: [h.symbol, h.isin].filter(Boolean).join(" · ") } });
  }, [snapshot]);

  return html`
    <${Shell}>
      <${LoadErrorBanner} />
      ${error ? html`<${ErrorCard} error=${error} onRetry=${resetError} />` : html`<${RoutedPage} route=${route} />`}
    <//>
    <${DrawerHost} />
    <${ModalHost} />
    <${CommandPalette} actions=${PALETTE_ACTIONS} />
    <${Toasts} />
  `;
}

/* ---------------- boot ---------------- */

function boot() {
  registerChartThemes();
  watchSystemTheme();
  startRouter();
  installShortcuts();
  render(html`<${Root} />`, document.getElementById("app"));
  getSettings().catch(() => { /* settings are optional for rendering */ });
}

window.addEventListener("error", (e) => console.error("[portfolio] uncaught", e.error || e.message));
window.addEventListener("unhandledrejection", (e) => {
  const reason = e.reason;
  if (reason && reason.name === "AbortError") return;
  console.error("[portfolio] unhandled rejection", reason);
});

boot();
