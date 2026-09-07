/* Settings — the data source, the prices, the analysis window, appearance,
   shortcuts and what this is. Props: { route, snapshot }.

   Two rules the page enforces on screen: the mode (demo or live) is never a
   guess — it is named, explained and switched only through a confirm step —
   and prices are never called live: the provenance line says "delayed" and
   "close" because that is what they are. */

import { html, useEffect, useState } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { Badge, ModeBadge } from "/static/components/Badge.js";
import { Banner } from "/static/components/Banner.js";
import { Button } from "/static/components/Button.js";
import { Field, Select } from "/static/components/Field.js";
import { Segmented } from "/static/components/Segmented.js";
import { confirm } from "/static/components/Modal.js";
import { toast } from "/static/components/Toast.js";
import { Icon } from "/static/components/Icons.js";
import { app, useStore, setBenchmark, setLookback, LOOKBACKS } from "/static/lib/store.js";
import { getTheme, setTheme } from "/static/lib/theme.js";
import { ROUTES } from "/static/lib/router.js";
import { api, getSettings, putSettings, refreshPrices, loadSnapshot, errorMessage } from "/static/lib/api.js";
import { MOD_KEY } from "/static/components/CommandPalette.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- helpers ---------------- */

function bytes(n) {
  if (n == null) return fmt.DASH;
  if (n < 1024) return `${fmt.int(n)} B`;
  if (n < 1024 * 1024) return `${fmt.num(n / 1024, { decimals: 0 })} KB`;
  return `${fmt.num(n / (1024 * 1024), { decimals: 1 })} MB`;
}

function themeChoice() {
  let stored = null;
  try { stored = localStorage.getItem("pt.theme"); } catch (_) { /* private mode */ }
  return stored === "light" || stored === "dark" ? stored : "system";
}

function applyTheme(choice) {
  if (choice === "system") {
    try { localStorage.removeItem("pt.theme"); } catch (_) { /* private mode */ }
    const light = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
    document.documentElement.setAttribute("data-theme", light ? "light" : "dark");
    app.patch({ theme: light ? "light" : "dark" });
  } else {
    setTheme(choice);
  }
}

const Row = ({ label, value, mono = false }) => html`<dt>${label}</dt><dd class=${mono ? "mono" : ""}>${value == null || value === "" ? fmt.DASH : value}</dd>`;

/* ---------------- cards ---------------- */

function DataSourceCard({ settings, snapshot, onChanged }) {
  const [busy, setBusy] = useState(false);
  const mode = (settings && settings.mode) || (snapshot && snapshot.meta.mode) || "seed";
  const live = mode === "user";
  const switchTo = async (next) => {
    if (next === mode || busy) return;
    const ok = await confirm({
      title: next === "user" ? "Switch to your own data?" : "Switch to the demo data?",
      body: next === "user"
        ? "Every page will read your instruments and ledger. The demo book disappears until you switch back; nothing is deleted either way."
        : "Every page will show the seed instruments and the synthetic ledger — none of it is your portfolio. Your own data stays on disk untouched.",
      confirmLabel: next === "user" ? "Use my data" : "Show the demo",
    });
    if (!ok) return;
    setBusy(true);
    try {
      await putSettings({ mode: next });
      await loadSnapshot();
      toast.success(next === "user" ? "Reading your own ledger." : "Reading the demo data.", { title: next === "user" ? "Live data" : "Demo data" });
      onChanged();
    } catch (err) { toast.error(errorMessage(err), { title: "Could not switch" }); }
    finally { setBusy(false); }
  };
  return html`<${Card} title="Data source" caption="Which ledger every page reads. The mode is always shown in the top bar, because demo figures shown as real and real figures shown as demo are both failures worth preventing loudly.">
    <div class="stack gap-3">
      <div class="set-modes">
        <div class="set-mode" data-active=${!live ? "true" : "false"}>
          <span class="set-mode-title"><${ModeBadge} mode="seed" />Demo data</span>
          <span class="set-mode-body">Ten real reference instruments and a synthetic ledger — a partial sell, a multi-currency purchase, a dividend, a fee, a voided-and-corrected entry. Prices are real where the provider is; the positions are invented.</span>
          <span class="set-mode-foot">${!live ? html`<${Icon} name="checkCircle" size=${13} /> active now` : html`<${Button} size="sm" variant="secondary" loading=${busy} onClick=${() => switchTo("seed")}>Show the demo<//>`}</span>
        </div>
        <div class="set-mode" data-active=${live ? "true" : "false"}>
          <span class="set-mode-title"><${ModeBadge} mode="user" />My data</span>
          <span class="set-mode-body">Your instruments and your ledger, kept as plain CSV files that outlive this tool. Never committed to git; the demo never reads them and they never read the demo.</span>
          <span class="set-mode-foot">${live ? html`<${Icon} name="checkCircle" size=${13} /> active now` : html`<${Button} size="sm" variant="primary" loading=${busy} onClick=${() => switchTo("user")}>Use my data<//>`}</span>
        </div>
      </div>
      <dl class="definition-list set-dl">
        <${Row} label="Data directory" value=${settings && settings.data_dir} mono />
        <${Row} label="Files" value="instruments.csv (rewritten wholesale — reference data), transactions.csv (append-only), amendments.csv (voids, append-only)" />
        <${Row} label="Backup" value="Copy the directory. There is no database to export; the CSVs are the source of truth and every other number is derived from them." />
        <${Row} label="Environment" value=${html`<code>PORTFOLIO_DATA_MODE=user</code> starts in live mode; <code>portfolio serve --mode user</code> does the same.`} />
      </dl>
    </div>
  <//>`;
}

function PricesCard({ settings, snapshot, onChanged }) {
  const [busy, setBusy] = useState(null);
  const meta = snapshot && snapshot.meta;
  const delay = snapshot ? Math.max(0, ...(snapshot.holdings || []).map((h) => h.price_delay_minutes || 0)) : null;
  const cache = settings && settings.cache;
  const refresh = async () => {
    setBusy("refresh");
    try { await refreshPrices(); toast.success("Quotes refetched; history extended incrementally.", { title: "Prices refreshed" }); onChanged(); }
    catch (err) { toast.error(errorMessage(err), { title: "Refresh failed" }); }
    finally { setBusy(null); }
  };
  const clear = async () => {
    const ok = await confirm({ title: "Clear the price cache?", body: "Every cached quote and price history is deleted and refetched on the next load. With a real provider that costs one request per instrument; with the offline provider it is free.", confirmLabel: "Clear cache", danger: true });
    if (!ok) return;
    setBusy("clear");
    try { await api.post("/cache/clear", {}); await getSettings(); await loadSnapshot(); toast.success("Cache cleared and rebuilt.", { title: "Price cache" }); onChanged(); }
    catch (err) { toast.error(errorMessage(err), { title: "Could not clear the cache" }); }
    finally { setBusy(null); }
  };
  return html`<${Card} title="Prices" caption="Delayed, never real time. Fetched once, cached on disk, extended incrementally. Nothing polls."
    actions=${html`<${Button} size="sm" variant="secondary" icon="refresh" loading=${busy === "refresh"} onClick=${refresh}>Refresh prices<//>`}>
    <div class="stack gap-3">
      <dl class="definition-list set-dl">
        <${Row} label="Provider" value=${settings ? `${settings.provider}${settings.provider === "fixture" ? " — deterministic synthetic prices, no network" : " — free, keyless; OpenFIGI for identity"}` : null} />
        <${Row} label="Base currency" value=${settings && settings.base_currency} />
        <${Row} label="Prices as of" value=${meta ? fmt.delayNote(meta.prices_as_of, delay || null, { prefix: "" }).trim() : null} />
        <${Row} label="Quote cache" value=${settings ? `${fmt.int(settings.quote_ttl_minutes)} minutes, then the next load refetches the quote` : null} />
        <${Row} label="Last forced refresh" value=${settings && settings.last_refresh ? fmt.dateTime(settings.last_refresh) : "not this session"} />
        <${Row} label="Snapshot" value=${meta ? `${fmt.int(meta.elapsed_ms)} ms · ${fmt.int(meta.cache_hits)} cache hits, ${fmt.int(meta.cache_misses)} misses · generated ${fmt.dateTime(meta.generated_at)}` : null} />
      </dl>
      <div class="set-stats">
        <div class="set-stat"><span class="label">Cached symbols</span><span class="value">${fmt.int(cache && cache.symbols)}</span></div>
        <div class="set-stat"><span class="label">Price rows</span><span class="value">${fmt.int(cache && cache.rows)}</span></div>
        <div class="set-stat"><span class="label">On disk</span><span class="value">${bytes(cache && cache.size_bytes)}</span></div>
        <div class="set-stat"><span class="label">Oldest · newest fetch</span><span class="value" title=${cache ? `${cache.oldest || ""} · ${cache.newest || ""}` : ""}>${cache && cache.newest ? fmt.relative(cache.newest) : fmt.DASH}</span></div>
      </div>
      <div class="set-actions">
        <${Button} size="sm" variant="danger" icon="trash" loading=${busy === "clear"} onClick=${clear}>Clear cache<//>
        <span class="set-para xs">${cache ? cache.path : ""}</span>
      </div>
      <p class="set-para">History is fetched from the last cached date forward, so a restart never refetches what is already on disk. If the provider is unreachable, cached history is served and the price is marked stale rather than blanked.</p>
    </div>
  <//>`;
}

function AnalysisCard({ snapshot }) {
  const benchmark = useStore((s) => s.benchmark);
  const lookback = useStore((s) => s.lookback);
  const benchmarks = (snapshot && snapshot.benchmarks) || [];
  const selected = benchmark || (snapshot && snapshot.selected_benchmark) || (benchmarks[0] && benchmarks[0].symbol) || "";
  const chosen = benchmarks.find((b) => b.symbol === selected);
  const risk = snapshot && snapshot.risk;
  return html`<${Card} title="Analysis" caption="The benchmark and the window behind every risk figure. Changing either reloads the snapshot.">
    <div class="stack gap-4">
      <${Field} label="Benchmark for beta and the comparison lines" hint=${chosen ? chosen.note : "All options are EUR-quoted, so a beta measures the index and not a currency as well."}>
        <${Select} value=${selected} options=${benchmarks.map((b) => ({ value: b.symbol, label: `${b.index} — ${b.name} (${b.symbol})` }))} onChange=${(v) => setBenchmark(v)} />
      <//>
      <${Field} label="Risk window, trading days" hint="The covariance matrix, volatility, VaR and the correlation grid are all computed over this many aligned daily returns. Shorter reacts faster and is noisier; 252 is one year.">
        <div class="set-lookbacks">
          <${Segmented} value=${lookback} onChange=${(v) => setLookback(Number(v))} options=${LOOKBACKS.map((d) => ({ value: d, label: `${d}` }))} ariaLabel="Risk window" />
          <span class="set-para xs">${risk && risk.window ? `Effective: ${fmt.int(risk.window.effective)} returns${risk.window.effective < risk.window.requested ? ` (shortened by ${risk.window.binding_name || risk.window.binding_isin})` : ""}` : ""}</span>
        </div>
      <//>
      <div class="set-threshold">
        <span class="value">${fmt.num(risk ? risk.threshold : null, { decimals: 2 })}</span>
        <span class="set-para">correlation above which two holdings are flagged as one bet. Set in <code>core/risk.py</code> on measured evidence and deliberately not tunable from here: a threshold chosen to fit one observation is a threshold that means nothing.</span>
      </div>
    </div>
  <//>`;
}

function AppearanceCard() {
  const [choice, setChoice] = useState(themeChoice());
  const theme = useStore((s) => s.theme);
  return html`<${Card} title="Appearance" caption="Dark first, light selected on its own — not an inversion.">
    <div class="stack gap-3">
      <${Field} label="Theme" hint=${`Currently ${theme}. ${MOD_KEY} ⇧ L toggles it from anywhere.`}>
        <${Segmented} value=${choice} onChange=${(v) => { setChoice(v); applyTheme(v); }} options=${[{ value: "system", label: "System" }, { value: "dark", label: "Dark" }, { value: "light", label: "Light" }]} ariaLabel="Theme" />
      <//>
      <p class="set-para xs">Motion is reduced automatically when the operating system asks for it. Figures use tabular numerals in every column and proportional ones on the hero figures.</p>
    </div>
  <//>`;
}

function ShortcutsCard() {
  const K = (k) => html`<kbd>${k}</kbd>`;
  return html`<${Card} title="Keyboard" caption="Everything is reachable without the mouse.">
    <table class="set-keys">
      <thead><tr><th>Action</th><th>Keys</th></tr></thead>
      <tbody>
        <tr class="set-keys-group"><td colspan="2">Navigate</td></tr>
        ${ROUTES.map((r) => html`<tr key=${r.page}><td>${r.label}</td><td>${K("g")}<span class="then">then</span>${K(r.key)}</td></tr>`)}
        <tr class="set-keys-group"><td colspan="2">Global</td></tr>
        <tr><td>Command palette — pages, actions, any holding by name or ISIN</td><td>${K(MOD_KEY)} ${K("K")} <span class="then">or</span> ${K("/")}</td></tr>
        <tr><td>Refresh prices</td><td>${K(MOD_KEY)} ${K("⇧")} ${K("R")}</td></tr>
        <tr><td>Toggle theme</td><td>${K(MOD_KEY)} ${K("⇧")} ${K("L")}</td></tr>
        <tr><td>Collapse the sidebar</td><td>${K("[")}</td></tr>
        <tr><td>Close a drawer or dialog</td><td>${K("esc")}</td></tr>
        <tr><td>Previous / next holding while the detail is open</td><td>${K("←")} ${K("→")}</td></tr>
        <tr><td>Shortcut help</td><td>${K("?")}</td></tr>
      </tbody>
    </table>
  <//>`;
}

function AboutCard({ settings, snapshot }) {
  const [health, setHealth] = useState(null);
  useEffect(() => { api.get("/health").then(setHealth).catch(() => {}); }, []);
  const version = (health && health.version) || (settings && settings.version);
  return html`<${Card} title="About" caption="A personal portfolio tracker for European-listed ETFs and ETCs, keyed on ISIN, with risk analytics that can be read against a textbook.">
    <div class="stack gap-3">
      <dl class="definition-list set-dl">
        <${Row} label="Version" value=${version} />
        <${Row} label="Mode · provider" value=${settings ? `${settings.mode === "user" ? "live" : "demo"} · ${settings.provider}` : null} />
        <${Row} label="Universe" value=${snapshot ? `${fmt.count(snapshot.meta.instrument_count, "instrument")}, ${fmt.count(snapshot.meta.transaction_count, "live transaction")}` : null} />
        <${Row} label="API" value=${html`<a href="/api/docs" target="_blank" rel="noopener">/api/docs</a> — every number on these pages comes from one JSON snapshot; the client formats and never computes.`} />
        <${Row} label="Architecture" value=${html`<code>docs/ARCHITECTURE.md</code> — the contract between core (pure mathematics), data (providers and cache), api and this client. The layering is enforced by tests.`} />
      </dl>
      <${Banner} kind="neutral" icon="info">Not real-time prices, not tax reporting, not investment advice. Cost basis uses weighted average cost, which is right for risk and performance and not for most capital-gains regimes.<//>
      <div class="row wrap" style="gap:6px">
        <${Badge} kind="neutral" outline>FastAPI<//><${Badge} kind="neutral" outline>Preact + htm<//><${Badge} kind="neutral" outline>ECharts<//><${Badge} kind="neutral" outline>numpy · pandas<//><${Badge} kind="neutral" outline>SQLite cache<//><${Badge} kind="neutral" outline>yfinance · OpenFIGI<//>
      </div>
    </div>
  <//>`;
}

/* ---------------- page ---------------- */

export default function SettingsPage({ snapshot }) {
  const settings = useStore((s) => s.settings);
  const [error, setError] = useState(null);
  const refresh = () => getSettings().then(() => setError(null)).catch((err) => setError(err));
  useEffect(() => { refresh(); }, []);
  const mode = (settings && settings.mode) || (snapshot && snapshot.meta.mode) || "seed";
  return html`<${Page} title="Settings" subtitle=${`${mode === "user" ? "Live data" : "Demo data"} · ${settings ? settings.provider : "…"} · ${settings ? settings.data_dir : ""}`}>
    <div class="stack gap-4">
      ${error ? html`<${Banner} kind="error" title="Could not load the settings." action=${html`<${Button} size="sm" variant="secondary" onClick=${refresh}>Retry<//>`}>${errorMessage(error)}<//>` : null}
      <div class="grid">
        <div class="col-6 stack gap-4">
          <${DataSourceCard} settings=${settings} snapshot=${snapshot} onChanged=${refresh} />
          <${AnalysisCard} snapshot=${snapshot} />
          <${AppearanceCard} />
        </div>
        <div class="col-6 stack gap-4">
          <${PricesCard} settings=${settings} snapshot=${snapshot} onChanged=${refresh} />
          <${ShortcutsCard} />
          <${AboutCard} settings=${settings} snapshot=${snapshot} />
        </div>
      </div>
    </div>
  <//>`;
}
