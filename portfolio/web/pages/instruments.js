/* Instruments — the universe, and adding by ISIN safely. Props: { route, snapshot }.

   Order, and why:
   1. The add flow is a card at the top with three steps — ISIN, choose a
      listing, confirm — because it is the one place a wrong key could enter
      the system. It mirrors data/resolve.py: resolve, confirm, save; never a
      text box that writes. The client checks the ISO 6166 check digit for
      immediate feedback; the server re-validates. THIN and STALE listings are
      shown but never preselected; FAILED ones are disabled with their reason;
      refused (US) ones sit in their own group with the collision story.
   2. The universe table: every instrument with the symbol the running
      provider fetches, its venue, quote and base currency, class, whether it
      is held (transaction count) or on the watchlist, and whether it is
      active. Row click opens the edit drawer (local state, ESC closes).
   3. The edit drawer: fields with hand-set values marked, the provider
      symbol override, deactivate / reactivate, and delete behind a confirm
      that shows the transaction count; a 409 shows the server's sentence. */

import { html, useCallback, useEffect, useMemo, useState } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { DataTable } from "/static/components/DataTable.js";
import { Drawer } from "/static/components/Drawer.js";
import { Modal } from "/static/components/Modal.js";
import { Badge, VerdictBadge } from "/static/components/Badge.js";
import { Banner } from "/static/components/Banner.js";
import { Notice } from "/static/components/Notice.js";
import { Button } from "/static/components/Button.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Field, Input, Select, Textarea } from "/static/components/Field.js";
import { Icon } from "/static/components/Icons.js";
import { toast } from "/static/components/Toast.js";
import { navigate } from "/static/lib/router.js";
import { listInstruments, resolveIsin, saveInstrument, patchInstrument, deleteInstrument, loadSnapshot, errorMessage } from "/static/lib/api.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- ISIN (ISO 6166) ---------------- */

const ISIN_SHAPE = /^[A-Z]{2}[A-Z0-9]{9}[0-9]$/;

/** Check digit for the first 11 characters: expand letters to numbers
    (A=10 … Z=35), then Luhn from the right. Mirrors core/models.py. */
export function isinCheckDigit(body) {
  const digits = String(body).toUpperCase().split("").map((c) => String(parseInt(c, 36))).join("");
  let total = 0;
  for (let i = 0; i < digits.length; i++) {
    let d = Number(digits[digits.length - 1 - i]);
    if (i % 2 === 0) { d *= 2; if (d > 9) d -= 9; }
    total += d;
  }
  return (10 - (total % 10)) % 10;
}

/** { isin, state: "empty"|"partial"|"invalid"|"valid", message } — for live feedback only. */
export function checkIsin(raw) {
  const s = String(raw || "").replace(/\s+/g, "").toUpperCase();
  if (!s) return { isin: s, state: "empty", message: "Twelve characters: a two-letter country code, nine letters or digits, then a check digit." };
  if (!/^[A-Z0-9]*$/.test(s)) return { isin: s, state: "invalid", message: "Only letters and digits belong in an ISIN." };
  if (s.length < 12) { const n = 12 - s.length; return { isin: s, state: "partial", message: `${n} more character${n === 1 ? "" : "s"} to go.` }; }
  if (s.length > 12) return { isin: s, state: "invalid", message: `${s.length} characters — an ISIN has exactly twelve.` };
  if (!ISIN_SHAPE.test(s)) return { isin: s, state: "invalid", message: "Not the shape of an ISIN: two letters, nine letters or digits, then one digit." };
  const expected = isinCheckDigit(s.slice(0, 11));
  if (expected !== Number(s[11])) return { isin: s, state: "invalid", message: `Check digit does not match: ${s.slice(0, 11)} needs ${expected}, not ${s[11]}. One wrong character here would create a second, empty instrument.` };
  return { isin: s, state: "valid", message: "Check digit verified (ISO 6166). The server re-validates before it resolves anything." };
}

/* ---------------- shared bits ---------------- */

const ASSET_CLASSES = [
  { value: "ETF", label: "ETF — UCITS exchange-traded fund" },
  { value: "ETC", label: "ETC — exchange-traded commodity" },
  { value: "EQUITY", label: "Equity" },
  { value: "FUND", label: "Fund" },
  { value: "OTHER", label: "Other" },
];
const ETC_HINT = "ETC is a collateralised note, not a UCITS fund, and carries issuer credit risk.";
const BASE_HINT = "The fund's reporting currency from its factsheet. It cannot be inferred from the listing — a USD-base fund often quotes in EUR on Xetra.";

/** Step indicator shared by the add flow and the import wizard (styles in instruments.css). */
export function Steps({ steps, current }) {
  return html`<ol class="wz-steps" aria-label="Steps">
    ${steps.map((label, i) => {
      const n = i + 1;
      const state = n < current ? "done" : n === current ? "active" : "todo";
      return html`<li key=${label} class="wz-step" data-state=${state} aria-current=${state === "active" ? "step" : undefined}>
          <span class="wz-step-n">${state === "done" ? html`<${Icon} name="check" size=${11} stroke=${2.5} />` : n}</span>
          <span class="wz-step-label">${label}</span>
        </li>${i < steps.length - 1 ? html`<li key=${label + "-sep"} class="wz-step-sep" aria-hidden="true"></li>` : null}`;
    })}
  </ol>`;
}

/** The symbol the running provider fetches (market.symbol_for falls back to yfinance). */
function symbolEntry(inst, provider) {
  const ps = (inst && inst.provider_symbols) || {};
  const overrides = (inst && inst.manual_overrides) || [];
  for (const k of [provider, "yfinance", ...Object.keys(ps)]) {
    if (k && ps[k]) return { provider: k, symbol: ps[k], manual: overrides.includes(`provider_symbols.${k}`) };
  }
  return { provider: null, symbol: (inst && inst.primary_symbol) || null, manual: false };
}

function Manual({ inst, field }) {
  const on = ((inst && inst.manual_overrides) || []).includes(field);
  return on ? html`<${Badge} kind="info" outline title="Set by hand; re-resolution will not overwrite it">manual<//>` : null;
}

function ObsSpan({ c }) {
  if (!c.observations) return html`<span class="faint">no history</span>`;
  return html`<span class="num">${fmt.int(c.observations)} obs · ${fmt.date(c.first_date)} → ${fmt.date(c.last_date)}</span>`;
}

/* ---------------- add flow ---------------- */

function CandidateRow({ c, selected, recommended, onSelect }) {
  const disabled = !c.selectable;
  return html`<button type="button" role="radio" aria-checked=${selected ? "true" : "false"} disabled=${disabled}
      class=${["ins-candidate", selected ? "selected" : ""].filter(Boolean).join(" ")} onClick=${onSelect}
      title=${disabled ? "Not selectable: " + (c.reasons || []).join(" ") : undefined}>
    <span class="ins-cand-radio" aria-hidden="true"></span>
    <div class="ins-cand-main">
      <div class="row wrap" style="gap:8px">
        <${VerdictBadge} verdict=${c.verdict} />
        <span class="mono strong">${c.yahoo_symbol}</span>
        <span class="truncate" style="max-width:320px" title=${c.name}>${fmt.text(c.name)}</span>
        ${recommended ? html`<${Badge} kind="info" icon="check">recommended<//>` : null}
        ${disabled ? html`<${Badge} kind="neutral" outline>not selectable<//>` : null}
      </div>
      <div class="ins-cand-facts small muted">
        <span>${c.venue_label}${c.reported_exchange ? ` (${c.reported_exchange})` : ""}</span>
        <span class="num strong">${c.currency || fmt.DASH}</span>
        <${ObsSpan} c=${c} />
        ${c.mic ? html`<span class="mono xs faint">${c.mic}</span>` : null}
      </div>
      ${c.reasons && c.reasons.length ? html`<ul class="ins-cand-reasons">${c.reasons.map((r) => html`<li key=${r}><${Icon} name="alertTriangle" size=${12} stroke=${2} /><span>${r}</span></li>`)}</ul>` : null}
    </div>
  </button>`;
}

function RefusedRow({ c }) {
  return html`<div class="ins-candidate ins-refused" aria-disabled="true">
    <span class="ins-cand-radio" aria-hidden="true" style="visibility:hidden"></span>
    <div class="ins-cand-main">
      <div class="row wrap" style="gap:8px">
        <${VerdictBadge} verdict=${c.verdict} />
        <span class="mono strong">${c.yahoo_symbol}</span>
        <span class="truncate" style="max-width:320px" title=${c.name}>${fmt.text(c.name)}</span>
      </div>
      <div class="ins-cand-facts small muted">
        <span>${c.venue_label}${c.reported_exchange ? ` (${c.reported_exchange})` : ""}</span>
        <span class="num strong">${c.currency || fmt.DASH}</span>
        <${ObsSpan} c=${c} />
      </div>
      ${c.reasons && c.reasons.length ? html`<ul class="ins-cand-reasons">${c.reasons.map((r) => html`<li key=${r}><${Icon} name="alertOctagon" size=${12} stroke=${2} /><span>${r}</span></li>`)}</ul>` : null}
    </div>
  </div>`;
}

const FILTER_LABEL = { us: "US", mtf: "MTF", german_regional: "German regional", dark: "dark", other: "other" };
function filterLabel(cls) { return FILTER_LABEL[cls] || String(cls).replace(/_/g, " "); }

function AddInstrumentCard({ onClose, onSaved, onOpenExisting }) {
  const [step, setStep] = useState(1);
  const [isin, setIsin] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [resolution, setResolution] = useState(null);
  const [chosen, setChosen] = useState(null);
  const [form, setForm] = useState({ name: "", issuer: "", asset_class: "ETF", base_currency: "EUR" });
  const check = checkIsin(isin);
  const candidate = resolution && chosen ? (resolution.candidates || []).find((c) => c.yahoo_symbol === chosen) || null : null;

  const lookup = async (e) => {
    if (e) e.preventDefault();
    if (check.state !== "valid" || busy) return;
    setBusy(true); setError(null);
    try {
      const r = await resolveIsin(check.isin);
      setResolution(r);
      setChosen(r.recommended || null);
      setForm({ name: "", issuer: "", asset_class: "ETF", base_currency: "EUR" });
      setStep(2);
    } catch (err) { setError(err); }
    finally { setBusy(false); }
  };
  const toConfirm = () => {
    if (!candidate) return;
    setForm((f) => ({ ...f, name: f.name || candidate.name || "" }));
    setError(null);
    setStep(3);
  };
  const save = async (e) => {
    e.preventDefault();
    if (!candidate || busy) return;
    if (!form.name.trim()) { setError(new Error("An instrument needs a name.")); return; }
    if (!/^[A-Z]{3}$/i.test(form.base_currency.trim())) { setError(new Error("Base currency is a three-letter ISO code, e.g. EUR or USD.")); return; }
    setBusy(true); setError(null);
    try {
      const saved = await saveInstrument({
        isin: check.isin, yahoo_symbol: candidate.yahoo_symbol, name: form.name.trim(), issuer: form.issuer.trim(),
        asset_class: form.asset_class, base_currency: form.base_currency.trim().toUpperCase(),
        quote_currency: candidate.currency || null, mic: candidate.mic || null, ticker: candidate.ticker || null,
      });
      toast.success(`${saved.name} is in the universe as ${candidate.yahoo_symbol}.`, { title: "Instrument saved" });
      onSaved(saved);
    } catch (err) { setError(err); }
    finally { setBusy(false); }
  };

  const existing = error && error.status === 409 ? check.isin : null;
  const errorBanner = error ? html`<${Banner} kind="error" title=${error.status === 409 ? "Already in the universe." : error.status === 400 ? "Refused." : "Something went wrong."}
      action=${existing ? html`<${Button} size="sm" variant="secondary" onClick=${() => onOpenExisting(existing)}>Open it<//>` : null}>${errorMessage(error)}<//>` : null;

  const stepOne = html`<form class="stack gap-3" onSubmit=${lookup}>
    <div class="ins-isin-row">
      <${Field} label="ISIN" id="ins-isin" required class="grow" hint=${null}>
        <${Input} id="ins-isin" class="ins-isin-input" value=${isin} placeholder="IE0002Y8CX98" maxlength="12" autocomplete="off" spellcheck="false"
          onInput=${(e) => { setIsin(e.target.value.toUpperCase()); setError(null); }} error=${check.state === "invalid"} aria-describedby="ins-isin-check" data-autofocus />
        <div id="ins-isin-check" class="ins-check" data-state=${check.state} aria-live="polite">
          ${check.state === "valid" ? html`<${Icon} name="checkCircle" size=${13} stroke=${2} />` : check.state === "invalid" ? html`<${Icon} name="alertCircle" size=${13} stroke=${2} />` : html`<${Icon} name="info" size=${13} />`}
          <span>${check.message}</span>
        </div>
      <//>
      <div class="ins-isin-actions">
        <${Button} type="submit" variant="primary" size="lg" icon="search" loading=${busy} disabled=${check.state !== "valid"}>Look up<//>
      </div>
    </div>
    ${busy ? html`<div class="row small muted" role="status"><span class="spinner" aria-hidden="true"></span><span>Resolving via OpenFIGI, then probing each listing…</span></div>` : null}
    ${errorBanner}
    <p class="small muted" style="max-width:720px">The ISIN is mapped to every listing it has, the non-European and secondary venues are filtered out and counted, and each remaining listing is probed for currency, venue and history. You then choose one. Nothing is written until you confirm — there is no text box that saves a symbol directly.</p>
  </form>`;

  const r = resolution || {};
  const filtered = Object.entries(r.filtered_by_class || {});
  const usFiltered = (r.filtered_by_class || {}).us || 0;
  const blockedAlerts = r.blocked && r.block_reason ? [{ code: "blocked", severity: "SERIOUS", title: "No listing can be chosen for you", detail: r.block_reason, isins: [], route: "" }] : [];
  const stepTwo = html`<div class="stack gap-4">
    <div class="stack gap-1">
      <div class="small" style="color:var(--color-text)">${r.summary}</div>
      ${filtered.length ? html`<div class="row wrap" style="gap:6px">
        <span class="xs faint">Filtered before probing:</span>
        ${filtered.map(([cls, n]) => html`<${Badge} key=${cls} kind=${cls === "us" ? "serious" : "neutral"} outline>${n} ${filterLabel(cls)}<//>`)}
      </div>` : null}
      ${usFiltered ? html`<p class="xs muted" style="max-width:760px">${usFiltered === 1 ? "One US listing was" : `${usFiltered} US listings were`} set aside without probing. A US venue is refused outright, not merely ranked down: a bare ticker such as WDEF, WEAT or GLUX returns a real, healthy security on a US exchange that is a different fund sharing the name — a clean 500-day series for the wrong instrument would outrank every genuine European listing on history alone.</p>` : null}
      ${(r.errors || []).length ? html`<ul class="ins-errors xs muted">${r.errors.map((e) => html`<li key=${e}><${Icon} name="alertTriangle" size=${12} stroke=${2} /><span>${e}</span></li>`)}</ul>` : null}
    </div>
    <${Notice} alerts=${blockedAlerts} summary="Saving is blocked — no listing passed" visible=${1} />
    ${(r.candidates || []).length ? html`<div class="stack gap-1">
      <div class="caps">Listings</div>
      <div class="ins-candidates" role="radiogroup" aria-label="Listings">
        ${r.candidates.map((c) => html`<${CandidateRow} key=${c.yahoo_symbol} c=${c} selected=${chosen === c.yahoo_symbol} recommended=${r.recommended === c.yahoo_symbol} onSelect=${() => setChosen(c.yahoo_symbol)} />`)}
      </div>
      <p class="xs faint">PASS is a primary European venue, current, with at least the lookback of history. THIN and STALE can be chosen deliberately but are never preselected; FAILED cannot be chosen.</p>
    </div>` : html`<${EmptyState} compact icon="search" title="No usable listing" body="Nothing on an allowlisted European venue answered with a price series for this ISIN." />`}
    ${(r.refused || []).length ? html`<div class="stack gap-1">
      <div class="caps" style="color:var(--color-serious)">Refused</div>
      <p class="xs muted" style="max-width:760px">These listings resolved and returned prices, and are refused regardless. A US venue is a hard gate because a ticker collision returns a real security with a clean series for the wrong fund — WDEF, WEAT and GLUX all do — and a penalty is a number that can be outweighed; a gate cannot.</p>
      <div class="ins-candidates">${r.refused.map((c) => html`<${RefusedRow} key=${c.yahoo_symbol} c=${c} />`)}</div>
    </div>` : null}
    ${candidate && candidate.verdict !== "PASS" ? html`<${Banner} kind="warn" title=${`Chosen deliberately: ${candidate.verdict}.`}>${(candidate.reasons || []).join(" ") || "This listing did not pass every check."}<//>` : null}
    <div class="row" style="justify-content:flex-end">
      <${Button} variant="ghost" icon="chevronLeft" onClick=${() => { setStep(1); setError(null); }}>Back<//>
      <${Button} variant="primary" iconRight="chevronRight" disabled=${!candidate} onClick=${toConfirm}>Continue with ${candidate ? candidate.yahoo_symbol : "a listing"}<//>
    </div>
  </div>`;

  const stepThree = candidate ? html`<form class="stack gap-4" onSubmit=${save}>
    <div class="ins-listing-summary">
      <${VerdictBadge} verdict=${candidate.verdict} />
      <span class="mono strong">${candidate.yahoo_symbol}</span>
      <span>${candidate.venue_label}${candidate.reported_exchange ? ` (${candidate.reported_exchange})` : ""}</span>
      <span class="num">quotes in <strong>${candidate.currency || "?"}</strong></span>
      <${ObsSpan} c=${candidate} />
      <span class="mono xs faint">ISIN ${check.isin}</span>
    </div>
    <div class="ins-confirm-grid">
      <${Field} label="Name" id="ins-name" required hint="Prefilled from the listing; use the fund's full name.">
        <${Input} id="ins-name" value=${form.name} onInput=${(e) => setForm({ ...form, name: e.target.value })} data-autofocus />
      <//>
      <${Field} label="Issuer" id="ins-issuer" hint="iShares, Amundi, WisdomTree … used for the issuer exposure view.">
        <${Input} id="ins-issuer" value=${form.issuer} onInput=${(e) => setForm({ ...form, issuer: e.target.value })} placeholder="Optional" />
      <//>
      <${Field} label="Asset class" id="ins-class" hint=${form.asset_class === "ETC" ? ETC_HINT : "Choose ETC for a commodity note — it is separated from ETF on purpose."}>
        <${Select} id="ins-class" value=${form.asset_class} options=${ASSET_CLASSES} onChange=${(v) => setForm({ ...form, asset_class: v })} />
      <//>
      <${Field} label="Base currency" id="ins-base" required hint=${BASE_HINT}>
        <${Input} id="ins-base" class="mono" value=${form.base_currency} maxlength="3" style="width:120px;text-transform:uppercase" onInput=${(e) => setForm({ ...form, base_currency: e.target.value.toUpperCase() })} />
      <//>
    </div>
    ${errorBanner}
    <div class="row" style="justify-content:flex-end">
      <${Button} variant="ghost" icon="chevronLeft" onClick=${() => { setStep(2); setError(null); }}>Back<//>
      <${Button} type="submit" variant="primary" icon="check" loading=${busy}>Save instrument<//>
    </div>
  </form>` : null;

  return html`<${Card} class="ins-add" title="Add an instrument" caption="Resolve, confirm, save — nothing is written until you have chosen a listing you can see."
      actions=${html`<${Button} variant="ghost" icon="x" iconOnly title="Close" onClick=${onClose} />`}>
    <div class="stack gap-4">
      <${Steps} steps=${["ISIN", "Choose a listing", "Confirm and save"]} current=${step} />
      ${step === 1 ? stepOne : step === 2 ? stepTwo : stepThree}
    </div>
  <//>`;
}

/* ---------------- edit drawer ---------------- */

function formFrom(inst, provider) {
  const ps = inst.provider_symbols || {};
  const keys = Array.from(new Set([provider, ...Object.keys(ps)].filter(Boolean)));
  const symbols = {};
  for (const k of keys) symbols[k] = ps[k] || "";
  return { isin: inst.isin, name: inst.name || "", issuer: inst.issuer || "", asset_class: inst.asset_class || "ETF", base_currency: inst.base_currency || "EUR", note: inst.note || "", keys, symbols };
}

function diffForm(form, inst) {
  const patch = {};
  if (form.name.trim() !== (inst.name || "")) patch.name = form.name.trim();
  if (form.issuer.trim() !== (inst.issuer || "")) patch.issuer = form.issuer.trim();
  if (form.asset_class !== inst.asset_class) patch.asset_class = form.asset_class;
  if (form.base_currency.trim().toUpperCase() !== (inst.base_currency || "")) patch.base_currency = form.base_currency.trim().toUpperCase();
  if (form.note !== (inst.note || "")) patch.note = form.note;
  const ps = {};
  let any = false;
  for (const k of form.keys) {
    const v = (form.symbols[k] || "").trim();
    if (v && v !== ((inst.provider_symbols || {})[k] || "")) { ps[k] = v; any = true; }
  }
  if (any) patch.provider_symbols = ps;
  return patch;
}

function DeleteModal({ inst, onClose, onDeleted, onDeactivate }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const n = inst.transaction_count || 0;
  const doDelete = async () => {
    if (busy) return;
    setBusy(true); setError(null);
    try {
      await deleteInstrument(inst.isin);
      toast.success(`${inst.name} removed from the universe.`, { title: "Deleted" });
      onDeleted();
    } catch (err) { setError(err); }
    finally { setBusy(false); }
  };
  return html`<${Modal} open title=${`Delete ${fmt.shortName(inst.name, 36)}?`} onClose=${onClose}
      footer=${html`
        <${Button} variant="secondary" onClick=${onClose}>Cancel<//>
        <${Button} variant="danger" icon="trash" loading=${busy} onClick=${doDelete} data-autofocus=${n === 0 ? true : undefined}>Delete<//>`}>
    <div class="stack gap-3">
      <dl class="definition-list">
        <dt>ISIN</dt><dd class="mono">${inst.isin}</dd>
        <dt>Transactions</dt><dd class=${n ? "strong" : ""}>${fmt.count(n, "transaction")}</dd>
      </dl>
      <p class="small">${n === 0
        ? "No transactions reference this instrument: it is a watchlist entry and deleting it is just tidying. Prices cached for it are kept."
        : `${fmt.count(n, "transaction")} reference this instrument. The server will refuse to delete it, because deleting would orphan those ledger rows and change historical portfolio values. Deactivate it instead — that hides it and keeps the history.`}</p>
      ${error ? html`<${Banner} kind="error" title=${error.status === 409 ? "Refused." : "Delete failed."}
        action=${error.status === 409 ? html`<${Button} size="sm" variant="secondary" icon="eyeOff" onClick=${onDeactivate}>Deactivate instead<//>` : null}>${errorMessage(error)}<//>` : null}
    </div>
  <//>`;
}

function EditDrawer({ inst, provider, onClose, onChanged, onDeleted }) {
  const [form, setForm] = useState(null);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [confirming, setConfirming] = useState(false);
  useEffect(() => {
    setForm(inst ? formFrom(inst, provider) : null);
    setError(null);
    setConfirming(false);
  }, [inst && inst.isin, provider]);

  const ready = !!(inst && form && form.isin === inst.isin);
  const patch = ready ? diffForm(form, inst) : {};
  const dirty = Object.keys(patch).length > 0;
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e && e.target ? e.target.value : e }));

  const apply = async (body, label) => {
    setBusy(label); setError(null);
    try {
      const updated = await patchInstrument(inst.isin, body);
      onChanged(updated);
      setForm(formFrom(updated, provider));
      loadSnapshot().catch(() => {});
      return updated;
    } catch (err) { setError(err); return null; }
    finally { setBusy(null); }
  };
  const save = async (e) => {
    e.preventDefault();
    if (!dirty || busy) return;
    if (patch.name === "") { setError(new Error("An instrument needs a name.")); return; }
    if (patch.base_currency != null && !/^[A-Z]{3}$/.test(patch.base_currency)) { setError(new Error("Base currency is a three-letter ISO code, e.g. EUR or USD.")); return; }
    const updated = await apply(patch, "save");
    if (updated) toast.success(`${updated.name} saved.`, { title: "Instrument updated" });
  };
  const toggleActive = async () => {
    if (busy) return;
    const next = !inst.active;
    const updated = await apply({ active: next }, "active");
    if (updated) toast.success(next ? `${updated.name} is active again.` : `${updated.name} is hidden from the watchlist and holdings; its history stays.`, { title: next ? "Reactivated" : "Deactivated" });
  };
  const deactivateFromDelete = async () => {
    setConfirming(false);
    const updated = await apply({ active: false }, "active");
    if (updated) toast.success(`${updated.name} deactivated instead of deleted.`, { title: "Deactivated" });
  };

  const sym = inst ? symbolEntry(inst, provider) : null;
  const footer = ready ? html`
    <div class="ins-actions-left">
      <${Button} variant="danger" icon="trash" onClick=${() => setConfirming(true)} disabled=${!!busy}>Delete…<//>
      <${Button} variant="secondary" icon=${inst.active ? "eyeOff" : "eye"} loading=${busy === "active"} onClick=${toggleActive}
        title=${inst.active ? "Hide from the watchlist and holdings; history is kept" : "Show again on the watchlist and in holdings"}>${inst.active ? "Deactivate" : "Reactivate"}<//>
    </div>
    <${Button} variant="ghost" onClick=${onClose}>Cancel<//>
    <${Button} type="submit" form="ins-edit-form" variant="primary" icon="check" loading=${busy === "save"} disabled=${!dirty}>Save changes<//>` : null;

  return html`<${Drawer} open=${!!inst} onClose=${onClose} title=${inst ? fmt.shortName(inst.name, 44) : ""} subtitle=${inst ? [sym && sym.symbol, inst.isin].filter(Boolean).join(" · ") : ""}
      badge=${inst ? (inst.active ? html`<${Badge} kind="good" dot>active<//>` : html`<${Badge} kind="neutral" icon="eyeOff">inactive<//>`) : null} footer=${footer}>
    ${ready ? html`<form id="ins-edit-form" class="stack gap-4" onSubmit=${save}>
      ${error ? html`<${Banner} kind="error" title=${error.status === 409 ? "Refused." : "Not saved."}>${errorMessage(error)}<//>` : null}
      <dl class="definition-list ins-facts">
        <dt>ISIN</dt><dd class="mono">${inst.isin}</dd>
        <dt>Venue</dt><dd>${fmt.text(inst.exchange)} <span class="faint small">· quotes in ${fmt.text(inst.quote_currency)}</span></dd>
        <dt>Ticker</dt><dd class="mono">${fmt.text(inst.primary_symbol)}</dd>
        <dt>Ledger</dt><dd>${inst.held ? html`<span class="row" style="gap:6px"><${Badge} kind="good">held<//><span>${fmt.count(inst.transaction_count, "transaction")}</span></span>` : html`<span class="row" style="gap:6px"><${Badge} kind="neutral" outline>watchlist<//><span class="muted">no transactions</span></span>`}</dd>
      </dl>
      <p class="xs muted">Fields you set by hand are marked <${Badge} kind="info" outline>manual<//> and re-resolution will not overwrite them.</p>
      <${Field} label=${html`Name <${Manual} inst=${inst} field="name" />`} id="ins-e-name" required>
        <${Input} id="ins-e-name" value=${form.name} onInput=${set("name")} data-autofocus />
      <//>
      <${Field} label=${html`Issuer <${Manual} inst=${inst} field="issuer" />`} id="ins-e-issuer">
        <${Input} id="ins-e-issuer" value=${form.issuer} onInput=${set("issuer")} placeholder="Optional" />
      <//>
      <div class="ins-confirm-grid">
        <${Field} label=${html`Asset class <${Manual} inst=${inst} field="asset_class" />`} id="ins-e-class" hint=${form.asset_class === "ETC" ? ETC_HINT : null}>
          <${Select} id="ins-e-class" value=${form.asset_class} options=${ASSET_CLASSES} onChange=${set("asset_class")} />
        <//>
        <${Field} label=${html`Base currency <${Manual} inst=${inst} field="base_currency" />`} id="ins-e-base" hint="Not inferable from the listing.">
          <${Input} id="ins-e-base" class="mono" value=${form.base_currency} maxlength="3" style="text-transform:uppercase" onInput=${(e) => setForm({ ...form, base_currency: e.target.value.toUpperCase() })} />
        <//>
      </div>
      <${Field} label="Note" id="ins-e-note" hint="Free text; deactivation appends a line here.">
        <${Textarea} id="ins-e-note" rows=${3} value=${form.note} onInput=${set("note")} />
      <//>
      <div class="stack gap-3">
        <div class="caps">Provider symbols</div>
        ${form.keys.map((k) => html`<${Field} key=${k} label=${html`${k}${k === provider ? html` <span class="faint" style="font-weight:400">· running provider</span>` : null} <${Manual} inst=${inst} field=${`provider_symbols.${k}`} />`} id=${`ins-e-sym-${k}`}>
          <${Input} id=${`ins-e-sym-${k}`} class="mono" value=${form.symbols[k]} placeholder=${k === provider ? "e.g. EUDF.DE" : "not set"} spellcheck="false"
            onInput=${(e) => setForm({ ...form, symbols: { ...form.symbols, [k]: e.target.value } })} />
        <//>`)}
        <p class="xs muted">A symbol typed here is a manual override: it is marked and re-resolution will not overwrite it. Use the provider's form with the venue suffix, e.g. <span class="mono">EUDF.DE</span>; a wrong symbol prices the holding against the wrong fund, so prefer re-adding through resolution when in doubt.</p>
      </div>
      <p class="xs muted">Deactivating hides the instrument from the watchlist and the holdings view; its transactions stay in the history and the ledger can still reference it. It is reversible. Deleting is only possible while nothing references it.</p>
    </form>` : null}
    ${ready && confirming ? html`<${DeleteModal} inst=${inst} onClose=${() => setConfirming(false)} onDeleted=${() => { setConfirming(false); onDeleted(inst); }} onDeactivate=${deactivateFromDelete} />` : null}
  <//>`;
}

/* ---------------- universe table ---------------- */

function universeColumns(provider) {
  return [
    { key: "name", label: "Instrument", primary: true, render: (r, v) => html`<div class="truncate" style="max-width:300px" title=${v}><span>${fmt.text(v)}</span><span class="cell-sub">${r.issuer || "issuer not set"}</span></div>` },
    { key: "isin", label: "ISIN", render: (r, v) => html`<span class="mono">${fmt.isin(v)}</span>` },
    { key: "symbol", label: "Symbol", sortValue: (r) => symbolEntry(r, provider).symbol, render: (r) => {
      const s = symbolEntry(r, provider);
      return html`<span class="row" style="gap:6px"><span class="mono">${fmt.text(s.symbol)}</span>${s.manual ? html`<${Badge} kind="info" outline title="Set by hand; re-resolution will not overwrite it">manual<//>` : null}</span>${s.provider && s.provider !== provider ? html`<span class="cell-sub">${s.provider}</span>` : null}`;
    } },
    { key: "exchange", label: "Venue", render: (r, v) => html`<span class="mono">${fmt.text(v)}</span>`, title: "Exchange MIC of the stored listing" },
    { key: "quote_currency", label: "Quote / base", render: (r, v) => html`<span class="num">${fmt.text(v)}</span><span class="faint num"> / ${fmt.text(r.base_currency)}</span>`, title: "Currency the listing quotes in / the fund's base currency" },
    { key: "asset_class", label: "Class", render: (r, v) => html`<${Badge} kind="neutral" outline title=${v === "ETC" ? ETC_HINT : undefined}>${v}<//>` },
    { key: "transaction_count", label: "Held", sortValue: (r) => r.transaction_count, render: (r, v) => (r.held
      ? html`<span class="row" style="gap:6px"><${Badge} kind="good">held<//><span class="muted num">${fmt.int(v)} ${v === 1 ? "txn" : "txns"}</span></span>`
      : html`<${Badge} kind="neutral" outline>watchlist<//>`) },
    { key: "active", label: "Active", render: (r, v) => (v ? html`<${Badge} kind="good" dot>active<//>` : html`<${Badge} kind="neutral" icon="eyeOff" title="Hidden from the watchlist and holdings; history kept">inactive<//>`) },
  ];
}

function matches(inst, q) {
  if (!q) return true;
  const hay = `${inst.name} ${inst.isin} ${inst.issuer} ${inst.primary_symbol} ${Object.values(inst.provider_symbols || {}).join(" ")} ${inst.exchange} ${inst.asset_class}`.toLowerCase();
  return q.split(/\s+/).every((w) => hay.includes(w));
}

/* ---------------- page ---------------- */

export default function InstrumentsPage({ route, snapshot }) {
  const provider = (snapshot && snapshot.meta && snapshot.meta.provider) || "yfinance";
  const generatedAt = snapshot && snapshot.meta ? snapshot.meta.generated_at : null;
  const [instruments, setInstruments] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState(null);
  const params = (route && route.params) || {};

  const reload = useCallback(async () => {
    setLoading(true);
    try { setInstruments(await listInstruments()); setLoadError(null); }
    catch (err) { setLoadError(err); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { reload(); }, [reload, generatedAt]);
  useEffect(() => { if (params.new === "1") setAdding(true); }, [params.new]);
  useEffect(() => { if (params.isin) setEditing(params.isin); }, [params.isin]);

  const list = instruments || [];
  const filtered = useMemo(() => list.filter((i) => matches(i, query.trim().toLowerCase())), [list, query]);
  const columns = useMemo(() => universeColumns(provider), [provider]);
  const current = editing ? list.find((i) => i.isin === editing) || null : null;
  const held = list.filter((i) => i.held).length;
  const inactive = list.filter((i) => !i.active).length;

  const closeAdd = () => { setAdding(false); if (params.new) navigate("/instruments", { replace: true }); };
  const closeEdit = () => { setEditing(null); if (params.isin) navigate("/instruments", { replace: true }); };
  const afterWrite = async () => { await reload(); loadSnapshot().catch(() => {}); };
  const onSaved = async (inst) => { closeAdd(); await afterWrite(); setEditing(inst.isin); };
  const onChanged = (inst) => setInstruments((cur) => (cur || []).map((i) => (i.isin === inst.isin ? inst : i)));
  const onDeleted = async () => { closeEdit(); await afterWrite(); };

  const subtitle = instruments
    ? `${fmt.count(list.length, "instrument")} · ${held} held · ${list.length - held} on the watchlist${inactive ? ` · ${inactive} inactive` : ""} · symbols for ${provider}`
    : "The universe: ISIN resolution, venues, symbols and notes.";

  const actions = html`<${Button} variant="primary" icon="plus" onClick=${() => setAdding(true)} disabled=${adding}>Add instrument<//>`;

  return html`<${Page} title="Instruments" subtitle=${subtitle} actions=${actions} wide>
    <div class="stack gap-4">
      ${loadError ? html`<${Banner} kind="error" title="Could not load the universe." action=${html`<${Button} size="sm" variant="secondary" loading=${loading} onClick=${reload}>Retry<//>`}>${errorMessage(loadError)}<//>` : null}
      ${adding ? html`<${AddInstrumentCard} onClose=${closeAdd} onSaved=${onSaved} onOpenExisting=${(isin) => { closeAdd(); setEditing(isin); }} />` : null}
      <${Card} title="Universe" caption="Every instrument the ledger may reference. Held ones are positions; the rest are priced as a watchlist. Click a row to edit." flush>
        ${instruments === null
          ? html`<div class="faint small" style="padding:24px;text-align:center">${loadError ? "Nothing loaded." : "Loading the universe…"}</div>`
          : list.length === 0
            ? html`<${EmptyState} icon="instruments" title="The universe is empty"
                body="Add an instrument by ISIN: it is resolved to its listings and you confirm one before anything is saved. The ledger can only reference instruments that are here, so this is the first step before recording a transaction."
                action=${html`<${Button} variant="primary" icon="plus" onClick=${() => setAdding(true)}>Add an instrument<//>`} />`
            : html`<${DataTable} columns=${columns} rows=${filtered} rowKey="isin" sort=${{ key: "name", dir: "asc" }} onRowClick=${(row) => setEditing(row.isin)} selectedKey=${editing}
                rowClass=${(r) => (r.active ? "" : "ins-inactive")} loading=${loading} caption="Universe"
                empty=${html`<span>No instrument matches “${query}”.</span>`}
                toolbar=${html`<${Input} class="ins-search" value=${query} placeholder="Search name, ISIN, symbol, issuer…" onInput=${(e) => setQuery(e.target.value)} aria-label="Search the universe" prefix=${html`<${Icon} name="search" size=${14} />`} />`}
                toolbarRight=${html`<span class="faint small num">${filtered.length === list.length ? fmt.count(list.length, "instrument") : `${filtered.length} of ${list.length}`}</span>`} />`}
      <//>
    </div>
    <${EditDrawer} inst=${current} provider=${provider} onClose=${closeEdit} onChanged=${onChanged} onDeleted=${onDeleted} />
  <//>`;
}
