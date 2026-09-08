/* Transactions — the append-only ledger. Props: { route, snapshot }.

   Nothing on this page edits a row. A correction is a VOID that names the
   original and leaves it on disk, and every write goes through the API,
   which replays the ledger before accepting a row (a SELL of more units than
   are held is refused before it is written). The page then reloads both
   the ledger and the snapshot so every other page moves with it.

   Layout: the shape of the ledger (counts, span, instruments) first; one
   filter row above the table, scoping the same slice; the table; and three
   dialogs — record, void, import — that are the only ways in.
   The one piece of arithmetic here is the live preview in the record form
   (gross = quantity × price, net signed by type), shown beside its terms so
   the user sees what the server will compute; the server remains the truth. */

import { html, useEffect, useMemo, useRef, useState } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { DataTable } from "/static/components/DataTable.js";
import { Badge, TxTypeBadge } from "/static/components/Badge.js";
import { Banner } from "/static/components/Banner.js";
import { Button } from "/static/components/Button.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Field, Input, Select, Textarea, Checkbox } from "/static/components/Field.js";
import { Modal } from "/static/components/Modal.js";
import { Icon } from "/static/components/Icons.js";
import { toast } from "/static/components/Toast.js";
import { navigate } from "/static/lib/router.js";
import { listTransactions, listInstruments, addTransaction, voidTransaction, previewImport, runImport, exportTransactionsCsv, exportWorkbook, loadSnapshot, errorMessage } from "/static/lib/api.js";
import * as fmt from "/static/lib/format.js";

/* ---------------- page-local helpers ---------------- */

const TYPES = ["BUY", "SELL", "DIVIDEND", "FEE"];
const SUBTITLE = "Append-only. Corrections are voids that name the original row; nothing is ever edited in place, so \"why did last month's value change\" always has an answer.";

const todayIso = () => new Date().toISOString().slice(0, 10);

function parseNum(s) {
  if (s == null || s === "") return null;
  const n = Number(String(s).replace(",", "."));
  return Number.isFinite(n) ? n : null;
}

/** Display arithmetic for the record form; the server recomputes and is the truth. */
function preview(form) {
  const q = parseNum(form.quantity) || 0, p = parseNum(form.price_per_unit) || 0, f = parseNum(form.fees) || 0;
  const gross = q * p;
  let net;
  switch (form.type) {
    case "BUY": net = -(gross + f); break;
    case "SELL": net = gross - f; break;
    case "DIVIDEND": net = (gross || p) - f; break;
    default: net = -(gross + f);
  }
  return { gross, net };
}

function matches(t, q) {
  if (!q) return true;
  const hay = `${t.name || ""} ${t.isin || ""} ${t.note || ""} ${t.id || ""}`.toLowerCase();
  return q.split(/\s+/).filter(Boolean).every((part) => hay.includes(part));
}

/* ---------------- stats ---------------- */

function Stats({ live, voided, rows }) {
  const dates = rows.filter((r) => !r.voided).map((r) => r.date).sort();
  const instruments = new Set(rows.filter((r) => !r.voided).map((r) => r.isin));
  const byType = TYPES.map((t) => [t, rows.filter((r) => !r.voided && r.type === t).length]).filter(([, n]) => n);
  return html`<div class="tx-stats">
    <div class="card tx-stat"><span class="label">Live entries</span><span class="value">${fmt.int(live)}</span><span class="sub">${byType.map(([t, n]) => `${n} ${t.toLowerCase()}`).join(" · ") || "none yet"}</span></div>
    <div class="card tx-stat"><span class="label">Voided</span><span class="value">${fmt.int(voided)}</span><span class="sub">${voided ? "kept on disk for the audit trail" : "no corrections so far"}</span></div>
    <div class="card tx-stat"><span class="label">Span</span><span class="value">${dates.length ? fmt.dateRange(dates[0], dates[dates.length - 1]) : fmt.DASH}</span><span class="sub">${dates.length ? "first to last live entry" : "no entries"}</span></div>
    <div class="card tx-stat"><span class="label">Instruments touched</span><span class="value">${fmt.int(instruments.size)}</span><span class="sub">across the live ledger</span></div>
  </div>`;
}

/* ---------------- ledger table ---------------- */

const strike = (row, content) => (row.voided ? html`<span class="tx-strike" title=${row.void_reason ? `Voided: ${row.void_reason}` : "Voided"}>${content}</span>` : content);

function columns(base, onVoid) {
  return [
    { key: "date", label: "Date", width: 104, render: (r, v) => strike(r, fmt.date(v)) },
    { key: "name", label: "Instrument", primary: true, render: (r, v) => html`<div class="truncate" style="max-width:240px" title=${v}>${strike(r, r.short_name || v)}<span class="cell-sub">${r.isin}</span></div>` },
    { key: "type", label: "Type", render: (r, v) => html`<span class="row" style="gap:4px"><${TxTypeBadge} type=${v} />${r.voided ? html`<${Badge} kind="neutral" outline title=${r.void_reason || "Voided"}>void<//>` : null}</span>` },
    { key: "quantity", label: "Qty", numeric: true, render: (r, v) => strike(r, fmt.qty(v)) },
    { key: "price_per_unit", label: "Price", numeric: true, render: (r, v) => strike(r, fmt.money(v, r.currency)) },
    { key: "fees", label: "Fees", numeric: true, render: (r, v) => strike(r, fmt.money(v, r.currency)) },
    { key: "gross", label: "Gross", numeric: true, title: "Quantity × price, before fees", render: (r, v) => strike(r, fmt.money(v, r.currency)) },
    { key: "net", label: "Net", numeric: true, title: "Signed cash effect in the transaction currency: negative is money out", render: (r, v) => strike(r, html`<span class=${fmt.polarityClass(v)}>${fmt.money(v, r.currency, { signed: true })}</span>`) },
    { key: "note", label: "Note", sortable: false, render: (r, v) => (v ? html`<span class="tx-note" title=${v}>${v}</span>` : html`<span class="faint">${fmt.DASH}</span>`) },
    { key: "id", label: "Id", sortable: false, render: (r, v) => html`<span class="tx-id" title=${v}>${String(v).replace(/^txn_/, "")}</span>` },
    { key: "_actions", label: "", sortable: false, width: 72, render: (r) => (r.voided ? "" : html`<span class="tx-row-actions"><${Button} size="sm" variant="ghost" onClick=${(e) => { e.stopPropagation(); onVoid(r); }} title="Append a void naming this row">Void…<//></span>`) },
  ];
}

/* ---------------- record dialog ---------------- */

function RecordDialog({ instruments, initialIsin, onClose, onDone }) {
  const options = useMemo(() => [...(instruments || [])].sort((a, b) => a.name.localeCompare(b.name)).map((i) => ({ value: i.isin, label: `${fmt.shortName(i.name, 44)} · ${i.isin}${i.active ? "" : " (inactive)"}` })), [instruments]);
  const first = initialIsin && options.some((o) => o.value === initialIsin) ? initialIsin : (options[0] ? options[0].value : "");
  const byIsin = useMemo(() => new Map((instruments || []).map((i) => [i.isin, i])), [instruments]);
  const [form, setForm] = useState({ isin: first, type: "BUY", date: todayIso(), quantity: "", price_per_unit: "", currency: (byIsin.get(first) && byIsin.get(first).quote_currency) || "EUR", fees: "0", note: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e && e.target ? e.target.value : e }));
  const pickIsin = (v) => setForm((f) => ({ ...f, isin: v, currency: (byIsin.get(v) && byIsin.get(v).quote_currency) || f.currency }));
  // The universe loads after the dialog can already be open (#/transactions?new=1
  // on a cold start), so the selection is settled once the options exist.
  useEffect(() => {
    if (!options.length || options.some((o) => o.value === form.isin)) return;
    pickIsin(first);
  }, [options, first]);
  const p = preview(form);
  const isFee = form.type === "FEE", isDiv = form.type === "DIVIDEND";

  const submit = async (e) => {
    if (e) e.preventDefault();
    if (busy) return;
    setBusy(true); setError(null);
    try {
      const saved = await addTransaction({ date: form.date, isin: form.isin, type: form.type, quantity: form.quantity || "0", price_per_unit: form.price_per_unit || "0", currency: form.currency || "EUR", fees: form.fees || "0", note: form.note || "" });
      toast.success(`${saved.type} of ${fmt.qty(saved.quantity)} ${fmt.shortName(saved.name, 28)} appended`, { title: "Ledger updated" });
      onDone(saved);
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  };

  if (!options.length) {
    return html`<${Modal} title="Record a transaction" onClose=${onClose} footer=${html`<${Button} variant="primary" onClick=${() => { onClose(); navigate("/instruments?new=1"); }}>Add an instrument<//>`}>
      <${EmptyState} compact icon="instruments" title="No instruments yet" body="Transactions are keyed by ISIN. Add the instrument first, so the ledger is never keyed on an unresolved identifier." />
    <//>`;
  }

  return html`<${Modal} title="Record a transaction" onClose=${onClose} size="lg"
    footer=${html`<${Button} variant="ghost" onClick=${onClose}>Cancel<//><${Button} variant="primary" loading=${busy} onClick=${submit} data-autofocus>Append to ledger<//>`}>
    <form class="tx-form" onSubmit=${submit}>
      <${Field} label="Instrument" required class="span-2"><${Select} value=${form.isin} options=${options} onChange=${pickIsin} /><//>
      <${Field} label="Type" required><${Select} value=${form.type} options=${TYPES} onChange=${set("type")} /><//>
      <${Field} label="Date" required hint="The trade date, not the settlement date."><${Input} type="date" value=${form.date} max=${todayIso()} onInput=${set("date")} /><//>
      <${Field} label=${isFee ? "Quantity (optional)" : "Quantity"} required=${!isFee} hint=${isDiv ? "Units held on the record date; leave 0 to enter the total amount as the price." : "A sell is a positive quantity of type SELL, never a negative buy."}>
        <${Input} numeric value=${form.quantity} onInput=${set("quantity")} placeholder="0" /><//>
      <${Field} label=${isDiv ? "Per unit (or total when quantity is 0)" : isFee ? "Charge (optional)" : "Price per unit"} required=${!isFee}>
        <${Input} numeric value=${form.price_per_unit} onInput=${set("price_per_unit")} placeholder="0.00" /><//>
      <${Field} label="Currency" required hint="The currency you actually paid in. Pence lines are entered as GBp and converted to GBP."><${Input} value=${form.currency} onInput=${set("currency")} placeholder="EUR" maxlength="3" style="text-transform:uppercase" /><//>
      <${Field} label="Fees" hint="Capitalised into the cost basis on a buy; deducted from proceeds on a sell."><${Input} numeric value=${form.fees} onInput=${set("fees")} placeholder="0.00" /><//>
      <${Field} label="Note" class="span-2"><${Input} value=${form.note} onInput=${set("note")} placeholder="optional" /><//>
      <div class="tx-preview">
        <span>Gross <b class="num">${fmt.money(p.gross, form.currency || "EUR")}</b></span>
        <span>Fees <b class="num">${fmt.money(parseNum(form.fees) || 0, form.currency || "EUR")}</b></span>
        <span class="lead">Net <b class=${["num", fmt.polarityClass(p.net)].join(" ")}>${fmt.money(p.net, form.currency || "EUR", { signed: true })}</b></span>
        <span class="xs faint">display arithmetic — the server recomputes and replays the ledger before accepting the row</span>
      </div>
      ${error ? html`<div class="tx-error"><${Banner} kind="error">${error}<//></div>` : null}
    </form>
  <//>`;
}

/* ---------------- void dialog ---------------- */

function VoidDialog({ row, onClose, onDone }) {
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const submit = async () => {
    if (busy) return;
    setBusy(true); setError(null);
    try {
      const out = await voidTransaction(row.id, reason.trim());
      toast.success("The original row stays on disk for the audit trail.", { title: "Voided" });
      onDone(out);
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  };
  return html`<${Modal} title="Void this entry" onClose=${onClose}
    footer=${html`<${Button} variant="ghost" onClick=${onClose}>Keep it<//><${Button} variant="danger" loading=${busy} onClick=${submit} data-autofocus>Append the void<//>`}>
    <div class="stack gap-3">
      <p class="small muted">A void is appended as a correction that names this row. The row is not deleted and not edited: it stays in the ledger, marked void, and every position is re-derived without it. To replace it, record the corrected entry afterwards.</p>
      <div class="card" style="padding:10px 12px">
        <div class="row" style="gap:8px"><${TxTypeBadge} type=${row.type} /><span class="strong">${fmt.shortName(row.name, 40)}</span><span class="faint small">${fmt.date(row.date)}</span></div>
        <div class="small muted" style="margin-top:4px">${fmt.qty(row.quantity)} × ${fmt.money(row.price_per_unit, row.currency)} · fees ${fmt.money(row.fees, row.currency)} · net <span class=${fmt.polarityClass(row.net)}>${fmt.money(row.net, row.currency, { signed: true })}</span> · <span class="tx-id">${row.id}</span></div>
      </div>
      <${Field} label="Reason" hint="Written into the amendment log next to the void."><${Textarea} rows=${2} value=${reason} onInput=${(e) => setReason(e.target.value)} placeholder="quantity mistyped" /><//>
      ${error ? html`<${Banner} kind="error">${error}<//>` : null}
    </div>
  <//>`;
}

/* ---------------- import wizard ---------------- */

const MAP_FIELDS = [["date", "Date"], ["isin", "ISIN"], ["type", "Type"], ["quantity", "Quantity"], ["price", "Price"], ["currency", "Currency"], ["fees", "Fees"], ["note", "Note"]];

function Steps({ current }) {
  const steps = ["Paste or drop", "Preview", "Import"];
  return html`<div class="tx-steps">
    ${steps.map((s, i) => html`<span key=${s} class="tx-step" data-state=${i + 1 < current ? "done" : i + 1 === current ? "active" : "todo"}><span class="tx-step-n">${i + 1 < current ? "✓" : i + 1}</span>${s}</span>${i < steps.length - 1 ? html`<span class="tx-step-sep"></span>` : null}`)}
  </div>`;
}

function ImportDialog({ onClose, onDone }) {
  const [step, setStep] = useState(1);
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState("");
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [mapping, setMapping] = useState(null);
  const fileRef = useRef(null);

  const readFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { setText(String(reader.result || "")); setFileName(file.name); };
    reader.readAsText(file);
  };

  const doPreview = async (m) => {
    if (!text.trim()) { setError("Paste some CSV text or choose a file first."); return; }
    setBusy(true); setError(null);
    try {
      const p = await previewImport(text, m || undefined);
      setPreview(p);
      setMapping(p.mapping);
      setStep(2);
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  };

  const remap = (key, value) => {
    const m = { ...mapping, [key]: value === "" ? null : value };
    setMapping(m);
    doPreview(m);
  };

  const doImport = async () => {
    setBusy(true); setError(null);
    try {
      const out = await runImport(text, mapping || undefined);
      toast.success(`${fmt.int(out.imported)} imported${out.skipped ? `, ${fmt.int(out.skipped)} skipped` : ""}`, { title: "Import complete" });
      setStep(3);
      onDone(out);
    } catch (err) { setError(errorMessage(err)); }
    finally { setBusy(false); }
  };

  const headers = preview ? preview.headers : [];
  const headerOptions = [{ value: "", label: "— not mapped —" }, ...headers.map((h) => ({ value: h, label: h }))];
  const rows = preview ? preview.rows.map((r) => ({ id: r.line, line: r.line, ok: r.ok, error: r.error, ...(r.transaction || {}), rawDate: r.raw.date, raw: r.raw })) : [];
  const previewColumns = [
    { key: "line", label: "Line", numeric: true, width: 56 },
    { key: "ok", label: "Status", sortable: true, render: (r, v) => (v ? html`<${Badge} kind="good" icon="check">ok<//>` : html`<span class="row" style="gap:6px;align-items:flex-start"><${Badge} kind="bad" icon="x">error<//><span class="tx-row-err">${r.error}</span></span>`) },
    { key: "date", label: "Date", render: (r, v) => (v ? fmt.date(v) : html`<span class="faint">${r.raw.date || fmt.DASH}</span>`) },
    { key: "isin", label: "ISIN", render: (r, v) => v || html`<span class="faint">${r.raw.isin || fmt.DASH}</span>` },
    { key: "type", label: "Type", render: (r, v) => (v ? html`<${TxTypeBadge} type=${v} />` : html`<span class="faint">${r.raw.type || fmt.DASH}</span>`) },
    { key: "quantity", label: "Qty", numeric: true, format: "qty" },
    { key: "price_per_unit", label: "Price", numeric: true, render: (r, v) => (v == null ? fmt.DASH : fmt.money(v, r.currency)) },
    { key: "fees", label: "Fees", numeric: true, render: (r, v) => (v == null ? fmt.DASH : fmt.money(v, r.currency)) },
  ];

  const footer = step === 1
    ? html`<${Button} variant="ghost" onClick=${onClose}>Cancel<//><${Button} variant="primary" loading=${busy} disabled=${!text.trim()} onClick=${() => doPreview()} data-autofocus>Preview<//>`
    : step === 2
      ? html`<${Button} variant="ghost" onClick=${() => setStep(1)}>Back<//><${Button} variant="primary" loading=${busy} disabled=${!preview || !preview.valid} onClick=${doImport} data-autofocus>Import ${preview ? fmt.int(preview.valid) : ""} ${preview && preview.valid === 1 ? "row" : "rows"}<//>`
      : html`<${Button} variant="primary" onClick=${onClose} data-autofocus>Done<//>`;

  return html`<${Modal} title="Import transactions from CSV" onClose=${onClose} size="lg" footer=${footer}>
    <${Steps} current=${step} />
    ${step === 1 ? html`<div class="stack gap-3">
      <p class="small muted">Comma or semicolon separated, with a header row. Dates as ISO, dd.mm.yyyy or dd/mm/yyyy; decimal commas are fine; BUY / Kauf / Achat and SELL / Verkauf / Vente are recognised. The columns are detected and shown for review before anything is written, and rows already in the ledger are flagged as duplicates.</p>
      <${Textarea} rows=${8} value=${text} onInput=${(e) => { setText(e.target.value); setFileName(""); }} placeholder=${"date;isin;type;quantity;price;currency;fees\n01.09.2026;LU1681048630;Kauf;3;120,50;EUR;1,00"} spellcheck="false" style="font-family:var(--font-mono);font-size:var(--fs-xs)" />
      <label class="tx-drop" data-over=${over ? "true" : "false"}
        onDragOver=${(e) => { e.preventDefault(); setOver(true); }} onDragLeave=${() => setOver(false)}
        onDrop=${(e) => { e.preventDefault(); setOver(false); readFile(e.dataTransfer.files && e.dataTransfer.files[0]); }}>
        <input ref=${fileRef} type="file" accept=".csv,.txt,text/csv" onChange=${(e) => readFile(e.target.files && e.target.files[0])} />
        <${Icon} name="upload" size=${16} />
        <span>${fileName ? `Loaded ${fileName} — ${fmt.int(text.split(/\r?\n/).filter(Boolean).length - 1)} rows` : "Drop a .csv here, or click to choose one"}</span>
      </label>
      ${error ? html`<${Banner} kind="error">${error}<//>` : null}
    </div>` : null}
    ${step === 2 && preview ? html`<div class="stack gap-2">
      <div class="tx-mapping">
        ${MAP_FIELDS.map(([k, label]) => html`<${Field} key=${k} label=${label}><${Select} compact value=${mapping[k] || ""} options=${headerOptions} onChange=${(v) => remap(k, v)} /><//>`)}
        <${Field} label="Decimal comma"><${Checkbox} checked=${!!mapping.decimal_comma} onChange=${(v) => remap("decimal_comma", v)} label=${mapping.decimal_comma ? "1.234,56" : "1,234.56"} /><//>
        <${Field} label="Date format" hint="optional strftime pattern"><${Input} value=${mapping.date_format || ""} placeholder="auto" onChange=${(e) => remap("date_format", e.target.value)} /><//>
      </div>
      <div class="tx-import-summary">
        <span><b>${fmt.int(preview.valid)}</b> ready</span>
        <span><b class=${preview.invalid ? "neg" : ""}>${fmt.int(preview.invalid)}</b> with problems (skipped)</span>
        ${preview.unknown_isins.length ? html`<span><b class="neg">${fmt.int(preview.unknown_isins.length)}</b> unknown ${preview.unknown_isins.length === 1 ? "ISIN" : "ISINs"}: ${preview.unknown_isins.join(", ")} — <a href="#/instruments?new=1" onClick=${(e) => { e.preventDefault(); onClose(); navigate("/instruments?new=1"); }}>add them first</a></span>` : null}
        ${busy ? html`<span class="faint">re-previewing…</span>` : null}
      </div>
      <${DataTable} class="tx-preview-table" columns=${previewColumns} rows=${rows} rowKey="id" maxHeight=${320} density="compact" loading=${false} rowClass=${(r) => (r.ok ? "" : "tx-voided")} empty="No rows parsed" />
      ${error ? html`<${Banner} kind="error">${error}<//>` : null}
    </div>` : null}
    ${step === 3 ? html`<${EmptyState} compact icon="checkCircle" title="Imported" body="The ledger and every page that reads it have been reloaded." />` : null}
  <//>`;
}

/* ---------------- page ---------------- */

export default function TransactionsPage({ route, snapshot }) {
  const base = (snapshot && snapshot.meta && snapshot.meta.base_currency) || "EUR";
  const [rows, setRows] = useState(null);
  const [instruments, setInstruments] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);
  const [busy, setBusy] = useState(null);
  const [dialog, setDialog] = useState(route && route.params && route.params.new ? { kind: "record", isin: route.params.isin } : null);
  const [filters, setFilters] = useState({ isin: "", types: new Set(TYPES), from: "", to: "", q: "", voided: false });
  const selectedId = route && route.params ? route.params.id : null;

  const reload = async () => {
    setLoading(true); setLoadError(null);
    try {
      const [tx, inst] = await Promise.all([listTransactions({ includeVoided: true }), listInstruments()]);
      setRows(tx); setInstruments(inst);
    } catch (err) { setLoadError(err); }
    finally { setLoading(false); }
  };
  useEffect(() => { reload(); }, []);
  useEffect(() => { if (route && route.params && route.params.new) setDialog({ kind: "record", isin: route.params.isin }); }, [route && route.params && route.params.new]);

  const afterWrite = async () => {
    setDialog(null);
    if (route && route.params && route.params.new) navigate("/transactions", { replace: true });
    await reload();
    loadSnapshot().catch(() => {});
  };

  const all = rows || [];
  const live = all.filter((r) => !r.voided).length;
  const voidedCount = all.length - live;
  const q = filters.q.trim().toLowerCase();
  const filtered = useMemo(() => all.filter((r) =>
    (filters.voided || !r.voided) && filters.types.has(r.type) && (!filters.isin || r.isin === filters.isin)
    && (!filters.from || r.date >= filters.from) && (!filters.to || r.date <= filters.to) && matches(r, q)), [all, filters, q]);
  const instrumentOptions = useMemo(() => [{ value: "", label: "All instruments" }, ...[...new Map(all.map((r) => [r.isin, r.name])).entries()].sort((a, b) => a[1].localeCompare(b[1])).map(([isin, name]) => ({ value: isin, label: `${fmt.shortName(name, 40)} · ${isin}` }))], [all]);
  const cols = useMemo(() => columns(base, (r) => setDialog({ kind: "void", row: r })), [base]);
  const toggleType = (t) => setFilters((f) => { const s = new Set(f.types); if (s.has(t) && s.size > 1) s.delete(t); else s.add(t); return { ...f, types: s }; });

  const doExport = async (kind) => {
    if (busy) return;
    setBusy(kind);
    try { await (kind === "csv" ? exportTransactionsCsv() : exportWorkbook()); }
    catch (err) { toast.error(errorMessage(err), { title: "Export failed" }); }
    finally { setBusy(null); }
  };

  const actions = html`
    <${Button} variant="ghost" icon="download" loading=${busy === "csv"} onClick=${() => doExport("csv")} title="Download transactions.csv">Export CSV<//>
    <${Button} variant="ghost" icon="download" loading=${busy === "xlsx"} onClick=${() => doExport("xlsx")} title="Download the full workbook">Export XLSX<//>
    <${Button} variant="secondary" icon="upload" onClick=${() => setDialog({ kind: "import" })}>Import CSV<//>
    <${Button} variant="primary" icon="plus" onClick=${() => setDialog({ kind: "record" })}>Record transaction<//>`;

  const filterBar = html`<div class="tx-filters" role="region" aria-label="Filters">
    <${Select} compact value=${filters.isin} options=${instrumentOptions} onChange=${(v) => setFilters((f) => ({ ...f, isin: v }))} aria-label="Instrument" />
    <div class="tx-types" role="group" aria-label="Types">
      ${TYPES.map((t) => html`<button key=${t} type="button" class="tx-type" aria-pressed=${filters.types.has(t) ? "true" : "false"} onClick=${() => toggleType(t)}>${t}</button>`)}
    </div>
    <div class="tx-dates">
      <${Input} type="date" value=${filters.from} onInput=${(e) => setFilters((f) => ({ ...f, from: e.target.value }))} aria-label="From date" />
      <span>to</span>
      <${Input} type="date" value=${filters.to} onInput=${(e) => setFilters((f) => ({ ...f, to: e.target.value }))} aria-label="To date" />
    </div>
    <div class="tx-search"><${Icon} name="search" size=${13} /><${Input} value=${filters.q} onInput=${(e) => setFilters((f) => ({ ...f, q: e.target.value }))} placeholder="Note, ISIN, id" aria-label="Search" spellcheck="false" /></div>
    <${Checkbox} checked=${filters.voided} onChange=${(v) => setFilters((f) => ({ ...f, voided: v }))} label="Show voided" />
    <span class="spacer"></span>
    <span class="tx-count">${filtered.length === all.length ? fmt.count(all.length, "entry", "entries") : `${filtered.length} of ${all.length}`}</span>
  </div>`;

  return html`<${Page} title="Transactions" subtitle=${SUBTITLE} actions=${actions} wide>
    <div class="stack gap-4">
      ${loadError ? html`<${Banner} kind="error" title="Could not load the ledger." action=${html`<${Button} size="sm" variant="secondary" onClick=${reload}>Retry<//>`}>${errorMessage(loadError)}<//>` : null}
      <${Stats} live=${live} voided=${voidedCount} rows=${all} />
      <${Card} flush class="tx-tablebox">
        ${filterBar}
        ${rows && !all.length
          ? html`<${EmptyState} icon="transactions" title="The ledger is empty" body="Positions are derived from it, so nothing appears in Holdings until a transaction is recorded. Record one by hand, or import the CSV your broker exports."
              action=${html`<span class="row" style="gap:8px"><${Button} variant="primary" icon="plus" onClick=${() => setDialog({ kind: "record" })}>Record a transaction<//><${Button} variant="secondary" icon="upload" onClick=${() => setDialog({ kind: "import" })}>Import CSV<//></span>`} />`
          : html`<${DataTable} columns=${cols} rows=${filtered} rowKey="id" sort=${{ key: "date", dir: "desc" }} loading=${loading}
              selectedKey=${selectedId} rowClass=${(r) => (r.voided ? "tx-voided" : "")} caption="Ledger"
              empty=${all.length ? "No entry matches the filters" : "No entries"} />`}
      <//>
    </div>
    ${dialog && dialog.kind === "record" ? html`<${RecordDialog} instruments=${instruments || []} initialIsin=${dialog.isin} onClose=${() => { setDialog(null); if (route.params.new) navigate("/transactions", { replace: true }); }} onDone=${afterWrite} />` : null}
    ${dialog && dialog.kind === "void" ? html`<${VoidDialog} row=${dialog.row} onClose=${() => setDialog(null)} onDone=${afterWrite} />` : null}
    ${dialog && dialog.kind === "import" ? html`<${ImportDialog} onClose=${() => setDialog(null)} onDone=${() => { reload(); loadSnapshot().catch(() => {}); }} />` : null}
  <//>`;
}
