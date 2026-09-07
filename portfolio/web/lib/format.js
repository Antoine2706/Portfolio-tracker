/* Formatting. Every function is null-safe: null/undefined/NaN renders "—".
   Fractions are fractions (0.12 → "12.0%"); the client formats, never computes. */

export const DASH = "—";
export const NBSP = " ";

const isNil = (v) => v == null || (typeof v === "number" && !Number.isFinite(v));

const currencyFormatters = new Map();
function currencyFormatter(ccy, decimals) {
  const key = `${ccy}|${decimals}`;
  let f = currencyFormatters.get(key);
  if (!f) {
    try {
      f = new Intl.NumberFormat("en-GB", {
        style: "currency", currency: ccy, currencyDisplay: "narrowSymbol",
        minimumFractionDigits: decimals, maximumFractionDigits: decimals,
      });
    } catch (_) {
      f = new Intl.NumberFormat("en-GB", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    }
    currencyFormatters.set(key, f);
  }
  return f;
}

const numFormatters = new Map();
function numFormatter(min, max) {
  const key = `${min}|${max}`;
  let f = numFormatters.get(key);
  if (!f) {
    f = new Intl.NumberFormat("en-GB", { minimumFractionDigits: min, maximumFractionDigits: max });
    numFormatters.set(key, f);
  }
  return f;
}

function withSign(str, v, signed) {
  if (!signed) return str;
  if (v > 0) return "+" + str;
  return str; // Intl already renders the minus sign
}

/** Compact scale: 1234 → [1.23, "k"], 1.2e6 → [1.2, "M"]. */
function compactParts(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return [v / 1e9, "B"];
  if (a >= 1e6) return [v / 1e6, "M"];
  if (a >= 1e4) return [v / 1e3, "k"];
  return [v, ""];
}

/** money(1234.5) → "€1,234.50"; {compact:true} → "€1.2k"; {signed:true} → "+€1,234.50" */
export function money(v, ccy = "EUR", { compact = false, signed = false, decimals } = {}) {
  if (isNil(v)) return DASH;
  ccy = ccy || "EUR";
  if (compact) {
    const [n, suffix] = compactParts(v);
    const d = decimals ?? (suffix ? (Math.abs(n) >= 100 ? 0 : 1) : 0);
    const s = currencyFormatter(ccy, d).format(n) + suffix;
    return withSign(s, v, signed);
  }
  const d = decimals ?? 2;
  return withSign(currencyFormatter(ccy, d).format(v), v, signed);
}

/** pct(0.123) → "12.3%"; signed → "+12.3%" */
export function pct(v, { signed = false, decimals = 1 } = {}) {
  if (isNil(v)) return DASH;
  const s = numFormatter(decimals, decimals).format(v * 100) + "%";
  return withSign(s, v, signed);
}

/** Percentage points, for differences between two percentages: pp(0.021) → "+2.1 pp" */
export function pp(v, { signed = true, decimals = 1 } = {}) {
  if (isNil(v)) return DASH;
  const s = numFormatter(decimals, decimals).format(v * 100) + NBSP + "pp";
  return withSign(s, v, signed);
}

/** Plain number with fixed decimals: num(1.2345, {decimals:2}) → "1.23" */
export function num(v, { decimals = 2, signed = false, compact = false, min } = {}) {
  if (isNil(v)) return DASH;
  if (compact) {
    const [n, suffix] = compactParts(v);
    const d = suffix ? (Math.abs(n) >= 100 ? 0 : 1) : decimals;
    return withSign(numFormatter(0, d).format(n) + suffix, v, signed);
  }
  return withSign(numFormatter(min ?? decimals, decimals).format(v), v, signed);
}

/** Integer with thousands separators. */
export function int(v, { signed = false } = {}) {
  if (isNil(v)) return DASH;
  return withSign(numFormatter(0, 0).format(Math.round(v)), v, signed);
}

/** Multiples: mult(1.34) → "1.34×" */
export function mult(v, { decimals = 2 } = {}) {
  if (isNil(v)) return DASH;
  return numFormatter(decimals, decimals).format(v) + "×";
}

/** Quantity of units: up to 4 decimals, trailing zeros trimmed. */
export function qty(v) {
  if (isNil(v)) return DASH;
  return numFormatter(0, 4).format(v);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const MONTHS_SHORT = MONTHS;

function parseDate(v) {
  if (isNil(v) || v === "") return null;
  if (v instanceof Date) return Number.isNaN(v.getTime()) ? null : v;
  if (typeof v === "number") return new Date(v);
  const s = String(v);
  // Date-only ISO strings are interpreted as UTC by Date.parse; keep them calendar-stable.
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) {
    const [y, m, d] = s.split("-").map(Number);
    return new Date(y, m - 1, d);
  }
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** date("2026-09-04") → "4 Sep 2026"; {year:false} → "4 Sep"; {month:true} → "Sep 2026" */
export function date(v, { year = true, month = false } = {}) {
  const d = parseDate(v);
  if (!d) return DASH;
  if (month) return `${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
  return year ? `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}` : `${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

/** dateTime(iso) → "4 Sep 2026, 16:32" */
export function dateTime(v, { seconds = false } = {}) {
  const d = parseDate(v);
  if (!d) return DASH;
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = seconds ? ":" + String(d.getSeconds()).padStart(2, "0") : "";
  return `${date(d)}, ${hh}:${mm}${ss}`;
}

/** time(iso) → "16:32" */
export function time(v) {
  const d = parseDate(v);
  if (!d) return DASH;
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** relative(iso) → "just now" | "12 min ago" | "3 h ago" | "2 days ago" | "4 Sep 2026" */
export function relative(v, now = Date.now()) {
  const d = parseDate(v);
  if (!d) return DASH;
  const diff = Math.max(0, now - d.getTime());
  const s = Math.round(diff / 1000);
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  const days = Math.round(h / 24);
  if (days < 7) return days === 1 ? "yesterday" : `${days} days ago`;
  return date(d);
}

/** "prices as of 4 Sep 2026, 16:32 (delayed ~15 min)" — from snapshot.meta / a holding. */
export function delayNote(asOf, delayMinutes = 15, { prefix = "prices as of" } = {}) {
  if (isNil(asOf)) return `${prefix} ${DASH}`;
  const delay = delayMinutes == null ? "" : ` (delayed ~${delayMinutes} min)`;
  return `${prefix} ${dateTime(asOf)}${delay}`;
}

const NAME_NOISE = [
  /\bUCITS\b/gi, /\bETF\b/g, /\bETC\b/g, /\bAcc\b/g, /\bDist\b/g, /\bEUR\b/g, /\bUSD\b/g, /\bGBP\b/g,
  /\bHedged\b/gi, /\(.*?\)/g, /\bScreened\b/gi,
];

/** shortName("iShares Europe Defence UCITS ETF EUR Acc", 24) → "iShares Europe Defence"
    Drops fund-name suffixes first, then truncates with an ellipsis if still too long. */
export function shortName(name, n = 28) {
  if (isNil(name) || name === "") return DASH;
  let s = String(name);
  if (s.length <= n) return s;
  for (const re of NAME_NOISE) {
    s = s.replace(re, "").replace(/\s{2,}/g, " ").trim();
    if (s.length <= n) return s;
  }
  s = s.replace(/[\s\-–—,]+$/g, "");
  if (s.length <= n) return s;
  return s.slice(0, Math.max(1, n - 1)).trimEnd() + "…";
}

/** "pos" | "neg" | "" for colouring a signed number. Zero and null are silent. */
export function polarityClass(v) {
  if (isNil(v) || v === 0) return "";
  return v > 0 ? "pos" : "neg";
}

/** Arrow glyph for a delta: "▲" / "▼" / "". */
export function arrow(v) {
  if (isNil(v) || v === 0) return "";
  return v > 0 ? "▲" : "▼";
}

/** ISIN pretty-printer: keeps as-is, monospace is a styling concern. */
export function isin(v) {
  return isNil(v) || v === "" ? DASH : String(v).toUpperCase();
}

/** Text with a fallback for empty. */
export function text(v, fallback = DASH) {
  return isNil(v) || v === "" ? fallback : String(v);
}

/** Days → "252 d" and the like. */
export function days(v) {
  if (isNil(v)) return DASH;
  return `${int(v)}${NBSP}d`;
}

/** fmtCell(kind, opts) → (value, row) => string, for DataTable column `format`.
    kinds: money | money-compact | pct | pct-signed | pp | num | int | qty | mult | date | dateTime | relative | text | isin */
export function fmtCell(kind, opts = {}) {
  switch (kind) {
    case "money": return (v, row) => money(v, opts.currency || (row && row.currency) || "EUR", opts);
    case "money-compact": return (v, row) => money(v, opts.currency || (row && row.currency) || "EUR", { compact: true, ...opts });
    case "money-signed": return (v, row) => money(v, opts.currency || (row && row.currency) || "EUR", { signed: true, ...opts });
    case "pct": return (v) => pct(v, opts);
    case "pct-signed": return (v) => pct(v, { signed: true, ...opts });
    case "pp": return (v) => pp(v, opts);
    case "num": return (v) => num(v, opts);
    case "int": return (v) => int(v, opts);
    case "qty": return (v) => qty(v);
    case "mult": return (v) => mult(v, opts);
    case "date": return (v) => date(v, opts);
    case "dateTime": return (v) => dateTime(v, opts);
    case "relative": return (v) => relative(v);
    case "isin": return (v) => isin(v);
    case "days": return (v) => days(v);
    case "text":
    default: return (v) => text(v, opts.fallback);
  }
}

/** Format an axis tick for a given kind, keeping ticks short. */
export function axisTick(kind, ccy = "EUR") {
  switch (kind) {
    case "money": return (v) => money(v, ccy, { compact: true });
    case "pct": return (v) => pct(v, { decimals: Math.abs(v) < 0.01 && v !== 0 ? 2 : 0 });
    case "pct1": return (v) => pct(v, { decimals: 1 });
    case "index": return (v) => num(v, { decimals: 0 });
    case "num": return (v) => num(v, { decimals: 2, compact: true });
    default: return (v) => String(v);
  }
}
