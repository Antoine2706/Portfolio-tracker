/* Badge({ kind, children, size: "sm"|"lg", dot, outline, icon, title })
   Semantic helpers: SeverityBadge({ severity }), VerdictBadge({ verdict }),
   TxTypeBadge({ type }), ModeBadge({ mode }). */

import { html } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";

const KINDS = new Set(["neutral", "info", "good", "warn", "serious", "bad", "live", "demo"]);

export function Badge({ kind = "neutral", children, size, dot = false, outline = false, icon, title, tip, tipPos, class: cls = "" }) {
  const k = KINDS.has(kind) ? kind : "neutral";
  const classes = ["badge", `badge-${k}`, size === "lg" ? "badge-lg" : "", dot ? "badge-dot" : "", outline ? "badge-outline" : "", cls].filter(Boolean).join(" ");
  return html`<span class=${classes} title=${title} data-tip=${tip} data-tip-pos=${tipPos}>
    ${icon ? html`<${Icon} name=${icon} size=${11} stroke=${2} />` : null}${children}
  </span>`;
}

const SEVERITY = {
  INFO: { kind: "info", icon: "info" },
  WARNING: { kind: "warn", icon: "alertTriangle" },
  SERIOUS: { kind: "serious", icon: "alertCircle" },
  CRITICAL: { kind: "bad", icon: "alertOctagon" },
};
export function severityKind(severity) { return (SEVERITY[severity] || SEVERITY.INFO).kind; }
export function severityIcon(severity) { return (SEVERITY[severity] || SEVERITY.INFO).icon; }

/** INFO / WARNING / SERIOUS / CRITICAL — always with an icon, never colour alone. */
export function SeverityBadge({ severity, size, showIcon = true, ...rest }) {
  const s = SEVERITY[severity] || SEVERITY.INFO;
  return html`<${Badge} kind=${s.kind} size=${size} icon=${showIcon ? s.icon : undefined} ...${rest}>${severity || "INFO"}<//>`;
}

const VERDICT = {
  PASS: { kind: "good", icon: "check" },
  THIN: { kind: "warn", icon: "alertTriangle" },
  STALE: { kind: "warn", icon: "clock" },
  FAILED: { kind: "bad", icon: "xCircle" },
  REFUSED: { kind: "serious", icon: "alertOctagon" },
  ABSENT: { kind: "neutral", icon: "minus" },
};
/** Resolution / provider verdicts. */
export function VerdictBadge({ verdict, size, ...rest }) {
  const v = VERDICT[String(verdict || "").toUpperCase()] || { kind: "neutral" };
  return html`<${Badge} kind=${v.kind} size=${size} icon=${v.icon} ...${rest}>${verdict}<//>`;
}

const TX = {
  BUY: "info", SELL: "neutral", DIVIDEND: "good", FEE: "warn",
};
/** BUY / SELL / DIVIDEND / FEE */
export function TxTypeBadge({ type, size, ...rest }) {
  const k = TX[String(type || "").toUpperCase()] || "neutral";
  return html`<${Badge} kind=${k} size=${size} outline=${k === "neutral"} ...${rest}>${type}<//>`;
}

/** DEMO (seed data) amber / LIVE (user data) green. */
export function ModeBadge({ mode, size = "md" }) {
  const live = mode === "user";
  const tip = live
    ? "LIVE: your own instruments and ledger (PORTFOLIO_DATA_MODE=user)"
    : "DEMO: synthetic seed data. Nothing here is your portfolio.";
  return html`<${Badge} kind=${live ? "live" : "demo"} size=${size} dot title=${tip}>${live ? "LIVE" : "DEMO"}<//>`;
}

export default Badge;
