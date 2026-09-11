/* Notice — the graded warning treatment.
   One summary line (worst severity, count), up to `visible` quiet items, and
   a disclosure for the rest. Nothing shouts; a holding behaving as expected
   is visually silent, so an empty list renders nothing.

   Notice({ alerts: [{ code, severity, title, detail, isins, route }], visible = 3,
            summary?: string, onNavigate?: (route) => void, compact })
   Also: severityRank(severity), worstSeverity(alerts). */

import { html } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";
import { SeverityBadge, severityIcon } from "/static/components/Badge.js";
import { href } from "/static/lib/router.js";

const RANK = { INFO: 0, WARNING: 1, SERIOUS: 2, CRITICAL: 3 };
export const severityRank = (s) => RANK[s] ?? 0;
export function worstSeverity(alerts = []) {
  return alerts.reduce((w, a) => (severityRank(a.severity) > severityRank(w) ? a.severity : w), "INFO");
}

function plural(n, one, many) { return `${n} ${n === 1 ? one : many || one + "s"}`; }

function defaultSummary(alerts) {
  const counts = { CRITICAL: 0, SERIOUS: 0, WARNING: 0, INFO: 0 };
  for (const a of alerts) counts[a.severity] = (counts[a.severity] || 0) + 1;
  const parts = [];
  if (counts.CRITICAL) parts.push(plural(counts.CRITICAL, "critical issue"));
  if (counts.SERIOUS) parts.push(plural(counts.SERIOUS, "serious issue"));
  if (counts.WARNING) parts.push(plural(counts.WARNING, "warning"));
  if (counts.INFO) parts.push(plural(counts.INFO, "note"));
  return parts.join(", ");
}

function Item({ a, onNavigate }) {
  const link = a.route ? href(a.route) : null;
  return html`<li class="notice-item">
    <${SeverityBadge} severity=${a.severity} showIcon=${false} />
    <div class="grow">
      <span class="notice-item-title">${a.title}</span>
      ${a.detail ? html` <span class="notice-item-detail">— ${a.detail}</span>` : null}
      ${link ? html` <a href=${link} onClick=${onNavigate ? (e) => { e.preventDefault(); onNavigate(a.route); } : undefined}>Open →</a>` : null}
    </div>
  </li>`;
}

export function Notice({ alerts = [], visible = 3, summary, onNavigate, compact = false, class: cls = "" }) {
  if (!alerts || !alerts.length) return null;
  const sorted = [...alerts].sort((a, b) => severityRank(b.severity) - severityRank(a.severity) || a.title.localeCompare(b.title));
  const worst = sorted[0].severity;
  const head = sorted.slice(0, visible);
  const rest = sorted.slice(visible);
  return html`<section class=${["notice", cls].filter(Boolean).join(" ")} data-severity=${worst} aria-label="Warnings">
    <div class="notice-summary">
      <span class="notice-icon"><${Icon} name=${severityIcon(worst)} size=${15} /></span>
      <span class="notice-text">${summary || defaultSummary(sorted)}</span>
      ${compact ? null : html`<span class="notice-count">${alerts.length}</span>`}
    </div>
    ${compact ? null : html`<ul class="notice-items">${head.map((a) => html`<${Item} key=${a.code + a.title} a=${a} onNavigate=${onNavigate} />`)}</ul>`}
    ${!compact && rest.length ? html`<details class="notice-more">
      <summary><${Icon} name="chevronRight" size=${14} />${plural(rest.length, "more item")}</summary>
      <ul class="notice-items">${rest.map((a) => html`<${Item} key=${a.code + a.title} a=${a} onNavigate=${onNavigate} />`)}</ul>
    </details>` : null}
  </section>`;
}

export default Notice;
