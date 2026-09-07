/* EmptyState({ icon, title, body, action, compact }) */

import { html } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";

export function EmptyState({ icon = "inbox", title, body, action, compact = false, class: cls = "" }) {
  return html`<div class=${["empty", compact ? "compact" : "", cls].filter(Boolean).join(" ")}>
    ${icon ? html`<div class="empty-icon"><${Icon} name=${icon} size=${18} /></div>` : null}
    ${title ? html`<div class="empty-title">${title}</div>` : null}
    ${body ? html`<div class="empty-body">${body}</div>` : null}
    ${action ? html`<div class="empty-action">${action}</div>` : null}
  </div>`;
}

export default EmptyState;
