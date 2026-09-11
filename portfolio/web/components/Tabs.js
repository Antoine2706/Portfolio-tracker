/* Tabs({ tabs: [{ id, label, count, icon }], active, onChange, children })
   Renders the tab strip; `children` (optional) is the active panel.
   Arrow keys move between tabs. */

import { html } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";

export function Tabs({ tabs = [], active, onChange, children, class: cls = "" }) {
  const onKey = (e, i) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const dir = e.key === "ArrowRight" ? 1 : -1;
    const next = tabs[(i + dir + tabs.length) % tabs.length];
    if (next && onChange) onChange(next.id);
    const btn = e.currentTarget.parentElement.querySelector(`[data-tab="${next.id}"]`);
    if (btn) btn.focus();
  };
  return html`<div class=${cls}>
    <div class="tabs" role="tablist">
      ${tabs.map((t, i) => html`<button key=${t.id} type="button" role="tab" class="tab" data-tab=${t.id}
        aria-selected=${t.id === active ? "true" : "false"} tabindex=${t.id === active ? 0 : -1}
        onClick=${() => onChange && onChange(t.id)} onKeyDown=${(e) => onKey(e, i)}>
        ${t.icon ? html`<${Icon} name=${t.icon} size=${14} />` : null}
        ${t.label}
        ${t.count != null ? html`<span class="tab-count">${t.count}</span>` : null}
      </button>`)}
    </div>
    ${children ? html`<div class="tab-panel" role="tabpanel">${children}</div>` : null}
  </div>`;
}

export default Tabs;
