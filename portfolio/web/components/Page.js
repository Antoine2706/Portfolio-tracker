/* Page({ title, subtitle, actions, children, wide })
   Section({ title, caption, actions, children }) — a titled block inside a page. */

import { html } from "/static/vendor/preact-htm.module.js";

export function Page({ title, subtitle, actions, children, class: cls = "", wide = false }) {
  return html`<div class=${["page", cls].filter(Boolean).join(" ")} style=${wide ? "max-width:none" : undefined}>
    ${title || actions ? html`<header class="page-header">
      <div class="page-titles">
        ${title ? html`<h1>${title}</h1>` : null}
        ${subtitle ? html`<div class="page-subtitle">${subtitle}</div>` : null}
      </div>
      ${actions ? html`<div class="page-actions">${actions}</div>` : null}
    </header>` : null}
    ${children}
  </div>`;
}

export function Section({ title, caption, actions, children, class: cls = "" }) {
  return html`<section class=${["section", cls].filter(Boolean).join(" ")}>
    ${title || actions ? html`<div class="section-title">
      <div class="row gap-3"><h2>${title}</h2>${caption ? html`<span class="section-caption">${caption}</span>` : null}</div>
      ${actions ? html`<div class="row">${actions}</div>` : null}
    </div>` : null}
    ${children}
  </section>`;
}

export default Page;
