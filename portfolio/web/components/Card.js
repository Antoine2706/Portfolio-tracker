/* Card({ title, caption, actions, footer, flush, tight, bordered, class, interactive, onClick, children })
   A surface with an optional header row (title/caption left, actions right). */

import { html } from "/static/vendor/preact-htm.module.js";

export function Card({ title, caption, actions, footer, flush = false, tight = false, bordered = false, class: cls = "", interactive = false, onClick, children, style, id }) {
  const hasHeader = title || caption || actions;
  return html`<section class=${["card", interactive ? "interactive" : "", cls].filter(Boolean).join(" ")} onClick=${onClick} style=${style} id=${id}>
    ${hasHeader ? html`<header class=${"card-header" + (bordered ? " bordered" : "")}>
      <div class="grow">
        ${title ? html`<h3 class="card-title">${title}</h3>` : null}
        ${caption ? html`<div class="card-caption">${caption}</div>` : null}
      </div>
      ${actions ? html`<div class="card-actions">${actions}</div>` : null}
    </header>` : null}
    <div class=${["card-body", flush ? "flush" : "", tight ? "tight" : ""].filter(Boolean).join(" ")}>${children}</div>
    ${footer ? html`<footer class="card-footer">${footer}</footer>` : null}
  </section>`;
}

export default Card;
