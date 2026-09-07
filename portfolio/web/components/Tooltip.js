/* Tooltip — CSS-only via data-tip on any element (see components.css).
   Tooltip({ text, pos: "bottom"|"top"|"right"|"bottom-right", children })  wraps children in a span.
   Help({ text }) renders a small (i) glyph with the tooltip. */

import { html } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";

export function Tooltip({ text, pos, children, class: cls = "", inline = true }) {
  if (!text) return children;
  return html`<span class=${cls} data-tip=${text} data-tip-pos=${pos} style=${inline ? "display:inline-flex;align-items:center" : undefined} tabindex="0">${children}</span>`;
}

export function Help({ text, pos = "bottom" }) {
  return html`<span class="faint" data-tip=${text} data-tip-pos=${pos} tabindex="0" style="display:inline-flex;cursor:help" aria-label=${text}>
    <${Icon} name="info" size=${13} />
  </span>`;
}

export default Tooltip;
