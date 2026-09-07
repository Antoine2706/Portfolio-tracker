/* Banner({ kind: "info"|"warn"|"error"|"good"|"neutral", icon, title, children, action, onDismiss }) */

import { html } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";
import { Button } from "/static/components/Button.js";

const ICONS = { info: "info", warn: "alertTriangle", error: "alertCircle", good: "checkCircle", neutral: "info" };

export function Banner({ kind = "neutral", icon, title, children, action, onDismiss, class: cls = "" }) {
  return html`<div class=${["banner", `banner-${kind}`, cls].filter(Boolean).join(" ")} role=${kind === "error" ? "alert" : "status"}>
    <span class="banner-icon"><${Icon} name=${icon || ICONS[kind] || "info"} size=${15} /></span>
    <div class="banner-text">${title ? html`<strong>${title}</strong> ` : null}${children}</div>
    ${action ? html`<div>${action}</div>` : null}
    ${onDismiss ? html`<${Button} variant="ghost" size="sm" icon="x" iconOnly title="Dismiss" onClick=${onDismiss} />` : null}
  </div>`;
}

export default Banner;
