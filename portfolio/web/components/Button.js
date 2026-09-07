/* Button({ variant: "primary"|"secondary"|"ghost"|"danger", size: "sm"|"md"|"lg",
            icon, iconRight, loading, disabled, type, onClick, title, tip, tipPos, class, href, ...rest }) */

import { html } from "/static/vendor/preact-htm.module.js";
import { Icon } from "/static/components/Icons.js";

export function Button({
  variant = "secondary", size = "md", icon, iconRight, loading = false, disabled = false,
  type = "button", onClick, title, tip, tipPos, class: cls = "", children, href, iconOnly, ariaLabel, ...rest
}) {
  const classes = [
    "btn", `btn-${variant}`,
    size !== "md" ? `btn-${size}` : "",
    iconOnly || (!children && icon) ? "btn-icon" : "",
    loading ? "is-loading" : "",
    cls,
  ].filter(Boolean).join(" ");
  const content = html`
    ${loading ? html`<span class="spinner" aria-hidden="true"></span>` : icon ? html`<${Icon} name=${icon} size=${size === "sm" ? 14 : 16} />` : null}
    ${children ? html`<span class="btn-label">${children}</span>` : null}
    ${iconRight ? html`<${Icon} name=${iconRight} size=${size === "sm" ? 14 : 16} />` : null}
  `;
  const common = {
    class: classes, title, "data-tip": tip, "data-tip-pos": tipPos,
    "aria-label": ariaLabel || (iconOnly ? title || tip : undefined), "aria-busy": loading ? "true" : undefined,
    ...rest,
  };
  if (href) return html`<a href=${href} ...${common} onClick=${onClick}>${content}</a>`;
  return html`<button type=${type} disabled=${disabled || loading} onClick=${onClick} ...${common}>${content}</button>`;
}

export default Button;
