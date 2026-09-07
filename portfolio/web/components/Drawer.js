/* Drawer({ open, title, subtitle, onClose, width: "default"|"wide", footer, flush, children, badge })
   Right-side panel, 520px (720px wide). Closes on ESC, backdrop, or the × button.
   Focus is trapped while open and restored on close. The shell mounts one
   <DrawerHost /> that renders store.drawer (see openDrawer in lib/store.js). */

import { html, useEffect, useRef } from "/static/vendor/preact-htm.module.js";
import { Button } from "/static/components/Button.js";
import { app, useStore, closeDrawer } from "/static/lib/store.js";

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function useFocusTrap(ref, active, { onEscape, initialFocus = true } = {}) {
  useEffect(() => {
    if (!active || !ref.current) return undefined;
    const root = ref.current;
    const previous = document.activeElement;
    if (initialFocus) {
      const auto = root.querySelector("[data-autofocus]") || root.querySelector(FOCUSABLE);
      const target = auto || root;
      setTimeout(() => target && target.focus && target.focus({ preventScroll: true }), 0);
    }
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); onEscape && onEscape(); return; }
      if (e.key !== "Tab") return;
      const items = Array.from(root.querySelectorAll(FOCUSABLE)).filter((el) => el.offsetParent !== null || el === document.activeElement);
      if (!items.length) { e.preventDefault(); return; }
      const first = items[0], last = items[items.length - 1];
      if (e.shiftKey && (document.activeElement === first || !root.contains(document.activeElement))) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    root.addEventListener("keydown", onKey);
    return () => {
      root.removeEventListener("keydown", onKey);
      if (previous && previous.focus && document.contains(previous)) previous.focus({ preventScroll: true });
    };
  }, [active]);
}

export function Drawer({ open = false, title, subtitle, onClose, width = "default", footer, flush = false, children, badge, ariaLabel }) {
  const ref = useRef(null);
  useFocusTrap(ref, open, { onEscape: onClose });
  return html`<div class="drawer-root" data-open=${open ? "true" : "false"} aria-hidden=${open ? "false" : "true"}>
    <div class="drawer-backdrop" onClick=${onClose}></div>
    <aside ref=${ref} class=${["drawer", width === "wide" ? "wide" : ""].filter(Boolean).join(" ")} role="dialog" aria-modal="true" aria-label=${ariaLabel || (typeof title === "string" ? title : "Details")} tabindex="-1">
      ${open ? html`
        <header class="drawer-header">
          <div class="grow">
            <div class="row"><h2 class="drawer-title truncate">${title}</h2>${badge}</div>
            ${subtitle ? html`<div class="drawer-subtitle">${subtitle}</div>` : null}
          </div>
          <${Button} variant="ghost" icon="x" iconOnly title="Close (Esc)" onClick=${onClose} />
        </header>
        <div class=${["drawer-body", flush ? "flush" : ""].filter(Boolean).join(" ")}>${children}</div>
        ${footer ? html`<footer class="drawer-footer">${footer}</footer>` : null}
      ` : null}
    </aside>
  </div>`;
}

/** Renders the drawer described by store.drawer. Mounted once by app.js. */
export function DrawerHost() {
  const drawer = useStore((s) => s.drawer);
  const body = drawer ? (typeof drawer.body === "function" ? drawer.body() : drawer.body) : null;
  return html`<${Drawer} open=${!!drawer} title=${drawer && drawer.title} subtitle=${drawer && drawer.subtitle} width=${drawer && drawer.width}
    footer=${drawer && drawer.footer} flush=${drawer && drawer.flush} badge=${drawer && drawer.badge} onClose=${() => { closeDrawer(); if (drawer && drawer.onAfterClose) drawer.onAfterClose(); }}>
    ${body}
  <//>`;
}

export { app };
export default Drawer;
