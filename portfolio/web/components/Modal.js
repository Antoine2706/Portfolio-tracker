/* Modal({ open, title, onClose, size: "md"|"lg", footer, children })
   confirm({ title, body, confirmLabel = "Confirm", cancelLabel = "Cancel", danger }) → Promise<boolean>
   ModalHost renders store.modal (mounted once by app.js). */

import { html, useRef } from "/static/vendor/preact-htm.module.js";
import { Button } from "/static/components/Button.js";
import { useFocusTrap } from "/static/components/Drawer.js";
import { useStore, openModal, closeModal } from "/static/lib/store.js";

export function Modal({ open = true, title, onClose, size = "md", footer, children, ariaLabel, class: cls = "" }) {
  const ref = useRef(null);
  useFocusTrap(ref, open, { onEscape: onClose });
  if (!open) return null;
  return html`<div class="modal-root">
    <div class="modal-backdrop" onClick=${onClose}></div>
    <div ref=${ref} class=${["modal", size === "lg" ? "lg" : "", cls].filter(Boolean).join(" ")} role="dialog" aria-modal="true" aria-label=${ariaLabel || (typeof title === "string" ? title : "Dialog")} tabindex="-1">
      ${title ? html`<header class="modal-header">
        <h2 class="modal-title">${title}</h2>
        <${Button} variant="ghost" icon="x" iconOnly title="Close (Esc)" onClick=${onClose} />
      </header>` : null}
      <div class="modal-body">${children}</div>
      ${footer ? html`<footer class="modal-footer">${footer}</footer>` : null}
    </div>
  </div>`;
}

export function ModalHost() {
  const modal = useStore((s) => s.modal);
  if (!modal) return null;
  const close = () => { closeModal(); if (modal.onClose) modal.onClose(); };
  const body = typeof modal.body === "function" ? modal.body() : modal.body;
  const footer = typeof modal.footer === "function" ? modal.footer() : modal.footer;
  return html`<${Modal} open title=${modal.title} size=${modal.size} onClose=${close} footer=${footer}>${body}<//>`;
}

/** Promise-based confirmation dialog. */
export function confirm({ title = "Are you sure?", body, confirmLabel = "Confirm", cancelLabel = "Cancel", danger = false } = {}) {
  return new Promise((resolve) => {
    const done = (v) => { closeModal(); resolve(v); };
    openModal({
      title,
      body,
      onClose: () => resolve(false),
      footer: html`
        <${Button} variant="secondary" onClick=${() => done(false)}>${cancelLabel}<//>
        <${Button} variant=${danger ? "danger" : "primary"} onClick=${() => done(true)} data-autofocus>${confirmLabel}<//>
      `,
    });
  });
}

export default Modal;
