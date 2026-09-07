/* toast(message, { kind: "success"|"warn"|"error"|"info", title, timeout }) → id
   dismissToast(id); <Toasts /> renders the stack (mounted once by app.js). */

import { html, useEffect, useState } from "/static/vendor/preact-htm.module.js";
import { app, useStore } from "/static/lib/store.js";
import { Icon } from "/static/components/Icons.js";

let seq = 0;
const timers = new Map();

const ICONS = { success: "checkCircle", warn: "alertTriangle", error: "alertCircle", info: "info" };
const DEFAULT_TIMEOUT = { success: 3500, info: 4000, warn: 6000, error: 8000 };

export function toast(message, { kind = "info", title, timeout } = {}) {
  const id = ++seq;
  const t = { id, kind, title, message: String(message ?? ""), timeout: timeout ?? DEFAULT_TIMEOUT[kind] ?? 4000 };
  app.patch({ toasts: [...app.get().toasts, t].slice(-5) });
  if (t.timeout > 0) timers.set(id, setTimeout(() => dismissToast(id), t.timeout));
  return id;
}
toast.success = (m, o) => toast(m, { ...o, kind: "success" });
toast.warn = (m, o) => toast(m, { ...o, kind: "warn" });
toast.error = (m, o) => toast(m, { ...o, kind: "error" });
toast.info = (m, o) => toast(m, { ...o, kind: "info" });

export function dismissToast(id) {
  const tm = timers.get(id);
  if (tm) { clearTimeout(tm); timers.delete(id); }
  app.patch({ toasts: app.get().toasts.filter((t) => t.id !== id) });
}

function ToastItem({ t }) {
  const [leaving, setLeaving] = useState(false);
  useEffect(() => {
    if (t.timeout <= 0) return undefined;
    const h = setTimeout(() => setLeaving(true), Math.max(0, t.timeout - 160));
    return () => clearTimeout(h);
  }, [t.timeout]);
  return html`<div class=${["toast", `toast-${t.kind}`, leaving ? "leaving" : ""].filter(Boolean).join(" ")} role=${t.kind === "error" ? "alert" : "status"}>
    <span class="toast-icon"><${Icon} name=${ICONS[t.kind] || "info"} size=${15} /></span>
    <div class="toast-body">
      ${t.title ? html`<div class="toast-title">${t.title}</div>` : null}
      <div class=${t.title ? "toast-msg" : ""}>${t.message}</div>
    </div>
    <button type="button" class="toast-close" aria-label="Dismiss" onClick=${() => dismissToast(t.id)}><${Icon} name="x" size=${14} /></button>
  </div>`;
}

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  if (!toasts.length) return null;
  return html`<div class="toast-stack" aria-live="polite">${toasts.map((t) => html`<${ToastItem} key=${t.id} t=${t} />`)}</div>`;
}

export default toast;
