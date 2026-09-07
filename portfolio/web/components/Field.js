/* Field({ label, hint, error, required, inline, id, children })
   Input({ type, value, onInput, onChange, placeholder, numeric, prefix, suffix, error, ...})
   Select({ value, options: [{value,label}] | ["a","b"], onChange, compact, placeholder })
   Textarea({ value, onInput, rows })
   Checkbox({ checked, onChange, label }) */

import { html } from "/static/vendor/preact-htm.module.js";

let uid = 0;
const nextId = () => `f${++uid}`;

export function Field({ label, hint, error, required = false, inline = false, id, children, class: cls = "" }) {
  return html`<div class=${["field", inline ? "inline" : "", cls].filter(Boolean).join(" ")}>
    ${label ? html`<label class="field-label" for=${id}>${label}${required ? html`<span class="req" aria-hidden="true">*</span>` : null}</label>` : null}
    ${children}
    ${error ? html`<div class="field-error" role="alert">${error}</div>` : hint ? html`<div class="field-hint">${hint}</div>` : null}
  </div>`;
}

export function Input({ type = "text", value, onInput, onChange, placeholder, numeric = false, prefix, suffix, error, id, disabled, autoFocus, class: cls = "", inputRef, ...rest }) {
  const input = html`<input ref=${inputRef} id=${id} type=${type} class=${["input", numeric ? "num" : "", cls].filter(Boolean).join(" ")}
    value=${value ?? ""} onInput=${onInput} onChange=${onChange} placeholder=${placeholder}
    inputMode=${numeric ? "decimal" : undefined} aria-invalid=${error ? "true" : undefined}
    disabled=${disabled} autofocus=${autoFocus} spellcheck=${type === "text" && numeric ? "false" : undefined} ...${rest} />`;
  if (!prefix && !suffix) return input;
  return html`<div class=${"input-wrap" + (suffix ? " has-suffix" : "")}>
    ${prefix ? html`<span class="input-prefix">${prefix}</span>` : null}
    ${input}
    ${suffix ? html`<span class="input-suffix">${suffix}</span>` : null}
  </div>`;
}

export function Select({ value, options = [], onChange, compact = false, placeholder, id, disabled, error, class: cls = "", ...rest }) {
  const opts = options.map((o) => (typeof o === "object" && o !== null ? o : { value: o, label: String(o) }));
  return html`<select id=${id} class=${["select", compact ? "compact" : "", cls].filter(Boolean).join(" ")} value=${value ?? ""}
    onChange=${(e) => onChange && onChange(e.target.value, e)} disabled=${disabled} aria-invalid=${error ? "true" : undefined} ...${rest}>
    ${placeholder ? html`<option value="" disabled selected=${value == null || value === ""}>${placeholder}</option>` : null}
    ${opts.map((o) => html`<option key=${o.value} value=${o.value} disabled=${o.disabled} selected=${String(o.value) === String(value)}>${o.label}</option>`)}
  </select>`;
}

export function Textarea({ value, onInput, onChange, placeholder, rows = 3, id, disabled, error, class: cls = "", ...rest }) {
  return html`<textarea id=${id} class=${["textarea", cls].filter(Boolean).join(" ")} rows=${rows} value=${value ?? ""}
    onInput=${onInput} onChange=${onChange} placeholder=${placeholder} disabled=${disabled}
    aria-invalid=${error ? "true" : undefined} ...${rest}></textarea>`;
}

export function Checkbox({ checked = false, onChange, label, disabled, id }) {
  return html`<label class="checkbox" for=${id}>
    <input id=${id} type="checkbox" checked=${checked} disabled=${disabled} onChange=${(e) => onChange && onChange(e.target.checked, e)} />
    ${label ? html`<span>${label}</span>` : null}
  </label>`;
}

export { nextId as fieldId };
export default Field;
