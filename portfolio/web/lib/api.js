/* API client. Every endpoint in docs/ARCHITECTURE.md has a helper here.
   Errors arrive as {error:{code,message}} and are thrown as ApiError.

   loadSnapshot() keeps the previous snapshot on screen while the next one is
   in flight (store.loading = true) and aborts a superseded request. */

import { app } from "/static/lib/store.js";

export class ApiError extends Error {
  constructor(message, { code = "error", status = 0, cause } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    if (cause) this.cause = cause;
  }
}

const BASE = "/api";

async function request(method, path, body, { signal, raw = false } = {}) {
  const init = { method, headers: { Accept: "application/json" }, signal };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(BASE + path, init);
  } catch (err) {
    if (err && err.name === "AbortError") throw err;
    throw new ApiError("The server could not be reached.", { code: "network", status: 0, cause: err });
  }
  if (raw) return res;
  const type = res.headers.get("content-type") || "";
  let data = null;
  if (type.includes("application/json")) {
    try { data = await res.json(); } catch (_) { data = null; }
  } else if (res.status !== 204) {
    const txt = await res.text().catch(() => "");
    data = txt ? { text: txt } : null;
  }
  if (!res.ok) {
    const env = data && data.error ? data.error : {};
    throw new ApiError(env.message || (data && data.detail) || `${res.status} ${res.statusText}`, {
      code: env.code || `http_${res.status}`,
      status: res.status,
    });
  }
  return data;
}

export const api = {
  get: (path, opts) => request("GET", path, undefined, opts),
  post: (path, body, opts) => request("POST", path, body ?? {}, opts),
  put: (path, body, opts) => request("PUT", path, body ?? {}, opts),
  patch: (path, body, opts) => request("PATCH", path, body ?? {}, opts),
  del: (path, opts) => request("DELETE", path, undefined, opts),
  /** Trigger a browser download of an API file (export endpoints). */
  download: async (path, filename) => {
    const res = await request("GET", path, undefined, { raw: true });
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try { const j = await res.json(); msg = j.error?.message || msg; } catch (_) { /* not json */ }
      throw new ApiError(msg, { status: res.status, code: `http_${res.status}` });
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || path.split("/").pop() || "download";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  },
};

/* ---------------- snapshot ---------------- */

let snapshotController = null;

function snapshotQuery({ benchmark, lookback }) {
  const q = new URLSearchParams();
  if (benchmark) q.set("benchmark", benchmark);
  if (lookback) q.set("lookback", String(lookback));
  const s = q.toString();
  return s ? `?${s}` : "";
}

/** Load /api/snapshot into the store. Previous data stays visible while loading.
    Returns the snapshot, or null when superseded by a newer call. */
export async function loadSnapshot({ benchmark, lookback, force = false } = {}) {
  const state = app.get();
  const b = benchmark === undefined ? state.benchmark : benchmark;
  const l = lookback === undefined ? state.lookback : lookback;
  if (snapshotController) snapshotController.abort();
  const controller = new AbortController();
  snapshotController = controller;
  app.patch({ loading: true, error: null });
  try {
    const snap = await api.get(`/snapshot${snapshotQuery({ benchmark: b, lookback: l })}`, { signal: controller.signal });
    if (controller !== snapshotController) return null;
    // store.benchmark / store.lookback stay the *requested* values (null = server
    // default); overwriting them from the response would retrigger the effect
    // that loads on change and fetch the snapshot twice. The selected benchmark
    // for display is snapshot.selected_benchmark.
    app.patch({ snapshot: snap, loading: false, loadedAt: new Date(), error: null });
    return snap;
  } catch (err) {
    if (err && err.name === "AbortError") return null;
    if (controller === snapshotController) app.patch({ loading: false, error: err });
    throw err;
  } finally {
    if (controller === snapshotController) snapshotController = null;
  }
}

/** POST /api/refresh (bypasses the quote TTL) and reload the snapshot. */
export async function refreshPrices() {
  await api.post("/refresh", {});
  return loadSnapshot({ force: true });
}

/* ---------------- health & settings ---------------- */

export const getHealth = () => api.get("/health");

export async function getSettings() {
  const settings = await api.get("/settings");
  app.patch({ settings });
  return settings;
}

/** PUT /api/settings {mode: "seed" | "user"} */
export async function putSettings(patch) {
  const settings = await api.put("/settings", patch);
  app.patch({ settings });
  return settings;
}

/* ---------------- holdings & instruments ---------------- */

export const getHoldingDetail = (isin) => api.get(`/holdings/${encodeURIComponent(isin)}`);
export const listInstruments = () => api.get("/instruments");
export const resolveIsin = (isin, lookback) => api.post("/instruments/resolve", lookback ? { isin, lookback } : { isin });
export const saveInstrument = (body) => api.post("/instruments", body);
export const patchInstrument = (isin, patch) => api.patch(`/instruments/${encodeURIComponent(isin)}`, patch);
export const deleteInstrument = (isin) => api.del(`/instruments/${encodeURIComponent(isin)}`);

/* ---------------- transactions ---------------- */

export const listTransactions = ({ includeVoided = false } = {}) =>
  api.get(`/transactions${includeVoided ? "?include_voided=true" : ""}`);
export const addTransaction = (body) => api.post("/transactions", body);
export const voidTransaction = (id, reason = "") => api.post(`/transactions/${encodeURIComponent(id)}/void`, { reason });
export const previewImport = (text, mapping) => api.post("/transactions/import/preview", mapping ? { text, mapping } : { text });
export const runImport = (text, mapping) => api.post("/transactions/import", mapping ? { text, mapping } : { text });

/* ---------------- simulator, benchmarks, exports ---------------- */

/** simulate({changes} | {targets} | {preset}, include?, min_trade?) */
export const simulate = (body) => api.post("/simulate", body);
export const listBenchmarks = () => api.get("/benchmarks");

export const exportTransactionsCsv = () => api.download("/export/transactions.csv", "transactions.csv");
export const exportHoldingsCsv = () => api.download("/export/holdings.csv", "holdings.csv");
export const exportWorkbook = () => api.download("/export/workbook.xlsx", "portfolio.xlsx");

/** Human-readable message for any thrown error. */
export function errorMessage(err) {
  if (!err) return "Unknown error";
  if (err instanceof ApiError) return err.message;
  if (typeof err === "string") return err;
  return err.message || String(err);
}
