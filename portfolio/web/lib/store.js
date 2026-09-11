/* Tiny observable store.
   createStore(initial) → { get, set, patch, subscribe }
   useStore(selector, store = app) → the selected slice, re-rendering only when it changes.

   The global `app` store holds everything the shell and pages share. Pages
   read from it with useStore and write through lib/api.js or the helpers
   below; nothing else mutates it. */

import { useState, useEffect, useRef } from "/static/vendor/preact-htm.module.js";

export function createStore(initial = {}) {
  let state = initial;
  const listeners = new Set();

  const get = () => state;
  const set = (next) => {
    const value = typeof next === "function" ? next(state) : next;
    if (value === state) return;
    state = value;
    listeners.forEach((fn) => fn(state));
  };
  const patch = (partial) => {
    const delta = typeof partial === "function" ? partial(state) : partial;
    if (!delta) return;
    let changed = false;
    for (const k in delta) if (!Object.is(state[k], delta[k])) { changed = true; break; }
    if (!changed) return;
    set({ ...state, ...delta });
  };
  const subscribe = (fn) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  };
  return { get, set, patch, subscribe };
}

const identity = (s) => s;

/** Subscribe a component to a slice of a store. Re-renders when the selected
    value changes by Object.is (select primitives or stable references). */
export function useStore(selector = identity, store = app) {
  const select = useRef(selector);
  select.current = selector;
  const [slice, setSlice] = useState(() => selector(store.get()));
  const last = useRef(slice);
  useEffect(() => {
    const check = (s) => {
      const next = select.current(s);
      if (!Object.is(next, last.current)) {
        last.current = next;
        setSlice(() => next);
      }
    };
    check(store.get());
    return store.subscribe(check);
  }, [store]);
  return slice;
}

function readLocal(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v == null ? fallback : JSON.parse(v);
  } catch (_) {
    return fallback;
  }
}

export const LOOKBACKS = [63, 126, 252, 504];

/** The global application store. */
export const app = createStore({
  snapshot: null,          // Snapshot (schemas.Snapshot) or null before first load
  settings: null,          // Settings (schemas.Settings) or null
  loading: false,          // a snapshot request is in flight (previous snapshot stays visible)
  loadedAt: null,          // Date of the last successful snapshot
  error: null,             // ApiError | Error from the last failed snapshot load, or null
  theme: document.documentElement.getAttribute("data-theme") || "dark",
  benchmark: readLocal("pt.benchmark", null),   // benchmark symbol; null = server default
  lookback: readLocal("pt.lookback", 252),      // trading days for the risk window
  route: { page: "overview", params: {}, path: "/overview" },
  drawer: null,            // { title, subtitle, body, width, footer, onClose } | null
  modal: null,             // { title, body, actions, onClose } | null
  paletteOpen: false,
  sidebarCollapsed: readLocal("pt.sidebar", false),
  toasts: [],              // [{ id, kind, title, message, timeout }]
});

/* ---- convenience mutators used by the shell and the pages ---- */

export function setBenchmark(symbol) {
  try { localStorage.setItem("pt.benchmark", JSON.stringify(symbol)); } catch (_) { /* private mode */ }
  app.patch({ benchmark: symbol });
}

export function setLookback(days) {
  try { localStorage.setItem("pt.lookback", JSON.stringify(days)); } catch (_) { /* private mode */ }
  app.patch({ lookback: days });
}

export function setSidebarCollapsed(collapsed) {
  try { localStorage.setItem("pt.sidebar", JSON.stringify(!!collapsed)); } catch (_) { /* private mode */ }
  app.patch({ sidebarCollapsed: !!collapsed });
}

/** Open the right-hand drawer with arbitrary content.
    body may be a vnode or a function returning one (called on each render). */
export function openDrawer(drawer) {
  app.patch({ drawer: { width: "default", ...drawer } });
}
export function closeDrawer() {
  const d = app.get().drawer;
  if (d && typeof d.onClose === "function") d.onClose();
  app.patch({ drawer: null });
}

export function openModal(modal) { app.patch({ modal }); }
export function closeModal() { app.patch({ modal: null }); }

export function openPalette() { app.patch({ paletteOpen: true }); }
export function closePalette() { app.patch({ paletteOpen: false }); }
