/* Hash router.
   Routes (docs/ARCHITECTURE.md, web contract):
     #/overview (default) · #/holdings · #/holdings/:isin (drawer over holdings)
     #/performance · #/risk · #/simulator · #/allocate · #/instruments · #/transactions
     #/settings
   plus #/kitchensink (component gallery for developers). */

import { useStore, app } from "/static/lib/store.js";

export const ROUTES = [
  { page: "overview",     path: "/overview",     label: "Overview",     key: "o" },
  { page: "holdings",     path: "/holdings",     label: "Holdings",     key: "h" },
  { page: "performance",  path: "/performance",  label: "Performance",  key: "p" },
  { page: "risk",         path: "/risk",         label: "Risk",         key: "r" },
  { page: "simulator",    path: "/simulator",    label: "Simulator",    key: "s" },
  { page: "allocate",     path: "/allocate",     label: "Allocate",     key: "a" },
  { page: "instruments",  path: "/instruments",  label: "Instruments",  key: "i" },
  { page: "transactions", path: "/transactions", label: "Transactions", key: "t" },
  { page: "settings",     path: "/settings",     label: "Settings",     key: "," },
];

const PAGES = new Set([...ROUTES.map((r) => r.page), "kitchensink"]);

/** parseRoute("#/holdings/IE0002Y8CX98?tab=trades") → { page, params: { isin, tab }, path } */
export function parseRoute(hash) {
  let h = (hash || "").replace(/^#/, "");
  if (!h.startsWith("/")) h = "/" + h;
  const [pathPart, queryPart] = h.split("?");
  const segments = pathPart.split("/").filter(Boolean);
  const page = segments[0] || "overview";
  const params = {};
  if (queryPart) {
    for (const [k, v] of new URLSearchParams(queryPart)) params[k] = v;
  }
  if (!PAGES.has(page)) return { page: "notfound", params: { path: pathPart }, path: pathPart };
  if (page === "holdings" && segments[1]) params.isin = decodeURIComponent(segments[1]).toUpperCase();
  if (page === "instruments" && segments[1]) params.isin = decodeURIComponent(segments[1]).toUpperCase();
  if (page === "transactions" && segments[1]) params.id = decodeURIComponent(segments[1]);
  return { page, params, path: pathPart };
}

/** navigate("/holdings/IE0002Y8CX98") or navigate("#/risk") */
export function navigate(path, { replace = false } = {}) {
  let p = String(path || "/overview");
  if (p.startsWith("#")) p = p.slice(1);
  if (!p.startsWith("/")) p = "/" + p;
  const target = "#" + p;
  if (location.hash === target) return;
  if (replace) {
    const url = location.pathname + location.search + target;
    history.replaceState(null, "", url);
    syncRoute();
  } else {
    location.hash = target;
  }
}

export function currentRoute() {
  return parseRoute(location.hash);
}

function syncRoute() {
  const route = currentRoute();
  const prev = app.get().route;
  if (prev && prev.page === route.page && prev.path === route.path
      && JSON.stringify(prev.params) === JSON.stringify(route.params)) return;
  app.patch({ route });
}

let started = false;
/** Install the hashchange listener once and sync the initial route. */
export function startRouter() {
  if (started) return;
  started = true;
  if (!location.hash) history.replaceState(null, "", location.pathname + location.search + "#/overview");
  window.addEventListener("hashchange", syncRoute);
  syncRoute();
}

/** Hook: the current route { page, params, path }. */
export function useRoute() {
  return useStore((s) => s.route);
}

/** Href for a route path — used by nav links so middle-click works. */
export function href(path) {
  let p = String(path || "/overview");
  if (p.startsWith("#")) p = p.slice(1);
  if (!p.startsWith("/")) p = "/" + p;
  return "#" + p;
}

export function routeFor(page) {
  return ROUTES.find((r) => r.page === page) || null;
}
