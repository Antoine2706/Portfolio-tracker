# Portfolio tracker

A personal investment portfolio tracker for European-listed ETFs and ETCs,
keyed on ISIN, with risk analytics you can read against a textbook.

The number no broker app shows you is the gap between a holding's share of
your money and its share of your risk. That gap is the headline of this tool;
everything else — performance attribution, value at risk, correlation
clusters, a what-if simulator, rebalancing — is supporting evidence for it.

![Overview](docs/screenshots/overview.png)

## Run it

```bash
pip install -e ".[app,data]"
portfolio serve                       # opens http://127.0.0.1:8765
```

It starts in **demo mode** on ten real reference instruments and a synthetic
ledger, clearly labelled on every page. Switch to your own data from the
settings page or with `portfolio serve --mode user`; your instruments and
ledger live in `portfolio/data_store/user/` (gitignored) as plain CSV that
outlives the tool.

No network? `portfolio serve --provider fixture` runs the full application on
deterministic synthetic prices.

## What it does

| Page | What you get |
|---|---|
| **Overview** | Portfolio value, day change, unrealised and realised P&L, value history against a benchmark, allocation, the risk headline, and an alert feed (stale prices, concentration, correlated holdings). |
| **Holdings** | Dense sortable table with sparklines, weight, risk share, day change and provenance for every price. Click a row for the holding's price chart with your trades marked, drawdown, beta and correlation to the rest of the book. |
| **Performance** | Time-weighted and money-weighted (XIRR) returns, benchmark comparison, drawdown, monthly and calendar-year returns, contribution by holding, rolling volatility and beta, Sharpe / Sortino / Calmar. |
| **Risk** | Capital share against risk share (component contribution to volatility), effective number of holdings, diversification ratio, beta against a named benchmark, value at risk (historical, parametric, Cornish-Fisher) and expected shortfall, worst periods, stress scenarios, the correlation grid, and correlation **clusters** — the group a pairwise grid cannot see. |
| **Simulator** | Add or remove money from any holding or watchlist instrument, set target weights, or apply equal-weight / risk-parity / minimum-variance presets, and see volatility, effective holdings and every risk share move before you trade. Produces the trade list. |
| **Instruments** | Add by ISIN: resolved via OpenFIGI, every listing probed, US ticker collisions refused outright, thin and stale series flagged. Nothing is saved until you confirm the listing you saw. |
| **Transactions** | Append-only ledger. Corrections are voids, never edits. CSV import with column detection (European decimal commas and semicolons included) and duplicate detection. CSV and Excel export. |

Prices are delayed and are never presented as live. A holding that cannot be
valued is named and excluded from the total rather than valued at zero. A
missing FX rate raises rather than defaulting to parity.

| Risk | Simulator (risk parity) |
|---|---|
| ![Risk](docs/screenshots/risk.png) | ![Simulator](docs/screenshots/simulator.png) |
| **Performance** | **Holding detail** |
| ![Performance](docs/screenshots/performance.png) | ![Holding detail](docs/screenshots/holding-detail.png) |

Every chart has a table twin (the grid icon in its corner), the whole
application is reachable from the keyboard (`?` lists the shortcuts, `Ctrl K`
opens the command palette), and dark and light themes are designed
separately rather than inverted.

## Architecture

```
portfolio/
  core/    pure numpy + pandas: ledger, positions, returns, risk, performance,
           VaR, simulator. No network, no UI, importable and tested alone.
  data/    CSV store, SQLite price cache, yfinance + OpenFIGI providers,
           daily FX, CSV import/export, deterministic offline provider.
  api/     FastAPI: one cached snapshot per (mode, benchmark, lookback),
           invalidated by ledger writes or an explicit refresh. Projects only.
  web/     Zero-build single-page client: Preact + htm + ECharts, vendored.
```

The layering is enforced by tests, not by convention: `core/` cannot import a
web framework or a network client, and `api/` cannot import numpy. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the contract between layers.

Performance: every network call goes through a thread pool and an on-disk
cache; history is fetched incrementally and quotes expire after 15 minutes.
Nothing polls. After the first load, a page is a dictionary lookup.

## Tests

```bash
pip install -e ".[app,data,test]"
pytest                    # everything, including the API against the offline provider
pip install -e ".[test]" && pytest     # core only: proves core needs nothing else
```

## Data providers

yfinance for prices, OpenFIGI for identity. Both are free and keyless; an
`OPENFIGI_API_KEY` raises the identity rate limit. The provider spike that
chose them, with its evidence, is in `spike/`.
