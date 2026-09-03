# Phase 0: provider validation spike

Answers one question before any application code is written: **which market
data provider can serve European-listed thematic UCITS ETFs and USD commodity
ETCs, keyed on ISIN**, with a current quote and two years of daily history?

Everything downstream depends on the answer, so it is settled with evidence.

## Run it

```bash
pip install -r spike/requirements.txt

# Optional. A provider without a key is skipped, not failed.
export TWELVEDATA_API_KEY=...
export FMP_API_KEY=...
export EODHD_API_KEY=...

python spike/check_providers.py
```

Useful flags:

```bash
python spike/check_providers.py --dry-run                    # cost it first, fetch nothing
python spike/check_providers.py --providers yfinance         # one provider
python spike/check_providers.py --checks coverage,collision  # subset of checks
python spike/check_providers.py --isins IE0002Y8CX98         # one instrument
python spike/check_providers.py --alias-scope all            # identity-check every fund
python spike/check_providers.py --max-fallbacks 2            # cheaper, less thorough
python spike/check_providers.py --offline                    # replay, spends no quota
```

Raw responses land in `spike/results/run-<timestamp>.json` (gitignored). Free
tiers have daily caps — read that file rather than re-running.

## Quota: run `--dry-run` first

Two of these free tiers are tight enough that one careless run costs you the
day. `--dry-run` prints the cost and fetches nothing:

```
[cost] yfinance:   41-83  calls, budget 300/day
[cost] twelvedata: 51-135 calls, budget 800/day
[cost] fmp:        61-187 calls, budget 250/day
[cost] eodhd:      51-135 calls
```

**FMP is the binding constraint** at 250 requests/day and three calls per fetch
(quote, history, profile-for-ISIN). Raising `--max-fallbacks` to 6 takes the
worst case to 247 of 250; adding `--alias-scope all` takes it to 258 and the
run warns you before it starts. The default of 4 fallbacks keeps a full
four-check run inside every provider's cap.

**Twelve Data is the slow one**, not the expensive one: 8 credits/minute means
a 135-call run takes about 18 minutes of wall clock. The run prints its own
worst-case duration up front.

Three guards, because a half-finished matrix is worse than a cheap one:

- Every call is counted; a provider stops at its budget rather than failing
  mid-instrument with an unexplained blank.
- Instruments not reached are marked **`skip`**, kept distinct from `FAIL`.
  "The provider cannot serve this" and "we ran out of quota before asking"
  point to opposite conclusions about whether to pay for it.
- A provider that raises is caught per-instrument. One misbehaving provider
  costs you its own column, not the whole matrix.

## Two files

- `spike/instruments.py` — the verified instrument reference. This is
  **evidence**, not code: ten instruments keyed by ISIN, their listings by
  venue, the two liquidated ISINs, and the five known US ticker collisions.
  Kept separate so a correction to the data is a diff you can read.
- `spike/check_providers.py` — the probes and the five checks.

## Five checks, not one

Coverage alone would have missed four of the five failure modes that actually
matter here.

### 1. Coverage

Quote plus ≥400 trading days for each instrument, resolved **from its ISIN**.
Reported separately for ETFs and ETCs, with subtotals — the three WisdomTree
ETCs are collateralised notes rather than UCITS funds, and several providers
classify or omit them differently. A provider that covers the seven ETFs and
none of the ETCs is not a provider that covers this portfolio.

| Verdict | Meaning |
|---|---|
| `PASS` | Quote **and** ≥400 days, identity confirmed |
| `PART` | Something returned, not enough to build a risk model on |
| `SUSP` | Data returned from a listing that looks **wrong** |
| `FAIL` | Nothing usable |

`SUSPECT` outranks `PASS` on purpose. Data from the wrong listing is worse than
no data: it is confidently wrong and it fails silently.

### 2. Symbol identity

Every symbol of a fund must resolve to that fund's ISIN. **WDEF and EUDF are
one fund** — `IE0002Y8CX98`, the WisdomTree Europe Defence UCITS ETF, trading
as EUDF on Xetra and WDEF on LSE, Borsa Italiana, Paris and SIX. A provider
that answers those two symbols with two different instruments has exactly the
failure ISIN keying exists to prevent, and it is invisible unless tested.

The check generalises: alias groups are derived from the listing table, so
every multi-symbol fund is testable. Default scope is funds with colliding
tickers (cheap); `--alias-scope all` covers everything.

A provider that never returns an ISIN fails this check by construction, and
that is the correct result — identity it cannot verify is identity you cannot
trust.

### 3. Ticker collisions

Five of these tickers collide with real US-listed products:

| Ticker | Wanted | Collides with |
|---|---|---|
| WDEF | `IE0002Y8CX98` WisdomTree Europe Defence UCITS | `US97717Y3374` WisdomTree Europe Defense Fund |
| NATO | `IE000OJ5TQP4` HANetf Future of Defence | `US8829277677` Themes Transatlantic Defense |
| ARMY | `IE000I7E6HL0` HANetf Future of European Defence | CUSIP 87975E784 Tema International Defense |
| WEAT | `JE00BN7KB664` WisdomTree Wheat ETC | `US88166A8707` Teucrium Wheat Fund |
| GLUX | `LU1681048630` Amundi Global Luxury | Great Lakes Aviation (US equity) |

The check asks each provider for the **bare ticker** and takes the provider's
own top answer, with no re-ranking from us — it measures what a ticker-keyed
design would have handed you. The report prints the resolved symbol, exchange,
currency and ISIN for each.

**WDEF is the dangerous one** and is called out explicitly in the output: same
issuer, same theme, same ticker, different fund. The US listing would pass
every sanity check a human would apply — right name, right sector, a sensible
price series. Only the ISIN and the exchange tell them apart.

### 4. Liveness of liquidated lines

Two of these funds have liquidated predecessors still sitting in fund
databases with the same name, TER and index:

- `GB00B15KY765` — WisdomTree Wheat predecessor. justETF still shows it quoting
  under WEAT and OD7S. Live line is `JE00BN7KB664`.
- `DE000A1JS9B2` — iShares Agribusiness duplicate. Live line is `IE00B6R52143`.

**The test is inverted: returning nothing is the correct answer.** A provider
that hands back history ending within 21 days for a liquidated ISIN is serving
stale data, and would feed a dead price series into the covariance matrix
without anything visibly going wrong.

### 5. Mechanics

Rate limit, batch quote support, adjusted closes, and currency.

- **Batching** changes the whole caching design: with batch quotes you cache
  one response per refresh, without them one per instrument.
- **Adjusted closes** are checked per provider. Unadjusted prices read a
  distribution as a large negative return and inflate measured volatility.
  Twelve Data's free tier is the one to watch — `adjust` is a paid parameter.
- **Base and quote currency are reported separately**, because they differ for
  most of this set. Only four of ten are EUR-based; ARMY, NATO and ISAG are USD
  base but trade in EUR on continental venues, and all three ETCs are USD. FX
  is the majority case here, not an edge case.
- **Observed staleness** is measured from the quote timestamp where one exists,
  rather than trusting the documented delay.

## Resolution models differ, and it matters

| Provider | ISIN → symbol | Consequence |
|---|---|---|
| EODHD | `/search/{isin}` directly | Only model that fits this architecture rather than fighting it |
| FMP | `/v4/search/isin` | Right shape; coverage is the open question |
| Twelve Data | `symbol_search`, ISIN support unconfirmed | Uses MIC codes for venues, which is the one standardised venue identifier |
| yfinance | none | Resolution runs **backwards**: candidates are built from the hand-verified listing table and `Ticker.isin` only *verifies*. It can never resolve an instrument not already catalogued by hand |

That asymmetry is a finding in its own right, and it is the reason yfinance is
treated as a coverage benchmark rather than assumed to be the production pick.

---

## Running the application

```bash
pip install -e ".[app,data,test]"
streamlit run portfolio/app/main.py
```

Starts in **demo mode** on the seed data. Switch to your own data with the
sidebar radio, or `PORTFOLIO_DATA_MODE=user`.

Tests:

```bash
python -m pytest --ignore=portfolio/tests/test_app_smoke.py   # 345, no streamlit needed
python -m pytest                                              # 356, includes UI smoke
```

The UI smoke tests take ~3 minutes because yfinance retries against a network
it cannot reach; they are worth the wait, since they are what caught the FX
failure that blanked the Holdings view.
