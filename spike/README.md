# Phase 0: provider validation spike

Answers one question before any application code is written: **which market
data provider can actually serve European-listed thematic UCITS ETFs and
commodity ETCs**, with both a current quote and two years of daily history?

Everything downstream depends on the answer, so it is settled with evidence.

## Run it

```bash
pip install -r spike/requirements.txt

# Optional. Any provider without a key is skipped, not failed.
export TWELVEDATA_API_KEY=...
export FMP_API_KEY=...
export EODHD_API_KEY=...

python spike/check_providers.py
```

Useful flags:

```bash
python spike/check_providers.py --providers yfinance          # one provider
python spike/check_providers.py --instruments WDEF,DFEU       # a few names
python spike/check_providers.py --offline                     # replay last run, no quota spent
```

Raw responses land in `spike/results/run-<timestamp>.json` (gitignored). Free
tiers have daily caps, so re-read that file rather than re-running.

## What it reports

Per provider and instrument: whether a quote returned, whether history
returned, how many trading days, first and last date, currency, exchange, the
symbol it resolved to and how it resolved it. Then a pass/fail matrix, the
documented rate limit and batching support per provider, and the resolved
`provider_symbols` mapping as JSON.

## Three verdicts, not two

| Verdict | Meaning |
|---|---|
| `PASS` | Quote **and** ≥400 trading days, from a plausibly European listing |
| `PART` | Something came back, but not enough to build a risk model on |
| `SUSP` | Data came back from a listing that looks **wrong** |
| `FAIL` | Nothing usable |

`SUSPECT` exists because of ticker collisions, and it deliberately outranks
`PASS` in the summary. `WEAT` is WisdomTree Wheat in London and Teucrium Wheat
Fund in New York. `NATO` collides too. A provider that answers a bare ticker
with the wrong fund is *worse* than one that answers nothing, because the
failure is silent and the risk numbers that follow are confidently wrong. So
every resolution is checked against an allowlist of European exchanges, and a
US listing is reported as a failure with a named reason.

This is the ISIN-is-the-primary-key constraint enforced at runtime.

## ISINs are blank on purpose

`TestInstrument.isin` is empty for every instrument. I did not invent them: a
wrong ISIN in the primary key column propagates into everything downstream and
is very hard to spot later. The ISIN-based resolution paths (EODHD's search
endpoint, FMP's `search/isin`) are already implemented and start being used the
moment the field is filled in from a broker statement.

## Provider notes

| Provider | Symbol resolution | Why it matters |
|---|---|---|
| yfinance | none — brute-force exchange suffixes | N calls per instrument, and no way to confirm we found the right listing except by inspecting what came back |
| Twelve Data | `/symbol_search` returns exchange + currency | One call, checkable |
| FMP | `/v3/search`, plus `/v4/search/isin` | ISIN path is the right shape |
| EODHD | `/search/{isin-or-ticker}` | Accepts an ISIN directly — the cleanest fit for this architecture |

Two things are checked beyond coverage:

- **Adjusted closes.** The script flags any provider whose history lacks an
  adjusted-close field. Unadjusted prices show a distribution as a price drop,
  which registers as a large negative return and inflates measured volatility.
  These ETCs distribute, so this is not a theoretical concern.
- **Observed staleness.** Where a quote carries a timestamp, the script reports
  how old it actually is, rather than trusting the documented delay.
