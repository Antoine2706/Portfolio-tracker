# Seed data

The ten researched reference instruments and a **synthetic** transaction
ledger. Committed to git on purpose: this is what the test suite runs against
and what a public demo shows.

**These are not anyone's real positions.** The instruments are real and the
reference data is verified; the transactions are invented to exercise the
model — a partial sell, a multi-currency purchase, a dividend, a standalone
fee, a voided-and-corrected entry, and two instruments held zero times so the
watchlist path has coverage.

Real data lives in `data_store/user/`, which is gitignored. The active mode is
chosen by `PORTFOLIO_DATA_MODE=seed|user` and is always displayed in the UI —
a demo that silently reads real data, and a real session that silently reads
demo data, are both failures worth preventing loudly.
