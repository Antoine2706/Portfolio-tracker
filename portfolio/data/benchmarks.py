"""Benchmark options for beta.

A beta without a named benchmark is meaningless, so the benchmark is explicit,
configurable, and displayed beside the number rather than assumed.

All three were validated against live yfinance on 2026-09-03: each resolves
cleanly with roughly two years of history and is EUR-quoted, which matters --
a benchmark quoted in another currency would embed FX movement in the beta,
measuring the portfolio against the index plus a currency, not the index.
"""

from __future__ import annotations

import dataclasses

__all__ = ["Benchmark", "BENCHMARKS", "DEFAULT_BENCHMARK", "benchmark_by_symbol"]


@dataclasses.dataclass(frozen=True)
class Benchmark:
    symbol: str
    name: str
    index: str
    currency: str
    observed_days: int
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.index} ({self.symbol})"


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark(
        "MEUD.PA", "Amundi Stoxx Europe 600 UCITS ETF Acc", "STOXX Europe 600",
        "EUR", 511,
        note="Default. These holdings are overwhelmingly European thematic "
             "equity, so a broad European index is the honest comparison. "
             "EUR-quoted, so no FX contamination, and the longest history of "
             "the three."),
    Benchmark(
        "IWDA.AS", "iShares Core MSCI World UCITS ETF Acc", "MSCI World",
        "EUR", 510,
        note="The right alternative if the book ever becomes globally weighted."),
    Benchmark(
        "SMEA.MI", "iShares Core MSCI Europe UCITS ETF Acc", "MSCI Europe",
        "EUR", 503,
        note="A second European reading, on a different index construction."),
)

DEFAULT_BENCHMARK = BENCHMARKS[0]


def benchmark_by_symbol(symbol: str) -> Benchmark | None:
    return next((b for b in BENCHMARKS if b.symbol == symbol), None)
