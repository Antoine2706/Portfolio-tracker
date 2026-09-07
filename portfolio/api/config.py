"""Runtime configuration, read once from the environment.

Everything the CLI can set is an environment variable too, so
`uvicorn portfolio.api.app:create_app --factory` and `portfolio serve` behave
identically and a deployment can be configured without a wrapper script.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import pathlib

from ..data.store import DEFAULT_ROOT, DataMode, resolve_mode

__all__ = ["Config", "VERSION"]


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("portfolio-tracker")
    except Exception:                                        # pragma: no cover
        return "0.2.0"


VERSION = _version()


@dataclasses.dataclass(frozen=True)
class Config:
    mode: DataMode = DataMode.SEED
    provider: str = "yfinance"                  # "yfinance" | "fixture"
    data_root: pathlib.Path = DEFAULT_ROOT
    base_currency: str = "EUR"
    quote_ttl: dt.timedelta = dt.timedelta(minutes=15)
    history_ttl: dt.timedelta = dt.timedelta(hours=6)
    lookback: int = 252
    max_workers: int = 6
    # The in-memory resolution cache: probing costs one call per candidate,
    # and a user re-reading the confirm screen must not spend them again.
    resolution_ttl: dt.timedelta = dt.timedelta(minutes=15)

    @classmethod
    def from_env(cls) -> "Config":
        provider = os.environ.get("PORTFOLIO_PROVIDER", "yfinance").strip().lower()
        if provider not in {"yfinance", "fixture"}:
            raise ValueError(f"PORTFOLIO_PROVIDER must be yfinance or fixture, not {provider!r}")
        root = os.environ.get("PORTFOLIO_DATA_ROOT")
        return cls(
            mode=resolve_mode(None),
            provider=provider,
            data_root=pathlib.Path(root).expanduser().resolve() if root else DEFAULT_ROOT,
            lookback=int(os.environ.get("PORTFOLIO_LOOKBACK", "252")),
        )
