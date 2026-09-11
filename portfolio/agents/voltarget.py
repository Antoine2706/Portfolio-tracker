"""Volatility targeting, the de-risking half only.

Pre-registered in `docs/PREREGISTRATION-volatility-targeting.md` before any
of this was written, and the parameters there are the defaults here. The
policy scales the book by a factor chosen so that its forecast volatility
equals a target, and never scales it up:

    k_t = min(1, sigma_target / sigma_hat_t)

`sigma_hat_t` is the annualised volatility of the fully invested book over
the trailing window, on the same sample covariance the equal risk
contribution policy uses, decided on data strictly before the bar it acts on.
The residual `1 - k_t` sits in cash at zero interest.

The `min(1, .)` is the whole point. Volatility targeting in the literature is
symmetric: lever up when realised volatility is below target, down when
above. This account cannot lever, so only the de-risking half is available,
and what is being tested is half of a policy with the compensating half
removed. The pre-registration expects it to fail, and says how.

Why "the book" is not quite the whole book
------------------------------------------
One holding sits at a different broker and cannot be traded against the
rest (see `agents.risk.EqualRiskContribution` and the referee). The scalar
cannot touch it, so what is scaled is the tradeable part, and the target
applies to the whole book including the part that is not scaled. That turns
the one-line formula into a quadratic in `k`, written out in `scale_factor`:
with `a` the fully invested tradeable weights, `f` the frozen ones and
`Sigma` the covariance,

    (k a + f)' Sigma (k a + f) = sigma_target^2

    A k^2 + 2 B k + (C - V) = 0,   A = a'Sigma a,  B = a'Sigma f,
                                    C = f'Sigma f,  V = sigma_target^2

and `k` is the positive root, capped at 1. With nothing frozen, `B = C = 0`
and the root is `sigma_target / sigma_hat`, the formula above. When the
frozen part alone is above the target (`C >= V`) the most the policy can do
is `k = 0`, and it says so.

What "fully invested" means after the policy has already de-risked: the
tradeable weights actually held sum to less than their share of the book,
the rest being the cash the previous decision left. The fully invested
composition is those weights at their held proportions, filling everything
the frozen part does not, so the policy scales a mix it never re-weights.
Among the tradeable lines it is buy-and-hold; the only trade it ever makes
is the scaling.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from .base import MarketView, Proposal

__all__ = ["VolatilityTarget", "ScaleFactor", "scale_factor",
           "TARGET_VOLATILITY", "TRADING_DAYS_PER_YEAR"]

# Fixed by the pre-registration. Changing it is a new trial.
TARGET_VOLATILITY = 0.15
TRADING_DAYS_PER_YEAR = 252


@dataclasses.dataclass(frozen=True)
class ScaleFactor:
    """The factor and the numbers it came from, so the reason can quote them."""
    k: float                              # applied to the tradeable part, in [0, 1]
    unconstrained: float                  # the root before the cap at 1
    forecast: float                       # annualised, of the fully invested book
    achieved: float                       # annualised, of the proposed book
    frozen_alone: float                   # annualised, of the frozen part on its own
    target: float

    @property
    def capped(self) -> bool:
        return self.unconstrained > 1.0

    @property
    def binding(self) -> bool:
        return self.k < 1.0


def scale_factor(cov: pd.DataFrame, invested: dict[str, float],
                 frozen: dict[str, float], *, target: float = TARGET_VOLATILITY,
                 periods_per_year: int = TRADING_DAYS_PER_YEAR) -> ScaleFactor:
    """Solve for the scale that puts the book's forecast volatility on target.

    `cov` is the covariance of DAILY returns, as the view supplies it;
    `target` is annualised. `invested` is the fully invested tradeable part
    and `frozen` the part the scalar may not touch; both are weights of the
    whole book and together they sum to one.

    Without a frozen part the answer is the one-line formula, which is the
    check that the quadratic is the right quadratic:

    >>> import numpy as np, pandas as pd
    >>> cov = pd.DataFrame(np.diag([0.0004, 0.0001]), index=["A", "B"],
    ...                    columns=["A", "B"])          # 2% and 1% a day
    >>> s = scale_factor(cov, {"A": 0.5, "B": 0.5}, {}, target=0.15)
    >>> sigma = math.sqrt(0.25 * 0.0004 + 0.25 * 0.0001) * math.sqrt(252)
    >>> round(s.forecast, 4) == round(sigma, 4), round(s.k, 4) == round(0.15 / sigma, 4)
    (True, True)
    >>> round(s.achieved, 4)
    0.15

    Calm markets: the root exceeds one and the cap binds. The book is fully
    invested and the policy does not lever:

    >>> s = scale_factor(cov, {"A": 0.5, "B": 0.5}, {}, target=0.40)
    >>> s.k, s.capped, round(s.unconstrained, 2)
    (1.0, True, 2.25)

    With a frozen part the achieved volatility is still the target, and the
    frozen weight is what it was:

    >>> cov = pd.DataFrame(np.diag([0.0004, 0.0001, 0.0002]),
    ...                    index=list("ABG"), columns=list("ABG"))
    >>> s = scale_factor(cov, {"A": 0.4, "B": 0.4}, {"G": 0.2}, target=0.10)
    >>> round(s.achieved, 4), s.binding, round(s.k, 3)
    (0.1, True, 0.629)

    A frozen part that is on its own above the target leaves nothing for the
    scalar to do but go to zero:

    >>> s = scale_factor(cov, {"A": 0.4, "B": 0.4}, {"G": 0.2}, target=0.04)
    >>> s.k, round(s.frozen_alone, 3), s.frozen_alone > 0.04
    (0.0, 0.045, True)
    """
    if target <= 0:
        raise ValueError(f"the target volatility must be positive, got {target}")
    columns = [str(c) for c in cov.columns]
    a = np.array([float(invested.get(c, 0.0)) for c in columns])
    f = np.array([float(frozen.get(c, 0.0)) for c in columns])
    sigma = cov.to_numpy(dtype=float)
    scale = float(periods_per_year)
    A = float(a @ sigma @ a) * scale
    B = float(a @ sigma @ f) * scale
    C = float(f @ sigma @ f) * scale
    V = target * target
    forecast = math.sqrt(max(A + 2.0 * B + C, 0.0))
    frozen_alone = math.sqrt(max(C, 0.0))

    if A <= 0.0:
        # Nothing tradeable carries any variance: there is nothing to scale.
        unconstrained = 1.0
    elif C >= V:
        unconstrained = 0.0
    else:
        # The positive root of A k^2 + 2 B k + (C - V) = 0. The discriminant
        # is B^2 - A (C - V) > 0 because C < V and A > 0.
        unconstrained = (-B + math.sqrt(B * B - A * (C - V))) / A
    k = min(1.0, max(0.0, unconstrained))
    achieved = math.sqrt(max(A * k * k + 2.0 * B * k + C, 0.0))
    return ScaleFactor(k=k, unconstrained=unconstrained, forecast=forecast,
                       achieved=achieved, frozen_alone=frozen_alone,
                       target=target)


@dataclasses.dataclass
class VolatilityTarget:
    """Scale the tradeable book so its forecast volatility is on target, and
    never above the fully invested book.

    `fixed` names the holdings the scalar may not touch, passed in rather
    than discovered, for the reason `EqualRiskContribution` gives: whether a
    holding is tradeable is a fact about the account, not the market.
    """
    target: float = TARGET_VOLATILITY
    lookback: int = 252
    fixed: "dict[str, float] | None" = None
    name: str = "volatility-target"
    periods_per_year: int = TRADING_DAYS_PER_YEAR

    def observe(self, view: MarketView) -> Proposal:
        available = list(view.available)
        held = {k: float(v) for k, v in view.held.items()}
        if not available:
            return Proposal(dict(held), 0.0,
                            "no instrument has enough history to model; holding",
                            self.name)

        pinned = {k: held.get(k, 0.0) for k in (self.fixed or {})
                  if k in available}
        tradeable = [k for k in available if k not in pinned]
        held_tradeable = {k: held.get(k, 0.0) for k in tradeable}
        scaled_total = sum(held_tradeable.values())
        room = max(0.0, 1.0 - sum(pinned.values()))
        if scaled_total <= 0.0 or room <= 0.0:
            return Proposal(dict(held), 0.0,
                            "nothing tradeable is held, so there is nothing "
                            "to scale; holding", self.name)
        # The fully invested composition: the tradeable mix at its held
        # proportions, filling everything the frozen part does not.
        invested = {k: v / scaled_total * room for k, v in held_tradeable.items()}

        cov = view.covariance(self.lookback, available)
        s = scale_factor(cov, invested, pinned, target=self.target,
                         periods_per_year=self.periods_per_year)
        weights = {k: float(v * s.k) for k, v in invested.items()}
        weights.update({k: float(v) for k, v in pinned.items()})
        return Proposal(weights, 0.5, self._reason(s, pinned, room), self.name)

    def _reason(self, s: ScaleFactor, pinned: dict, room: float) -> str:
        head = (f"volatility target {s.target:.1%}: the fully invested book "
                f"forecasts {s.forecast:.1%} annualised over {self.lookback} "
                f"days")
        if s.k <= 0.0:
            body = (f", and the frozen part alone forecasts "
                    f"{s.frozen_alone:.1%}, above the target; the tradeable "
                    f"{room:.0%} goes entirely to cash, which is the most "
                    f"this policy can do")
        elif s.capped:
            body = (f", under the target; fully invested, because the "
                    f"unconstrained factor would be {s.unconstrained:.2f} "
                    f"and this policy cannot lever")
        else:
            body = (f", so the tradeable {room:.0%} is scaled by "
                    f"{s.k:.2f} to a forecast {s.achieved:.1%}, leaving "
                    f"{room * (1.0 - s.k):.1%} of the book in cash")
        if pinned:
            names = ", ".join(f"{k} at {v:.1%}" for k, v in sorted(pinned.items()))
            body += f"; outside the scaling: {names}"
        return head + body
