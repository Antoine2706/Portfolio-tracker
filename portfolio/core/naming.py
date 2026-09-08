"""Display names: a short one for labels, the legal one for the record.

A fund's registered name is built for a prospectus, not for a chart axis.
"iShares MSCI Europe Industrials Sector UCITS ETF EUR Acc" is fifty-six
characters of which perhaps twenty-three identify the fund; the rest is
wrapper (UCITS ETF), share class (Acc), and currency (EUR), repeated on every
line of a seven-holding portfolio. Truncating that to fit produces
"iShares MSCI Europe Industria...", which is the worst outcome: it spends the
space on the part that is identical across four of the seven holdings and
elides the part that distinguishes them.

So a short name is *derived*, not truncated, and then stored on the instrument
where the user can correct it. Derivation is a heuristic over fund-naming
conventions and will occasionally be wrong; that is precisely why the result is
a stored, overridable field rather than something recomputed at render time.

The rules, in the order they are applied:

1.  Drop parenthesised groups. In fund names these are almost always share
    class -- "(Acc)", "(EUR Hedged)", "(Dist)".
2.  Drop trailing wrapper, share-class, currency, hedging and domicile tokens.
    Only from the end, so "EURO STOXX Banks" keeps its EURO and only a trailing
    "EUR" is taken.
3.  Drop a trailing legal form: SE, NV, PLC, AG.
4.  Drop the issuer from the front, but only when enough remains to identify
    the instrument. This is what stops "Schneider Electric SE" collapsing to
    nothing once its legal form has gone.

Nothing here shortens by cutting words in half. If the derived name is still
over the limit it is trimmed at a word boundary, which at least leaves whole
words, and the caller is expected to show the full name in a tooltip.
"""

from __future__ import annotations

import re

__all__ = ["shorten_name", "short_names", "derive_issuer",
           "DEFAULT_SHORT_NAME_LIMIT", "ISSUER_PREFIXES"]

# Wide enough for "European Property Yield" (23) to survive intact, narrow
# enough that a horizontal bar chart's labels do not eat the plot area.
DEFAULT_SHORT_NAME_LIMIT = 24

# Removed from the END only, repeatedly. A leading or middle occurrence is
# left alone: "EURO STOXX Banks" and "Core MSCI EM IMI" both contain tokens
# that appear in this set and both need them.
_TRAILING_NOISE = {
    # wrapper
    "UCITS", "ETF", "ETC", "ETP", "ETN", "FUND", "SICAV", "ICAV", "INDEX",
    "SECTOR", "SWAP",
    # share class and payout policy
    "ACC", "ACC.", "DIST", "DIS", "DIST.", "ACCUMULATING", "DISTRIBUTING",
    "CAP", "INC.", "CLASS", "SHARES", "SHARE",
    # currency of the share class
    "EUR", "USD", "GBP", "GBX", "CHF", "JPY", "SEK", "NOK", "DKK", "CAD",
    "AUD", "HKD", "CNY", "CNH", "PLN", "CZK", "HUF",
    # hedging and domicile
    "HEDGED", "HEDGE", "HDG", "UNHEDGED", "DE", "IE", "LU", "FR", "CH",
}

# Single letters and short codes are share classes often enough that removing
# them is right on balance, but only where a name remains without them, and
# only from the end. "Xtrackers MSCI World Swap UCITS ETF 1C" needs this.
_TRAILING_CLASS_CODE = re.compile(r"^(?:[A-Z]|[0-9]{1,2}[A-Z]{1,2}|[A-Z]{1,2}[0-9]{1,2})$")

# Company legal forms. Deliberately excludes GROUP and HOLDINGS, which are
# frequently part of the name people actually use.
_LEGAL_FORMS = {
    "SE", "SA", "S.A.", "SAS", "NV", "N.V.", "BV", "B.V.", "AG", "AG.",
    "PLC", "P.L.C.", "LTD", "LTD.", "LIMITED", "INC", "INC.", "CORP",
    "CORP.", "CORPORATION", "SPA", "S.P.A.", "AB", "ASA", "OYJ", "KGAA",
    "A/S", "AS", "NA", "SCA", "SPRL", "GMBH",
}

# Matched as phrases at the start of the name, longest first, so "Global X"
# never leaves a stray "X" and "Legal & General" is taken whole. The issuer
# recorded on the instrument is tried first; this list is the fallback for
# instruments whose issuer was never filled in, and the source for the
# backfill in `derive_issuer`.
ISSUER_PREFIXES: tuple[str, ...] = (
    # Sub-brands first in spirit, though matching is by length: an issuer
    # that markets under "Amundi Index Solutions" puts three uninformative
    # words in front of every fund it runs.
    "Amundi Index Solutions", "Amundi Funds", "Amundi ETF",
    "Lyxor Index Fund", "Invesco Markets", "HSBC ETFs", "UBS ETF",
    "Xtrackers II", "L&G ETF",
    "BNP Paribas Easy", "Legal & General", "Goldman Sachs", "State Street",
    "Multi Units Luxembourg", "First Trust", "Global X", "JPMorgan",
    "WisdomTree", "iShares", "Xtrackers", "Amundi", "Lyxor", "SPDR",
    "Vanguard", "Invesco", "VanEck", "HSBC", "UBS", "Franklin", "Fidelity",
    "Deka", "ComStage", "Rize", "Tabula", "Ossiam", "DWS", "BlackRock",
    "PIMCO", "Schwab", "21Shares", "CoinShares", "Sparkasse", "L&G",
    "Han-GINS", "HANetf", "Market Access", "Expat", "Vontobel",
)

_PARENTHESISED = re.compile(r"\s*\([^)]*\)")
_WHITESPACE = re.compile(r"\s+")


def _tokens(name: str) -> list[str]:
    return [t for t in _WHITESPACE.split(_PARENTHESISED.sub("", name).strip()) if t]


def _strip_trailing_noise(tokens: list[str]) -> list[str]:
    """Remove wrapper, class, currency and domicile tokens from the end."""
    out = list(tokens)
    changed = True
    while changed and len(out) > 1:
        changed = False
        last = out[-1].upper().strip(",")
        if last in _TRAILING_NOISE:
            out.pop()
            changed = True
        elif len(out) > 2 and _TRAILING_CLASS_CODE.match(last):
            # Only with three or more tokens: a two-token name ending in a
            # short code is more likely "Gold 4X" than a share class.
            out.pop()
            changed = True
    return out


def _strip_legal_form(tokens: list[str]) -> list[str]:
    out = list(tokens)
    while len(out) > 1 and out[-1].upper().strip(",") in _LEGAL_FORMS:
        out.pop()
    return out


def _strip_issuer(text: str, issuer: str) -> str:
    """Remove the issuer from the front, unless doing so guts the name.

    The guard is the whole point. Stripping "Schneider Electric" from
    "Schneider Electric" leaves nothing, and a blank axis label is worse than
    a redundant one.
    """
    candidates = [issuer] if issuer.strip() else []
    candidates += list(ISSUER_PREFIXES)
    for prefix in sorted(candidates, key=len, reverse=True):
        prefix = prefix.strip()
        if not prefix:
            continue
        if text.upper().startswith(prefix.upper()):
            rest = text[len(prefix):].lstrip(" -,")
            # Enough must remain to identify the instrument on its own.
            if len(rest) >= 6 and len(rest.split()) >= 1:
                return rest
    return text


def _trim_to_limit(text: str, limit: int) -> str:
    """Last resort, on a word boundary. Never cuts a word in half."""
    if len(text) <= limit:
        return text
    words = text.split()
    out: list[str] = []
    for word in words:
        candidate = " ".join(out + [word])
        if len(candidate) + 1 > limit:      # +1 leaves room for the ellipsis
            break
        out.append(word)
    if not out:                             # a single word longer than the limit
        return text[:max(limit - 1, 1)] + "…"
    return " ".join(out) + "…"


def shorten_name(name: str, issuer: str = "",
                 limit: int = DEFAULT_SHORT_NAME_LIMIT,
                 strip_issuer: bool = True) -> str:
    """A label-length name derived from a fund's registered name.

    >>> shorten_name("iShares MSCI Europe Industrials Sector UCITS ETF EUR Acc")
    'MSCI Europe Industrials'
    >>> shorten_name("iShares European Property Yield UCITS ETF EUR Acc")
    'European Property Yield'
    >>> shorten_name("iShares Core MSCI EM IMI UCITS ETF USD Acc")
    'Core MSCI EM IMI'
    >>> shorten_name("iShares EURO STOXX Banks 30-15 UCITS ETF DE EUR Acc")
    'EURO STOXX Banks 30-15'
    >>> shorten_name("VanEck Semiconductor UCITS ETF")
    'Semiconductor'
    >>> shorten_name("Invesco Physical Gold ETC")
    'Physical Gold'

    An equity keeps its identity: the legal form goes, the name does not, even
    though the issuer recorded against it is the company itself.

    >>> shorten_name("Schneider Electric SE", issuer="Schneider Electric")
    'Schneider Electric'
    >>> shorten_name("ASML Holding NV")
    'ASML Holding'

    Parenthesised share classes and hedging go with the rest of the wrapper.

    >>> shorten_name("Xtrackers MSCI World Swap UCITS ETF 1C")
    'MSCI World'
    >>> shorten_name("iShares Core MSCI World UCITS ETF USD (Acc)")
    'Core MSCI World'

    What is left over the limit is trimmed between words, never through one.

    >>> shorten_name("Amundi Index Solutions Global Aggregate Green Bond")
    'Global Aggregate Green…'

    An empty or whitespace name gives an empty result rather than raising: the
    caller has an ISIN to fall back on and a chart should not fail to draw.

    >>> shorten_name("   ")
    ''

    `strip_issuer=False` keeps the provider on the front. Two funds tracking
    the same thing from different houses shorten to the same string once the
    issuer is gone, and where both are held that is the one piece of the name
    that distinguishes them — see `short_names`.

    >>> shorten_name("iShares Europe Defence UCITS ETF", strip_issuer=False)
    'iShares Europe Defence'
    """
    tokens = _tokens(name)
    if not tokens:
        return ""
    tokens = _strip_trailing_noise(tokens)
    tokens = _strip_legal_form(tokens)
    text = " ".join(tokens)
    if strip_issuer:
        text = _strip_issuer(text, issuer)
    return _trim_to_limit(text.strip(), limit)


def short_names(instruments, limit: int = DEFAULT_SHORT_NAME_LIMIT) -> dict[str, str]:
    """Short names for a whole universe, kept distinct from one another.

    A short name is only worth having if it still identifies the instrument.
    Derived one at a time, "WisdomTree Europe Defence" and "iShares Europe
    Defence" both come out as "Europe Defence", and an alert reading "Europe
    Defence and Europe Defence move together" is worse than either registered
    name — it does not merely fail to help, it actively misinforms.

    So the set is derived together. Where two labels collide the issuer goes
    back on, since that is exactly the distinguishing part the general rule
    threw away; where that is still not enough, the ISIN tail decides. A name
    the user pinned by hand is always left alone, collision or not: it is
    their label and they can see what it collides with.

    >>> from .models import Instrument, AssetClass
    >>> book = {
    ...   "IE0002Y8CX98": Instrument("IE0002Y8CX98", "WisdomTree Europe Defence UCITS ETF"),
    ...   "IE000IAXNM41": Instrument("IE000IAXNM41", "iShares Europe Defence UCITS ETF"),
    ...   "IE00BMC38736": Instrument("IE00BMC38736", "VanEck Semiconductor UCITS ETF"),
    ... }
    >>> names = short_names(book)
    >>> names["IE00BMC38736"]
    'Semiconductor'
    >>> names["IE0002Y8CX98"], names["IE000IAXNM41"]
    ('WisdomTree Europe Defence', 'iShares Europe Defence')

    A universe with no collisions is unaffected, so the common case pays
    nothing for this:

    >>> short_names({"IE00BMC38736": book["IE00BMC38736"]})
    {'IE00BMC38736': 'Semiconductor'}
    """
    items = (instruments.items() if hasattr(instruments, "items")
             else [(i.isin, i) for i in instruments])
    pinned: dict[str, str] = {}
    derived: dict[str, str] = {}
    for isin, inst in items:
        if inst.short_name.strip():
            pinned[isin] = inst.short_name.strip()
        else:
            derived[isin] = shorten_name(inst.name, inst.issuer, limit)

    universe = dict(instruments.items()) if hasattr(instruments, "items") else {
        i.isin: i for i in instruments}

    def collisions(mapping: dict[str, str]) -> set[str]:
        counts: dict[str, int] = {}
        for label in list(mapping.values()) + list(pinned.values()):
            counts[label] = counts.get(label, 0) + 1
        return {label for label, n in counts.items() if n > 1}

    clashing = collisions(derived)
    if clashing:
        # Round two: put the issuer back on, and give the label room for it.
        for isin, label in list(derived.items()):
            if label in clashing:
                inst = universe[isin]
                derived[isin] = shorten_name(
                    inst.name, inst.issuer,
                    limit + len(inst.issuer) + 1, strip_issuer=False)
        clashing = collisions(derived)
    if clashing:
        # Round three: nothing in the name separates them, so the key does.
        for isin, label in list(derived.items()):
            if label in clashing:
                derived[isin] = f"{label} · {isin[-4:]}"

    return {**derived, **pinned}


def derive_issuer(name: str) -> str:
    """The issuer implied by a fund's name, or "" when none is recognised.

    Used to backfill instruments whose issuer was never recorded. An issuer
    breakdown showing "Unknown 12%" is not a portfolio insight, it is a data
    quality report, and this removes most of the cases that produce it.

    >>> derive_issuer("iShares Core MSCI EM IMI UCITS ETF USD Acc")
    'iShares'
    >>> derive_issuer("VanEck Semiconductor UCITS ETF")
    'VanEck'
    >>> derive_issuer("Global X Copper Miners UCITS ETF")
    'Global X'
    >>> derive_issuer("Schneider Electric SE")
    ''

    Matching is longest-first, so an issuer whose name begins with another
    issuer's is still resolved correctly.

    >>> derive_issuer("Legal & General Cyber Security UCITS ETF")
    'Legal & General'
    """
    text = _PARENTHESISED.sub("", name or "").strip()
    for prefix in sorted(ISSUER_PREFIXES, key=len, reverse=True):
        if text.upper().startswith(prefix.upper()):
            return prefix
    return ""
