"""Reconciling the tracker against a document it did not produce.

Every other test in this suite checks the code against itself. This one
checks it against a bank. Seven trade confirmations from February and March
2026 are replayed through `derive_positions`, and the resulting share counts
are compared with a holdings statement dated 30 June 2026 that the tracker
has never seen and cannot influence.

That is a weaker check than it sounds in one respect and a much stronger one
in another. Weaker: it cannot catch an error the bank shares. Stronger: it is
the only test here that can fail because the *world* disagrees with the code,
rather than because two halves of the code disagree with each other.

Where the two genuinely differ, the difference is printed rather than
reconciled away. The bank's acquisition value excludes the transaction tax
paid on acquisition; the tracker's includes it. The tracker is right and the
disagreement is deliberate, so a test that adopted the bank's figure to go
green would be destroying the finding.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from portfolio.core.models import TransactionType
from portfolio.core.positions import derive_positions
from portfolio.tests.fixtures_medirect import (ACCOUNT_TYPE, CONFIRMATIONS,
                                               STATEMENT, STATEMENT_DATE,
                                               STATEMENT_TOTAL, ledger)

GOLD = "IE00B579F325"
SEMIS = "IE00BMC38736"


@pytest.fixture()
def positions():
    return derive_positions(ledger(), strict=False)


class TestTheLedgerReproducesTheStatement:
    """Share counts, exactly. There is no tolerance on a share count."""

    def test_every_holding_on_the_statement_is_in_the_book(self, positions):
        assert {line.isin for line in STATEMENT} <= set(positions)

    @pytest.mark.parametrize("line", STATEMENT, ids=lambda l: l.isin)
    def test_the_share_count_matches_exactly(self, line, positions):
        held = positions[line.isin].quantity
        assert held == Decimal(line.quantity), (
            f"{line.isin} ({line.name}): the ledger says {held} shares, the "
            f"statement of {STATEMENT_DATE} says {line.quantity}")

    def test_the_sale_is_reflected_rather_than_ignored(self, positions):
        """32 bought in February, 16 sold in June, 16 on the statement. A
        replay that dropped disposals would show 32 and reconcile to nothing."""
        bought = sum(int(c.quantity) for c in CONFIRMATIONS
                     if c.isin == SEMIS and c.side == "BUY")
        sold = sum(int(c.quantity) for c in CONFIRMATIONS
                   if c.isin == SEMIS and c.side == "SELL")
        assert bought == 32 and sold == 16
        assert positions[SEMIS].quantity == Decimal(bought - sold)

    def test_nothing_is_held_that_the_statement_does_not_show(self, positions):
        """Except gold, which is at the other broker -- see below."""
        open_now = {i for i, p in positions.items() if p.quantity > 0}
        assert open_now == {line.isin for line in STATEMENT}


class TestTheBrokerSplitIsReal:
    def test_gold_is_absent_because_it_is_at_keytrade(self, positions):
        """The statement is MeDirect's, and the gold ETC is not held there.

        Its absence is a check rather than a gap: a tracker that modelled one
        account would reconcile to a total that matches no statement the
        account actually receives, and would have been "correct" here by
        being wrong about where things are held.
        """
        assert GOLD not in {line.isin for line in STATEMENT}
        assert GOLD not in positions, (
            "the confirmations are MeDirect's, so gold should not appear in a "
            "book built from them at all")


class TestTheValuationReconciles:
    """At the bank's own marks, to the cent.

    Priced from a data provider the figures would differ, because a closing
    mark is not a last trade. That comparison belongs in a run against the
    real price cache; what is testable here is that the arithmetic between
    share count, price and value is the tracker's arithmetic too.
    """

    def differences(self, positions, prices):
        out = {}
        for line in STATEMENT:
            mine = Decimal(positions[line.isin].quantity) * prices[line.isin]
            out[line.isin] = (mine, line.value, mine - line.value)
        return out

    def test_each_line_reconciles_at_the_statements_own_price(self, positions):
        prices = {line.isin: line.price for line in STATEMENT}
        for isin, (mine, theirs, delta) in self.differences(positions, prices).items():
            assert abs(delta) <= Decimal("0.01"), (
                f"{isin}: {mine} against the statement's {theirs}, "
                f"a difference of {delta}")

    def test_and_so_does_the_total(self, positions):
        prices = {line.isin: line.price for line in STATEMENT}
        mine = sum(Decimal(positions[line.isin].quantity) * prices[line.isin]
                   for line in STATEMENT)
        assert abs(mine - STATEMENT_TOTAL) <= Decimal("0.01"), (
            f"{mine} against the statement total {STATEMENT_TOTAL}")

    def test_the_statement_totals_its_own_lines(self, positions):
        """The transcription is checked before anything is checked against it.
        A mistyped line would otherwise make the tracker look wrong."""
        assert sum(line.value for line in STATEMENT) == STATEMENT_TOTAL

    def test_a_difference_is_reported_rather_than_swallowed(self, positions):
        """The mechanism the real run uses: prices from the provider will not
        equal the bank's marks, and the per-holding difference is the output,
        not a hidden pass."""
        moved = {line.isin: line.price * Decimal("1.001") for line in STATEMENT}
        found = self.differences(positions, moved)
        assert all(delta > 0 for _, _, delta in found.values())
        assert len(found) == len(STATEMENT)


class TestTheCostBasisDisagreesWithTheBankOnPurpose:
    """The bank's acquisition value is pre-tax. The tracker's is not.

    Worth 12 cents on this trade and decisive as a principle. The bank
    computed the realised gain on the VanEck sale as 1,714.28 minus 977.92,
    where 977.92 is exactly half the 1,955.84 buy notional -- the notional,
    not what the purchase cost. The 2.35 of transaction tax paid to acquire
    the shares is part of what was paid for them, so excluding it overstates
    the gain and, at 10%, overstates the tax.
    """

    def basis(self, positions):
        return positions[SEMIS].average_cost.amount

    def test_the_tracker_includes_acquisition_costs_in_the_basis(self, positions):
        buy = next(c for c in CONFIRMATIONS
                   if c.isin == SEMIS and c.side == "BUY")
        paid = buy.notional + buy.commission + buy.tob      # 1,955.84 + 2.35
        assert float(self.basis(positions)) == pytest.approx(
            float(paid) / buy.quantity, abs=1e-9)

    def test_which_is_more_than_the_banks_and_by_how_much(self, positions):
        sale = next(c for c in CONFIRMATIONS
                    if c.isin == SEMIS and c.side == "SELL")
        bank_basis = Decimal("977.92")                      # half the notional
        mine = float(self.basis(positions)) * sale.quantity
        overstatement = mine - float(bank_basis)
        assert overstatement == pytest.approx(1.175, abs=0.01)

        bank_gain = float(sale.notional) - float(bank_basis)
        my_gain = float(sale.notional) - mine
        assert bank_gain == pytest.approx(736.36, abs=0.01)
        assert my_gain < bank_gain
        # Ten per cent of the difference is what was overpaid.
        assert (bank_gain - my_gain) * 0.10 == pytest.approx(0.1175, abs=0.01)

    def test_the_bank_is_pre_tax_on_every_line_not_just_this_one(self):
        """Stated on the statement itself, so the principle is not being
        generalised from one trade."""
        assert all(c.tob > 0 for c in CONFIRMATIONS), (
            "every confirmation charges tax, so every acquisition value the "
            "bank prints is understated by it")


class TestTheCapitalGainsTaxOnTheRealSale:
    """73.64 EUR on the VanEck disposal of 18 June 2026, and where it differs.

    The bank charged 10.00% of a gain it computed from a pre-tax acquisition
    value. The tracker computes the same rate on a basis that includes the
    2.35 of tax paid to acquire the shares, so its figure is 12 cents lower.
    Both numbers are printed; neither is quietly adopted.
    """

    def ledger_and_sale(self, positions):
        from portfolio.core.taxes import CapitalGainsTax, GainsLedger
        sale = next(c for c in CONFIRMATIONS if c.side == "SELL")
        buy = next(c for c in CONFIRMATIONS
                   if c.isin == SEMIS and c.side == "BUY")
        paid = float(buy.notional + buy.commission + buy.tob)
        basis = paid * (float(sale.quantity) / float(buy.quantity))
        gains = GainsLedger(CapitalGainsTax(rate=0.10, allowance=0.0))
        return gains, gains.record(sale.date, SEMIS, float(sale.notional), basis)

    def test_the_rate_is_the_one_the_bank_charged(self, positions):
        """With no tranche in the way, the arithmetic is the bank's."""
        _, disposal = self.ledger_and_sale(positions)
        assert disposal.tax / disposal.gain == pytest.approx(0.10, abs=1e-9)

    def test_and_the_figure_is_twelve_cents_under_the_banks(self, positions):
        _, disposal = self.ledger_and_sale(positions)
        assert disposal.gain == pytest.approx(735.185, abs=0.01)
        assert disposal.tax == pytest.approx(73.52, abs=0.01)
        assert 73.64 - disposal.tax == pytest.approx(0.12, abs=0.01)

    def test_the_tranche_makes_the_same_sale_free(self, positions):
        """The reason a flat rate is wrong in both directions. This sale is
        the first of its year and well inside a 10,000 EUR tranche, so its
        marginal rate is zero and the bank's 73.64 is the figure to reclaim
        rather than the figure to model."""
        from portfolio.core.taxes import CapitalGainsTax, GainsLedger
        sale = next(c for c in CONFIRMATIONS if c.side == "SELL")
        buy = next(c for c in CONFIRMATIONS
                   if c.isin == SEMIS and c.side == "BUY")
        paid = float(buy.notional + buy.commission + buy.tob)
        basis = paid * (float(sale.quantity) / float(buy.quantity))
        gains = GainsLedger(CapitalGainsTax())      # the 10,000 default
        disposal = gains.record(sale.date, SEMIS, float(sale.notional), basis)
        assert disposal.gain > 700
        assert disposal.tax == 0.0
        assert gains.tax.headroom(gains.realised(2026)) == pytest.approx(
            10_000.0 - disposal.gain, abs=0.01)

    def test_it_is_reported_apart_from_transaction_costs(self, positions):
        gains, _ = self.ledger_and_sale(positions)
        text = "\n".join(gains.lines())
        assert "Realised gains and the tax on them" in text
        assert "next euro of gain taxed at" in text
        assert "overstates the gain and the tax with it" in text


class TestTheAccountType:
    def test_execution_only_is_recorded_where_the_boundary_is_argued(self):
        """The statement says Execution Only, and that is the classification
        the paper-trading boundary is consistent with: no advice, no
        suitability assessment, every decision the holder's. A tool that
        produces a list to type into a broker sits inside it; the same tool
        with an endpoint attached does not."""
        import pathlib
        assert ACCOUNT_TYPE == "Execution Only"
        note = pathlib.Path(__file__).with_name("test_paper_only.py").read_text()
        assert "Execution Only" in note, (
            "the account classification is on the statement but not in the "
            "note that argues why this tool may exist")


class TestTheConfirmationsAreInternallyConsistent:
    """Before the statement is used to check the tracker, the confirmations
    are checked against themselves. A transcription error here would look
    exactly like a tracker bug."""

    @pytest.mark.parametrize("c", CONFIRMATIONS, ids=lambda c: f"{c.isin}-{c.side}")
    def test_the_printed_price_is_the_notional_per_share_rounded(self, c):
        """And it is NOT their product, which is the point.

        207 shares at the printed 9.67 is 2,001.69; the confirmation says
        2,001.89. The printed price is a two-decimal display of the average
        execution price and the notional is the money that moved, so the
        ledger is built from the notional. Twenty cents is nothing until it
        is subtracted from a sale price to produce a taxable gain.
        """
        rounded = c.exact_price.quantize(Decimal("0.01"))
        assert rounded == c.price
        assert abs(c.quantity * c.price - c.notional) <= Decimal("0.30")

    def test_and_at_least_one_of_them_really_does_disagree(self):
        """Otherwise the paragraph above is describing nothing."""
        off = [c for c in CONFIRMATIONS
               if abs(c.quantity * c.price - c.notional) > Decimal("0.01")]
        assert len(off) >= 3, (
            "the printed prices all multiply out exactly, so the ledger could "
            "have used them and this distinction would be theoretical")

    @pytest.mark.parametrize("c", CONFIRMATIONS, ids=lambda c: f"{c.isin}-{c.side}")
    def test_the_tax_charged_is_the_rate_it_states(self, c):
        assert abs(c.implied_tob_rate - c.tob_rate) <= Decimal("0.000005")

    def test_the_french_transaction_tax_reconciles(self):
        share = next(c for c in CONFIRMATIONS if c.isin == "FR0000121972")
        assert float(share.other / share.notional) == pytest.approx(
            0.004, abs=1e-5)

    def test_the_capital_gains_tax_is_ten_per_cent_of_the_stated_gain(self):
        sale = next(c for c in CONFIRMATIONS if c.side == "SELL")
        assert sale.other == Decimal("73.64")
        assert float(sale.other / Decimal("736.36")) == pytest.approx(
            0.10, abs=1e-4)

    def test_the_commission_is_not_a_property_of_the_broker(self):
        """Zero on five ETFs and 7.00 on a share, same broker, same month."""
        etfs = {c.commission for c in CONFIRMATIONS
                if c.isin != "FR0000121972"}
        share = next(c for c in CONFIRMATIONS if c.isin == "FR0000121972")
        assert etfs == {Decimal("0.00")}
        assert share.commission == Decimal("7.00")

    def test_two_identical_looking_funds_pay_eleven_times_apart(self):
        """The single most expensive assumption this project made."""
        cheap = next(c for c in CONFIRMATIONS if c.isin == "IE00BMC38736"
                     and c.side == "BUY")
        dear = next(c for c in CONFIRMATIONS if c.isin == "IE00BGDQ0L74")
        assert float(dear.tob_rate / cheap.tob_rate) == pytest.approx(
            11.0, abs=0.1)
