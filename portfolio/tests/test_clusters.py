"""Correlation clusters: the group the pairwise list cannot see.

The IDEAS.md case: four defence ETFs, three of them mutually correlated near
0.9, the fourth at 0.84 to each -- just under the 0.85 threshold -- and a
commodity ETC correlated with nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.core.risk import (HIGH_CORRELATION_THRESHOLD, CorrelationCluster,
                                 correlation_clusters, correlation_matrix,
                                 high_correlation_pairs)

D1, D2, D3, D4, GOLD = "IE0002Y8CX98", "IE000IAXNM41", "IE000I7E6HL0", "IE000OJ5TQP4", "JE00BN7KB664"


def corr(matrix: dict[tuple[str, str], float], names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for (a, b), rho in matrix.items():
        frame.loc[a, b] = frame.loc[b, a] = rho
    return frame


@pytest.fixture
def defence() -> pd.DataFrame:
    return corr({(D1, D2): 0.92, (D1, D3): 0.90, (D2, D3): 0.88,
                 (D1, D4): 0.84, (D2, D4): 0.84, (D3, D4): 0.84,
                 (D1, GOLD): 0.10, (D2, GOLD): 0.05, (D3, GOLD): 0.12, (D4, GOLD): 0.08},
                [D1, D2, D3, D4, GOLD])


class TestTheIdeasCase:
    def test_three_defence_etfs_are_one_cluster_at_the_threshold(self, defence):
        clusters = correlation_clusters(defence)
        assert len(clusters) == 1
        assert clusters[0].members == (D1, D3, D2)[:0] + tuple(sorted([D1, D2, D3]))
        assert GOLD not in clusters[0].members and D4 not in clusters[0].members

    def test_mean_and_min_are_over_every_pair_inside(self, defence):
        c = correlation_clusters(defence)[0]
        assert c.mean_correlation == pytest.approx((0.92 + 0.90 + 0.88) / 3)
        assert c.min_correlation == pytest.approx(0.88)

    def test_lowering_the_threshold_admits_the_fourth(self, defence):
        clusters = correlation_clusters(defence, threshold=0.80)
        assert len(clusters) == 1 and clusters[0].size == 4
        assert D4 in clusters[0].members and GOLD not in clusters[0].members
        assert clusters[0].min_correlation == pytest.approx(0.84)

    def test_the_pairwise_list_reports_three_edges_for_one_fact(self, defence):
        """Six facts on screen versus one: the reason the cluster exists."""
        assert len(high_correlation_pairs(defence)) == 3
        assert len(correlation_clusters(defence)) == 1

    def test_combined_weight(self, defence):
        weights = {D1: 0.20, D2: 0.15, D3: 0.10, D4: 0.25, GOLD: 0.30}
        c = correlation_clusters(defence, weights=weights)[0]
        assert c.combined_weight == pytest.approx(0.45)

    def test_no_weights_means_none(self, defence):
        assert correlation_clusters(defence)[0].combined_weight is None

    def test_member_missing_from_weights_contributes_nothing(self, defence):
        c = correlation_clusters(defence, weights={D1: 0.2, D2: 0.1})[0]
        assert c.combined_weight == pytest.approx(0.3)

    def test_default_threshold_is_the_shared_constant(self, defence):
        assert correlation_clusters(defence) == correlation_clusters(
            defence, threshold=HIGH_CORRELATION_THRESHOLD)


class TestStructure:
    def test_chain_is_one_cluster_with_the_weak_link_reported(self):
        """A~B and B~C above the threshold, A~C far below: connected components
        keep all three together and min_correlation shows the 0.30."""
        m = corr({(D1, D2): 0.9, (D2, D3): 0.9, (D1, D3): 0.3}, [D1, D2, D3])
        clusters = correlation_clusters(m)
        assert len(clusters) == 1
        assert clusters[0].members == tuple(sorted([D1, D2, D3]))
        assert clusters[0].min_correlation == pytest.approx(0.3)
        assert clusters[0].mean_correlation == pytest.approx(0.7)

    def test_two_separate_clusters(self):
        m = corr({(D1, D2): 0.9, (D3, D4): 0.95}, [D1, D2, D3, D4, GOLD])
        clusters = correlation_clusters(m)
        assert [c.members for c in clusters] == [tuple(sorted([D1, D2])), tuple(sorted([D3, D4]))]

    def test_singletons_are_dropped(self):
        m = corr({(D1, D2): 0.9}, [D1, D2, GOLD])
        assert [c.members for c in correlation_clusters(m)] == [tuple(sorted([D1, D2]))]

    def test_nothing_correlated_is_empty(self):
        m = corr({(D1, D2): 0.5}, [D1, D2, GOLD])
        assert correlation_clusters(m) == []

    def test_empty_and_single(self):
        assert correlation_clusters(pd.DataFrame()) == []
        assert correlation_clusters(corr({}, [D1])) == []

    def test_threshold_is_inclusive(self):
        m = corr({(D1, D2): 0.85}, [D1, D2])
        assert len(correlation_clusters(m, threshold=0.85)) == 1
        assert correlation_clusters(corr({(D1, D2): 0.8499}, [D1, D2]), threshold=0.85) == []

    def test_members_are_sorted(self):
        m = corr({(GOLD, D1): 0.9}, [GOLD, D1])
        assert correlation_clusters(m)[0].members == (D1, GOLD)

    def test_sorted_by_combined_weight_then_size(self):
        m = corr({(D1, D2): 0.9, (D3, D4): 0.9, (D4, GOLD): 0.9}, [D1, D2, D3, D4, GOLD])
        heavy_pair = correlation_clusters(m, weights={D1: 0.4, D2: 0.3, D3: 0.1, D4: 0.1, GOLD: 0.1})
        assert [c.size for c in heavy_pair] == [2, 3], "weight beats size"
        by_size = correlation_clusters(m)
        assert [c.size for c in by_size] == [3, 2], "without weights, size decides"

    def test_from_a_covariance_matrix(self):
        """End to end with the existing correlation_matrix."""
        n, rho = 3, 0.9
        block = np.full((n, n), rho * 0.0004)
        np.fill_diagonal(block, 0.0004)
        cov = pd.DataFrame(np.zeros((4, 4)), index=[D1, D2, D3, GOLD], columns=[D1, D2, D3, GOLD])
        cov.iloc[:3, :3] = block
        cov.loc[GOLD, GOLD] = 0.0009
        clusters = correlation_clusters(correlation_matrix(cov))
        assert len(clusters) == 1 and clusters[0].members == (D1, D3, D2)[:0] + tuple(sorted([D1, D2, D3]))
        assert clusters[0].mean_correlation == pytest.approx(0.9)

    def test_sentence_names_the_funds_and_the_weight(self):
        c = CorrelationCluster((D1, D2, D3), 0.9, 0.88, 0.45)
        s = c.sentence({D1: "WisdomTree Europe Defence", D2: "iShares Europe Defence"})
        assert "WisdomTree Europe Defence" in s and "iShares Europe Defence" in s and D3 in s
        assert "90%" in s and "88%" in s and "45%" in s
        assert "one position" in s

    def test_sentence_without_a_weight(self):
        s = CorrelationCluster((D1, D2), 0.9, 0.9, None).sentence()
        assert "of the portfolio" not in s
