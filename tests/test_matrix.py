import numpy as np
import pytest

from wc26.scoreline.matrix import (
    draw_prob_after_90,
    et_convolve,
    expected_goals,
    outcome_probs,
    score_matrix,
)


def test_matrix_sums_to_one():
    for lam, mu in [(0.6, 0.5), (1.5, 1.2), (4.5, 0.3)]:
        m = score_matrix(lam, mu, rho=-0.08)
        assert m.sum() == pytest.approx(1.0)
        assert (m >= 0).all()


def test_tail_folding_keeps_mass():
    # Huge rates: the top bin must absorb the tail, total still 1.
    m = score_matrix(8.0, 7.0, rho=0.0, max_goals=10)
    assert m.sum() == pytest.approx(1.0)
    assert m[10, :].sum() > 0.01


def test_negative_rho_inflates_draws():
    base = score_matrix(1.3, 1.1, rho=0.0)
    adj = score_matrix(1.3, 1.1, rho=-0.1)
    # Dixon-Coles with negative rho raises 0-0 and 1-1 mass.
    assert adj[0, 0] > base[0, 0]
    assert adj[1, 1] > base[1, 1]
    assert adj[1, 0] < base[1, 0]


def test_outcome_probs_orientation():
    m = score_matrix(2.5, 0.5, rho=0.0)
    ph, pd_, pa = outcome_probs(m)
    assert ph > 0.75 > pa
    assert ph + pd_ + pa == pytest.approx(1.0)


def test_et_convolution():
    m90 = score_matrix(1.4, 1.1, rho=-0.06)
    m120 = et_convolve(m90, 1.4, 1.1, kappa=0.9)
    assert m120.sum() == pytest.approx(1.0)
    # Non-draw 90' outcomes carry over unchanged.
    assert m120[2, 1] >= m90[2, 1] - 1e-12
    expected_21 = (
        m90[2, 1]
        + m90[0, 0] * _et_cell(1.4, 1.1, 0.9, 2, 1)
        + m90[1, 1] * _et_cell(1.4, 1.1, 0.9, 1, 0)
    )
    assert m120[2, 1] == pytest.approx(expected_21, rel=1e-6)
    # Draws lose mass to ET goals but some draws survive (shootout excluded).
    assert draw_prob_after_90(m120) < draw_prob_after_90(m90)
    assert np.trace(m120) > 0.05
    # Expected goals rise once ET is possible.
    assert sum(expected_goals(m120)) > sum(expected_goals(m90))


def _et_cell(lam, mu, kappa, u, v):
    from scipy.stats import poisson

    return poisson.pmf(u, kappa * lam / 3) * poisson.pmf(v, kappa * mu / 3)
