"""Layer 3: scoreline probability matrix and extra-time convolution.

90' matrix: P(x, y) = tau(x, y; rho) * Pois(x; lam) * Pois(y; mu) on a
0..max_goals grid, tail mass folded into the top bin, renormalized.

120' matrix (knockouts): non-draw 90' outcomes carry over unchanged; each 90'
draw (d, d) is convolved with an independent-Poisson 30' extra-time process at
kappa-scaled rates (plan section 8). Shootouts are outside the target.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def _pois_vector(rate: float, max_goals: int) -> np.ndarray:
    p = poisson.pmf(np.arange(max_goals + 1), rate)
    p[max_goals] = max(1.0 - poisson.cdf(max_goals - 1, rate), 0.0)  # fold tail
    return p


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    m = np.outer(_pois_vector(lam, max_goals), _pois_vector(mu, max_goals))
    m[0, 0] *= max(1.0 - lam * mu * rho, 1e-8)
    m[0, 1] *= max(1.0 + lam * rho, 1e-8)
    m[1, 0] *= max(1.0 + mu * rho, 1e-8)
    m[1, 1] *= max(1.0 - rho, 1e-8)
    return m / m.sum()


def et_convolve(m90: np.ndarray, lam: float, mu: float, kappa: float = 0.9) -> np.ndarray:
    """120-minute score distribution from the 90-minute matrix."""
    n = m90.shape[0] - 1
    out = m90.copy()
    draws_90 = np.diag(m90).copy()
    np.fill_diagonal(out, 0.0)  # 90' draw mass is redistributed through ET below
    et_h = _pois_vector(kappa * lam / 3.0, n)
    et_a = _pois_vector(kappa * mu / 3.0, n)
    add = np.outer(et_h, et_a)
    for d, p_draw in enumerate(draws_90):
        if p_draw <= 0:
            continue
        w = n + 1 - d  # ET-goal grid that still fits; excess folds into the top bin
        block = add[:w, :w].copy()
        block[-1, :] += add[w:, :w].sum(axis=0)
        block[:, -1] += add[:w, w:].sum(axis=1)
        block[-1, -1] += add[w:, w:].sum()
        out[d:, d:] += p_draw * block
    return out / out.sum()


def outcome_probs(m: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.tril(m, -1).sum()),   # home win (x > y): lower triangle
        float(np.trace(m)),
        float(np.triu(m, 1).sum()),
    )


def expected_goals(m: np.ndarray) -> tuple[float, float]:
    g = np.arange(m.shape[0])
    return float(g @ m.sum(axis=1)), float(g @ m.sum(axis=0))


def draw_prob_after_90(m90: np.ndarray) -> float:
    return float(np.trace(m90))
