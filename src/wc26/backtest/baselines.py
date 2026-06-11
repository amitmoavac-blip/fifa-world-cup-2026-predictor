"""Point-in-time baselines the model must beat (plan section 9).

- constant_modal: the most frequent exact score across prior major-tournament
  matches. Surprisingly hard to beat on hit rate; the honesty anchor.
- elo_poisson: independent Poisson with goal rates from a 2-parameter GLM on
  the (self-computed, pre-match) Elo difference. The classic simple model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc26.scoreline.matrix import et_convolve, expected_goals, outcome_probs, score_matrix
from wc26.predict.select import select_score

_MAJORS = ("world_cup", "continental_finals")


def constant_modal_score(history: pd.DataFrame, asof: pd.Timestamp) -> tuple[int, int]:
    past = history[(history.date < asof) & history.importance.isin(_MAJORS) & history.played]
    if past.empty:
        return (1, 1)
    counts = past.groupby([past.hg.astype(int), past.ag.astype(int)]).size()
    return tuple(int(v) for v in counts.idxmax())


class EloPoisson:
    """log(goals) = b0 + b1 * elo_edge/400, fitted by Poisson MLE."""

    def __init__(self, b0: float, b1: float):
        self.b0, self.b1 = b0, b1

    @classmethod
    def fit(cls, history: pd.DataFrame, asof: pd.Timestamp,
            home_adv: float = 100.0, window_years: float = 20.0) -> "EloPoisson":
        df = history[
            (history.date < asof)
            & history.played
            & (history.date >= asof - pd.Timedelta(days=365.25 * window_years))
        ]
        edge_h = (df.elo_home + home_adv * df.home_at_home - df.elo_away).to_numpy() / 400.0
        x = np.concatenate([edge_h, -edge_h])
        g = np.concatenate([df.hg.to_numpy(dtype=float), df.ag.to_numpy(dtype=float)])
        g = np.minimum(g, 10)

        def nll(theta):
            log_lam = theta[0] + theta[1] * x
            lam = np.exp(log_lam)
            f = -np.sum(g * log_lam - lam)
            grad = -np.array([np.sum(g - lam), np.sum((g - lam) * x)])
            return f, grad

        res = minimize(nll, np.array([0.1, 0.5]), jac=True, method="L-BFGS-B")
        return cls(float(res.x[0]), float(res.x[1]))

    def rates(self, elo_home: float, elo_away: float, home_at_home: bool,
              home_adv: float = 100.0) -> tuple[float, float]:
        edge = (elo_home + (home_adv if home_at_home else 0.0) - elo_away) / 400.0
        return float(np.exp(self.b0 + self.b1 * edge)), float(np.exp(self.b0 - self.b1 * edge))


def elo_poisson_predict(model: EloPoisson, elo_home: float, elo_away: float,
                        home_at_home: bool, knockout: bool,
                        max_goals: int = 10, kappa: float = 0.9, mu_cap: float = 4.5) -> dict:
    lam, mu = model.rates(elo_home, elo_away, home_at_home)
    lam, mu = min(lam, mu_cap), min(mu, mu_cap)
    m90 = score_matrix(lam, mu, rho=0.0, max_goals=max_goals)  # plain Poisson: no DC tau
    matrix = et_convolve(m90, lam, mu, kappa) if knockout else m90
    (sx, sy), modal_prob, _ = select_score(matrix, lam, mu)
    p_h, p_d, p_a = outcome_probs(matrix)
    return {
        "modal_score": f"{sx}-{sy}", "modal_prob": modal_prob, "matrix": matrix,
        "p_home": p_h, "p_draw": p_d, "p_away": p_a, "confidence": "-",
    }


def constant_modal_predict(score: tuple[int, int], max_goals: int = 10) -> dict:
    # Degenerate distribution: metrics other than hit rate are not meaningful,
    # so give it a uniform matrix for logloss/rps to avoid -inf artifacts.
    m = np.full((max_goals + 1, max_goals + 1), 1.0 / (max_goals + 1) ** 2)
    return {
        "modal_score": f"{score[0]}-{score[1]}", "modal_prob": np.nan, "matrix": m,
        "p_home": 1 / 3, "p_draw": 1 / 3, "p_away": 1 / 3, "confidence": "-",
    }
