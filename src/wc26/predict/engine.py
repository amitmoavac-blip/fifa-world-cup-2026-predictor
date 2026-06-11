"""Prediction engine: orchestrates Layers 1 -> 3 -> 5 for one fixture.

Single entry point used identically by the backtest harness and live
prediction — the as-of discipline lives in the DCFit, which only ever saw
matches before its asof timestamp.
"""
from __future__ import annotations

import numpy as np

from wc26.predict.explain import build_explanation
from wc26.predict.select import confidence_label, select_score, top_scores
from wc26.ratings.dixon_coles import DCFit
from wc26.scoreline.matrix import (
    draw_prob_after_90,
    et_convolve,
    expected_goals,
    outcome_probs,
    score_matrix,
)


def predict_match(
    fit: DCFit,
    home: str,
    away: str,
    home_at_home: bool,
    knockout: bool,
    cfg: dict,
) -> dict:
    sc = cfg.get("scoreline", {})
    sel = cfg.get("selection", {})
    max_goals = int(sc.get("max_goals", 10))
    mu_cap = float(sc.get("mu_cap", 4.5))
    kappa = float(sc.get("et_kappa", 0.9))

    lam_raw, mu_raw = fit.rates(home, away, home_at_home)
    lam, mu = min(lam_raw, mu_cap), min(mu_raw, mu_cap)
    capped = (lam_raw > mu_cap) or (mu_raw > mu_cap)

    m90 = score_matrix(lam, mu, fit.rho, max_goals)
    p_draw_90 = draw_prob_after_90(m90)
    matrix = et_convolve(m90, lam, mu, kappa) if knockout else m90

    (sx, sy), modal_prob, ties = select_score(
        matrix, lam, mu, float(sel.get("tie_epsilon", 0.005))
    )
    p_home, p_draw, p_away = outcome_probs(matrix)
    xg_h, xg_a = expected_goals(matrix)

    info = {
        "home": home,
        "away": away,
        "home_at_home": home_at_home,
        "knockout": knockout,
        "lam": lam,
        "mu": mu,
        "host_boost": float(np.expm1(fit.h)) if home_at_home else 0.0,
        "p_draw_90": p_draw_90,
        "et_xg": kappa * (lam + mu) / 3.0 if knockout else 0.0,
        "mu_capped": capped,
        "stale": (home not in fit.attack) or (away not in fit.attack),
        "modal_score": f"{sx}-{sy}",
        "modal_prob": modal_prob,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
    }
    demote = capped or info["stale"]
    return {
        **info,
        "confidence": confidence_label(modal_prob, sel.get("confidence_bands"), demote),
        "explanation": build_explanation(info),
        "top_scores": top_scores(matrix, 5),
        "tie_candidates": [f"{a}-{b}" for a, b in ties] if len(ties) > 1 else [],
        "matrix": matrix,
        "xg_120_home": xg_h,
        "xg_120_away": xg_a,
        "asof": str(fit.asof.date()),
    }
