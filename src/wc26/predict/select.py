"""Layer 5: pick the single most likely exact score from the matrix.

Argmax is Bayes-optimal under exact-score 0-1 loss (the stated objective).
Tie-break within epsilon is deterministic and logged: prefer the score whose
sign matches sign(lam - mu), then lower total goals, then lower home goals.
"""
from __future__ import annotations

import numpy as np


def select_score(
    m: np.ndarray,
    lam: float,
    mu: float,
    tie_epsilon: float = 0.005,
) -> tuple[tuple[int, int], float, list[tuple[int, int]]]:
    """Returns ((x, y), modal_prob, tie_candidates)."""
    best = float(m.max())
    cand = [(int(i), int(j)) for i, j in zip(*np.where(m >= best - tie_epsilon))]
    rate_sign = np.sign(lam - mu)

    def key(s: tuple[int, int]):
        sign_consistent = np.sign(s[0] - s[1]) == rate_sign
        return (not sign_consistent, s[0] + s[1], s[0], s[1])

    cand.sort(key=key)
    chosen = cand[0]
    return chosen, float(m[chosen]), cand


def confidence_label(
    modal_prob: float,
    bands: dict | None = None,
    demote: bool = False,
) -> str:
    """Interval-band confidence, tuned on the backtest (plan section 2, Layer 5).

    Counter-intuitive but empirically robust: very HIGH modal probability marks
    a blowout favourite whose exact margin is inherently uncertain -> low.
    """
    b = bands or {"high": [0.15, 0.19], "low_below": 0.10, "low_above": 0.19}
    if modal_prob < b["low_below"] or modal_prob >= b["low_above"]:
        label = "low"
    elif b["high"][0] <= modal_prob < b["high"][1]:
        label = "high"
    else:
        label = "medium"
    if demote and label != "low":
        label = "medium" if label == "high" else "low"
    return label


def top_scores(m: np.ndarray, k: int = 5) -> list[dict]:
    flat = [((i, j), float(m[i, j])) for i in range(m.shape[0]) for j in range(m.shape[1])]
    flat.sort(key=lambda e: -e[1])
    return [{"score": f"{i}-{j}", "prob": round(p, 4)} for (i, j), p in flat[:k]]
