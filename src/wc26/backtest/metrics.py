"""Evaluation metrics for exact-score predictions (plan section 9)."""
from __future__ import annotations

import numpy as np
import pandas as pd

_LOG_FLOOR = 1e-6


def score_logloss(matrix: np.ndarray, hg: int, ag: int) -> float:
    n = matrix.shape[0] - 1
    return -float(np.log(max(matrix[min(hg, n), min(ag, n)], _LOG_FLOOR)))


def rps_1x2(p_home: float, p_draw: float, p_away: float, hg: int, ag: int) -> float:
    pred = np.cumsum([p_home, p_draw, p_away])
    obs = np.cumsum(np.eye(3)[0 if hg > ag else (1 if hg == ag else 2)])
    return float(np.sum((pred[:2] - obs[:2]) ** 2) / 2.0)


def match_record(pred: dict, hg: int, ag: int) -> dict:
    sx, sy = (int(s) for s in pred["modal_score"].split("-"))
    actual_outcome = "H" if hg > ag else ("D" if hg == ag else "A")
    pred_outcome = max(
        zip("HDA", (pred["p_home"], pred["p_draw"], pred["p_away"])), key=lambda e: e[1]
    )[0]
    return {
        "exact_hit": int(sx == hg and sy == ag),
        "logloss": score_logloss(pred["matrix"], hg, ag),
        "rps": rps_1x2(pred["p_home"], pred["p_draw"], pred["p_away"], hg, ag),
        "home_mae": abs(sx - hg),
        "away_mae": abs(sy - ag),
        "total_mae": abs((sx + sy) - (hg + ag)),
        "gd_mae": abs((sx - sy) - (hg - ag)),
        "outcome_hit": int(pred_outcome == actual_outcome),
        "modal_prob": pred["modal_prob"],
        "confidence": pred["confidence"],
    }


def aggregate(records: pd.DataFrame) -> dict:
    return {
        "n": int(len(records)),
        "exact_hit_rate": round(records.exact_hit.mean(), 4),
        "logloss": round(records.logloss.mean(), 4),
        "rps": round(records.rps.mean(), 4),
        "outcome_acc": round(records.outcome_hit.mean(), 4),
        "home_goals_mae": round(records.home_mae.mean(), 3),
        "away_goals_mae": round(records.away_mae.mean(), 3),
        "total_goals_mae": round(records.total_mae.mean(), 3),
        "gd_mae": round(records.gd_mae.mean(), 3),
        "mean_modal_prob": round(records.modal_prob.mean(), 4),
    }
