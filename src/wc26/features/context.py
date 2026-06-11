"""Layer-2 Tier-A context features (plan section 4).

Every function takes the team and an `asof` timestamp and reads only matches
strictly before it — the point-in-time discipline that makes the same code
leakage-safe in backtest and live. These are the features that are honestly
reconstructible across the full history from dates + results + the Layer-1 fit
(unlike lineups/injuries, which are Tier-C and live-only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REST_CAP_DAYS = 30.0
FORM_WINDOW = 10
FORM_SHRINK = 6.0  # pseudo-matches pulling the residual toward 0


def _team_history(history: pd.DataFrame, team: str, asof: pd.Timestamp) -> pd.DataFrame:
    past = history[(history.date < asof) & ((history.home == team) | (history.away == team))]
    return past.sort_values("date")


def rest_days(history: pd.DataFrame, team: str, asof: pd.Timestamp) -> float:
    """Days since the team's last match, capped (beyond the cap = fully rested)."""
    past = _team_history(history, team, asof)
    if past.empty:
        return REST_CAP_DAYS
    gap = (asof - past.date.iloc[-1]).days
    return float(min(max(gap, 0), REST_CAP_DAYS))


def form_residual(history, fit, team: str, asof: pd.Timestamp,
                  window: int = FORM_WINDOW, shrink: float = FORM_SHRINK) -> float:
    """Recent goals scored minus the Layer-1 rating-implied expectation, per match.

    Positive = scoring above what the (time-decayed) rating predicts. Shrunk
    toward zero by `shrink` pseudo-matches because short-window form in
    international football is mostly noise. Uses the point-in-time fit, so all
    referenced matches predate `asof`.
    """
    past = _team_history(history, team, asof).tail(window)
    if past.empty:
        return 0.0
    resid = 0.0
    for home, away, hg, ag, at_home in past[
        ["home", "away", "hg", "ag", "home_at_home"]
    ].itertuples(index=False):
        if home == team:
            exp, _ = fit.rates(home, away, bool(at_home))
            resid += int(hg) - exp
        else:
            _, exp = fit.rates(home, away, bool(at_home))
            resid += int(ag) - exp
    return float(resid / (len(past) + shrink))


def match_features(history, fit, home: str, away: str, asof: pd.Timestamp) -> dict:
    """Symmetric context features oriented from the home side.

    rest_adv  : home rest minus away rest (days, capped)  -> more rest, score more
    form_home : home recent goal residual                 -> applied to home rate
    form_away : away recent goal residual                 -> applied to away rate
    """
    rh = rest_days(history, home, asof)
    ra = rest_days(history, away, asof)
    return {
        "rest_adv": (rh - ra),
        "form_home": form_residual(history, fit, home, asof),
        "form_away": form_residual(history, fit, away, asof),
    }
