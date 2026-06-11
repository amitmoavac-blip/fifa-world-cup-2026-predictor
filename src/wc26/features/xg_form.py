"""Tier-B within-tournament xG-form features (plan: Tier-B, leakage-clean).

For a target match, look only at each team's EARLIER matches in the same
tournament (xG exists only within StatsBomb tournaments) and summarise how the
team's chance creation/prevention and finishing have deviated from what the
Layer-1 rating predicts. Opponent adjustment comes for free via the rating-
implied expectation. All inputs predate the target match.

Per side, three signals (lower-noise than the goals-residual that tested null):
  att_resid : xG created minus rating-expected goals          (expect + persist)
  fin_resid : goals scored minus xG created                   (expect - revert)
  def_resid : xG conceded minus rating-expected goals against (opponent's leaks)
"""
from __future__ import annotations

import pandas as pd


def _prior(xg: pd.DataFrame, label: str, team: str, asof: pd.Timestamp) -> pd.DataFrame:
    same = xg[(xg.tournament_label == label) & (xg.date < asof)]
    return same[(same.home == team) | (same.away == team)]


def _team_residuals(xg, fit, label, team, asof) -> tuple[float, float, float, int]:
    rows = _prior(xg, label, team, asof)
    if rows.empty:
        return 0.0, 0.0, 0.0, 0
    att = fin = dfn = 0.0
    for home, away, hg, ag, xgh, xga in rows[
        ["home", "away", "hg", "ag", "xg_home", "xg_away"]
    ].itertuples(index=False):
        if home == team:
            xg_for, xg_against, g_for = xgh, xga, hg
            exp_for, exp_against = fit.rates(team, away, False)
        else:
            xg_for, xg_against, g_for = xga, xgh, ag
            exp_for, exp_against = fit.rates(team, home, False)
        att += xg_for - exp_for
        dfn += xg_against - exp_against
        fin += g_for - xg_for
    n = len(rows)
    return att / n, fin / n, dfn / n, n


def match_xg_features(xg, fit, label: str, home: str, away: str, asof: pd.Timestamp) -> dict:
    h_att, h_fin, h_def, h_n = _team_residuals(xg, fit, label, home, asof)
    a_att, a_fin, a_def, a_n = _team_residuals(xg, fit, label, away, asof)
    return {
        "home_att": h_att, "home_fin": h_fin, "home_def": h_def,
        "away_att": a_att, "away_fin": a_fin, "away_def": a_def,
        "n_prior_home": h_n, "n_prior_away": a_n,
        "xg_available": int(h_n > 0 and a_n > 0),
    }
