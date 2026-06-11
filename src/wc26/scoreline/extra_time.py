"""Fit the extra-time intensity multiplier kappa from goal-minute data.

The convolution in matrix.py models extra time as a 30' independent-Poisson
process at rates kappa * lam * (30/90). kappa captures ET-specific effects
(fatigue, caution) on top of each match's own regulation rate.

Estimation is awkward because (a) goalscorers.csv is incomplete for many
matches and (b) ET-bound matches are tied at 90', so they are systematically
tighter than average. We therefore:
  - keep only matches whose goal rows fully reconstruct the final score, and
  - report two bracketing estimators plus the match-rate synthesis the
    convolution actually needs.
The data supports a single global kappa and no more (plan section 8).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

KEY = ["date", "home", "away"]


@dataclass
class KappaFit:
    n_matches: int
    et_goals_per_match: float
    kappa_self_normalized: float   # vs the ET matches' own regulation rate (biased high)
    kappa_population: float        # vs all complete matches' regulation rate (biased low)
    kappa_match_rate: float        # vs assumed knockout regulation rate (the headline)
    assumed_knockout_rate: float


def fit_kappa(
    matches: pd.DataFrame,
    goalscorers: pd.DataFrame,
    shootouts: pd.DataFrame,
    knockout_rate: float = 2.5,
) -> KappaFit:
    g = goalscorers.dropna(subset=["minute"])
    gk = g[KEY].apply(tuple, axis=1)

    n_goals = g.groupby(KEY).size().rename("n_goals")
    mm = matches.merge(n_goals, on=KEY, how="left").fillna({"n_goals": 0})
    mm["total"] = mm.hg + mm.ag
    complete = set(map(tuple, mm[(mm.n_goals == mm.total) & (mm.total > 0)][KEY].values))

    shoot = set(map(tuple, shootouts[KEY].values))
    late = set(map(tuple, g[g.minute >= 100][KEY].drop_duplicates().values))
    et = (shoot | late) & complete                 # ET matches with complete data
    reg_pool = complete - shoot - late             # non-ET matches with complete data

    ge = g[gk.isin(et)]
    et_goals = int(((ge.minute > 90) & (ge.minute <= 120)).sum())
    et_reg_goals = int((ge.minute <= 90).sum())

    gr = g[gk.isin(reg_pool) & (g.minute <= 90)]
    pop_reg_per_match = len(gr) / max(len(reg_pool), 1)

    n = len(et)
    et_per_match = et_goals / max(n, 1)
    rate_et = et_goals / max(n * 30, 1)
    rate_reg_self = et_reg_goals / max(n * 90, 1)

    return KappaFit(
        n_matches=n,
        et_goals_per_match=round(et_per_match, 4),
        kappa_self_normalized=round(rate_et / rate_reg_self, 4) if rate_reg_self else float("nan"),
        kappa_population=round(et_per_match / (pop_reg_per_match * 30 / 90), 4),
        kappa_match_rate=round(et_per_match / (knockout_rate * 30 / 90), 4),
        assumed_knockout_rate=knockout_rate,
    )
