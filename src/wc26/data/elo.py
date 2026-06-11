"""Point-in-time Elo ratings, recomputed in-repo from the match stream.

Published eloratings.net snapshots embed future information relative to
historical cutoffs (plan section 14, risk 2); recursive recomputation is the
only leakage-safe option. Follows the World Football Elo conventions:
importance-based K, goal-difference multiplier, +100 home advantage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EloConfig:
    init_rating: float = 1500.0
    home_advantage: float = 100.0
    k_factors: dict[str, float] = field(
        default_factory=lambda: {
            "world_cup": 60.0,
            "continental_finals": 50.0,
            "qualifier": 40.0,
            "nations_league": 40.0,
            "minor": 30.0,
            "friendly": 20.0,
        }
    )


def _gd_multiplier(gd: int) -> float:
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + (gd - 3) / 8.0


def compute_elo_history(matches: pd.DataFrame, cfg: EloConfig | None = None) -> pd.DataFrame:
    """Sequential Elo over played matches (must be date-sorted).

    Returns the input frame plus elo_home/elo_away columns holding each
    team's rating *before* the match — i.e. the point-in-time feature.
    """
    cfg = cfg or EloConfig()
    ratings: dict[str, float] = {}
    pre_home = np.empty(len(matches))
    pre_away = np.empty(len(matches))

    rows = zip(
        matches["home"].to_numpy(),
        matches["away"].to_numpy(),
        matches["hg"].to_numpy(),
        matches["ag"].to_numpy(),
        matches["importance"].to_numpy(),
        matches["home_at_home"].to_numpy(),
    )
    for idx, (home, away, hg, ag, imp, at_home) in enumerate(rows):
        rh = ratings.get(home, cfg.init_rating)
        ra = ratings.get(away, cfg.init_rating)
        pre_home[idx] = rh
        pre_away[idx] = ra

        diff = rh - ra + (cfg.home_advantage if at_home else 0.0)
        expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        result = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k = cfg.k_factors.get(imp, 30.0) * _gd_multiplier(abs(int(hg) - int(ag)))
        delta = k * (result - expected)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

    out = matches.copy()
    out["elo_home"] = pre_home
    out["elo_away"] = pre_away
    return out


def ratings_asof(matches: pd.DataFrame, asof: pd.Timestamp, cfg: EloConfig | None = None) -> dict[str, float]:
    """Ratings using only matches strictly before `asof`."""
    cfg = cfg or EloConfig()
    past = matches[matches.date < asof]
    ratings: dict[str, float] = {}
    rows = zip(
        past["home"].to_numpy(), past["away"].to_numpy(), past["hg"].to_numpy(),
        past["ag"].to_numpy(), past["importance"].to_numpy(), past["home_at_home"].to_numpy(),
    )
    for home, away, hg, ag, imp, at_home in rows:
        rh = ratings.get(home, cfg.init_rating)
        ra = ratings.get(away, cfg.init_rating)
        diff = rh - ra + (cfg.home_advantage if at_home else 0.0)
        expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        result = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k = cfg.k_factors.get(imp, 30.0) * _gd_multiplier(abs(int(hg) - int(ag)))
        delta = k * (result - expected)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings
