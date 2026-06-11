import numpy as np
import pandas as pd

from wc26.features.xg_form import match_xg_features


class _StubFit:
    def rates(self, team, opp, at_home):
        return 1.2, 1.0  # constant rating-expected for/against


def _xg_table():
    rows = [
        # WC group stage: A creates lots of xG but under-finishes; B over-finishes.
        ("2022-11-21", "A", "X", 1, 0, 2.4, 0.5, "WC"),
        ("2022-11-25", "A", "Y", 0, 1, 2.0, 0.8, "WC"),
        ("2022-11-21", "B", "Z", 3, 0, 1.0, 0.9, "WC"),
        # A different tournament — must not leak into WC features.
        ("2024-06-15", "A", "Q", 5, 0, 3.0, 0.2, "Euro"),
    ]
    df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag", "xg_home", "xg_away", "tournament_label"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_within_tournament_and_leakage_clean():
    xg = _xg_table()
    fit = _StubFit()
    # Predicting A vs B on MD3; only A's two earlier WC matches count.
    feats = match_xg_features(xg, fit, "WC", "A", "B", pd.Timestamp("2022-11-29"))
    assert feats["n_prior_home"] == 2          # A's two WC matches
    assert feats["n_prior_away"] == 1          # B's one WC match
    assert feats["xg_available"] == 1
    # A created ~2.2 xG vs 1.2 expected -> positive attacking residual.
    assert feats["home_att"] > 0.5
    # A scored 0.5 goals/match vs 2.2 xG -> strongly negative finishing residual.
    assert feats["home_fin"] < 0
    # The Euro match (future tournament, and after asof) must not appear.
    assert feats["n_prior_home"] == 2


def test_no_prior_matches_returns_zeros():
    xg = _xg_table()
    feats = match_xg_features(xg, _StubFit(), "WC", "A", "B", pd.Timestamp("2022-11-21"))
    # As of the first matchday, neither team has a prior WC match.
    assert feats["n_prior_home"] == 0
    assert feats["xg_available"] == 0
    assert feats["home_att"] == 0.0


def test_statsbomb_cache_present(matches):
    # The cached xG table should exist and align with the registry.
    from wc26.data.statsbomb import load_xg
    from wc26.data.registry import confed_map

    try:
        xg = load_xg()
    except FileNotFoundError:
        import pytest

        pytest.skip("StatsBomb xG not ingested in this environment")
    assert len(xg) > 300
    cmap = confed_map()
    teams = set(xg.home) | set(xg.away)
    assert not [t for t in teams if t not in cmap], "unmapped xG team names"
