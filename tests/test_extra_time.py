from wc26.scoreline.extra_time import fit_kappa


def test_kappa_in_plausible_range(matches, shootouts):
    import pandas as pd

    from wc26.paths import processed_dir

    goalscorers = pd.read_parquet(processed_dir() / "goalscorers.parquet")
    res = fit_kappa(matches, goalscorers, shootouts)
    # Enough complete ET periods to be meaningful, and all three estimators land
    # in the physically plausible band the plan anticipated (~0.7-1.2).
    assert res.n_matches > 150
    assert 0.4 < res.et_goals_per_match < 1.5
    assert 0.6 < res.kappa_self_normalized < 1.3
    assert 0.6 < res.kappa_population < 1.1
    assert 0.7 < res.kappa_match_rate < 1.2
