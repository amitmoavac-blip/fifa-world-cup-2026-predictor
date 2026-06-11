import numpy as np
import pandas as pd
import pytest

from wc26.ratings import dixon_coles as dc


def _synthetic_matches(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    teams = ["Brazil", "Argentina", "France", "Germany", "Japan", "Morocco", "Panama", "Fiji"]
    strength = {t: s for t, s in zip(teams, [0.5, 0.45, 0.4, 0.35, 0.0, -0.05, -0.4, -0.8])}
    rows = []
    base_date = pd.Timestamp("2022-01-01")
    for k in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        at_home = bool(rng.random() < 0.5)
        lam = np.exp(0.1 + strength[h] - (-strength[a]) * 0.5 + (0.25 if at_home else 0))
        mu = np.exp(0.1 + strength[a] - (-strength[h]) * 0.5)
        rows.append(
            dict(
                date=base_date + pd.Timedelta(days=int(k / 4)),
                home=h, away=a,
                hg=rng.poisson(lam), ag=rng.poisson(mu),
                tournament="Friendly", importance="qualifier",
                neutral=not at_home, home_at_home=at_home, played=True,
            )
        )
    return pd.DataFrame(rows)


def test_gradient_matches_finite_differences():
    """Analytic gradient of the DC objective vs numerical differentiation."""
    df = _synthetic_matches(n=400)
    cfg = dc.DCConfig(halflife_years=5.0)
    fun, dims = dc._build_objective(df, pd.Timestamp("2025-01-01"), cfg)

    rng = np.random.default_rng(3)
    theta = 0.05 * rng.standard_normal(dims["n_par"])
    theta[2] = 0.05  # rho inside bounds
    f0, g = fun(theta)
    eps = 1e-6
    for idx in [0, 1, 2, 5, 10, dims["n_par"] - 1]:
        tp = theta.copy()
        tp[idx] += eps
        fp, _ = fun(tp)
        num = (fp - f0) / eps
        assert num == pytest.approx(g[idx], rel=2e-3, abs=2e-3), f"grad mismatch at {idx}"


def test_fit_recovers_ordering():
    df = _synthetic_matches()
    f = dc.fit(df, asof=pd.Timestamp("2025-01-01"), cfg=dc.DCConfig(halflife_years=5.0))
    # Strong teams should have higher attack and defense than weak ones.
    assert f.attack["Brazil"] > f.attack["Panama"] > f.attack["Fiji"]
    lam, mu = f.rates("Brazil", "Fiji", home_at_home=False)
    assert lam > mu
    lam_home, _ = f.rates("Brazil", "Fiji", home_at_home=True)
    assert lam_home > lam  # host advantage positive on synthetic data


def test_unseen_team_falls_back_to_confederation():
    df = _synthetic_matches()
    f = dc.fit(df, asof=pd.Timestamp("2025-01-01"), cfg=dc.DCConfig(halflife_years=5.0))
    lam, mu = f.rates("Tuvalu", "Brazil", home_at_home=False)  # never in training
    assert 0 < lam < mu


def test_leakage_guard():
    """A future result, however extreme, must not move the as-of fit."""
    df = _synthetic_matches()
    f1 = dc.fit(df, asof=pd.Timestamp("2024-01-01"), cfg=dc.DCConfig(halflife_years=5.0))
    future = pd.DataFrame(
        [dict(date=pd.Timestamp("2024-06-01"), home="Fiji", away="Brazil", hg=9, ag=0,
              tournament="FIFA World Cup", importance="world_cup",
              neutral=False, home_at_home=True, played=True)]
    )
    f2 = dc.fit(pd.concat([df, future], ignore_index=True),
                asof=pd.Timestamp("2024-01-01"), cfg=dc.DCConfig(halflife_years=5.0))
    assert f1.attack["Fiji"] == pytest.approx(f2.attack["Fiji"], abs=1e-9)
    assert f1.attack["Brazil"] == pytest.approx(f2.attack["Brazil"], abs=1e-9)
