import numpy as np
import pandas as pd
import pytest

from wc26.adjust.glm import GLMFit, build_sideframe, fit_glm, loto_cv
from wc26.features.context import match_features, rest_days


class _StubFit:
    """Minimal stand-in for DCFit.rates for feature tests."""

    def rates(self, home, away, at_home):
        return (1.5 if at_home else 1.3), 1.1


def _history():
    rows = [
        ("2024-01-01", "A", "B", 2, 0),
        ("2024-01-08", "C", "A", 1, 1),
        ("2024-01-20", "A", "D", 3, 1),
    ]
    df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
    df["date"] = pd.to_datetime(df["date"])
    df["home_at_home"] = True
    return df


def test_rest_days_capped_and_pointintime():
    h = _history()
    # As of 2024-01-22, A last played 2024-01-20 -> 2 days rest.
    assert rest_days(h, "A", pd.Timestamp("2024-01-22")) == 2.0
    # Team with no prior match -> treated as fully rested (cap).
    assert rest_days(h, "Z", pd.Timestamp("2024-01-22")) == 30.0


def test_match_features_orientation_and_leakage():
    h = _history()
    asof = pd.Timestamp("2024-01-22")
    feats = match_features(h, _StubFit(), "A", "B", asof)
    assert set(feats) == {"rest_adv", "form_home", "form_away"}
    # B never appears before asof except as A's opponent on 2024-01-01.
    # A overperformed its rate (scored 2,1,3 vs ~1.3-1.5 expectation) -> positive form.
    assert feats["form_home"] > 0
    # A future match must not change features computed as of an earlier date.
    future = pd.concat([h, pd.DataFrame([{
        "date": pd.Timestamp("2024-02-01"), "home": "A", "away": "B",
        "hg": 9, "ag": 0, "home_at_home": True}])], ignore_index=True)
    assert match_features(future, _StubFit(), "A", "B", asof) == feats


def test_glm_sign_constraint_and_offset():
    rng = np.random.default_rng(1)
    n = 800
    offset = rng.normal(0.1, 0.3, n)
    rest_adv = rng.normal(0, 2, n)
    form = rng.normal(0, 0.3, n)
    # True effect: rest helps (+0.05/day), form persists (+0.4).
    mu = np.exp(offset + 0.05 * rest_adv + 0.4 * form)
    goals = rng.poisson(mu).astype(float)
    sf = pd.DataFrame({"goals": goals, "offset": offset, "rest_adv": rest_adv,
                       "form": form, "tournament": "t"})
    fit = fit_glm(sf, ridge=1.0)
    assert fit.coef["rest_adv"] >= 0.0          # box constraint honored
    assert fit.coef["form"] > 0.2                # recovers positive persistence


def test_glm_adjust_moves_rates_symmetrically():
    fit = GLMFit(coef={"rest_adv": 0.02, "form": 0.1}, ridge=50.0, n_obs=100)
    feats = {"rest_adv": 3.0, "form_home": 0.5, "form_away": -0.5}
    lam, mu = fit.adjust(1.4, 1.1, feats)
    # Home: more rest (+) and good form (+) -> rate up. Away: less rest, bad form -> down.
    assert lam > 1.4
    assert mu < 1.1


def test_loto_cv_runs_on_two_tournaments():
    rng = np.random.default_rng(2)
    frames = []
    for t in ("t1", "t2"):
        n = 200
        offset = rng.normal(0.1, 0.3, n)
        frames.append(pd.DataFrame({
            "goals": rng.poisson(np.exp(offset)).astype(float), "offset": offset,
            "rest_adv": rng.normal(0, 2, n), "form": rng.normal(0, 0.3, n), "tournament": t}))
    sf = pd.concat(frames, ignore_index=True)
    cv = loto_cv(sf, ridge=50.0)
    assert cv["n_obs"] == 400
    assert "deviance_delta_pct" in cv
