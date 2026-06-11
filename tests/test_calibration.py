import numpy as np
import pandas as pd

from wc26.calibrate.scalars import apply_calibration, fit_calibration


def test_identity_when_none():
    assert apply_calibration(1.5, 1.1, -0.05, None) == (1.5, 1.1, -0.05)


def test_temperature_amplifies_and_compresses_gap():
    lam, mu = 1.8, 1.0
    # T < 1 widens the strength gap; T > 1 narrows it; total scoring preserved at c=0.
    wide_l, wide_m, _ = apply_calibration(lam, mu, 0.0, {"c_cal": 0.0, "T": 0.5, "rho_cal": None})
    narrow_l, narrow_m, _ = apply_calibration(lam, mu, 0.0, {"c_cal": 0.0, "T": 2.0, "rho_cal": None})
    assert wide_l / wide_m > lam / mu > narrow_l / narrow_m
    # c_cal=0 keeps the geometric mean unchanged (temperature only rotates the gap).
    assert np.isclose(np.sqrt(wide_l * wide_m), np.sqrt(lam * mu))
    assert np.isclose(np.sqrt(narrow_l * narrow_m), np.sqrt(lam * mu))


def test_intercept_shifts_total():
    up_l, up_m, _ = apply_calibration(1.4, 1.1, 0.0, {"c_cal": 0.2, "T": 1.0, "rho_cal": None})
    assert up_l > 1.4 and up_m > 1.1
    assert np.isclose(np.log(up_l) - np.log(1.4), 0.2)


def test_rho_override():
    _, _, rho = apply_calibration(1.4, 1.1, -0.03, {"c_cal": 0.0, "T": 1.0, "rho_cal": -0.09})
    assert rho == -0.09


def test_fit_reduces_or_holds_nll_on_synthetic():
    # Build predictions whose true rates are a temperature-distorted version of
    # the model's, so a non-trivial T should help.
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):
        lam = float(np.exp(rng.normal(0.1, 0.4)))
        mu = float(np.exp(rng.normal(0.0, 0.4)))
        # actuals drawn from a sharper gap than the model states (true T = 0.7)
        m = 0.5 * (np.log(lam) + np.log(mu))
        d = 0.5 * (np.log(lam) - np.log(mu))
        tl, tm = np.exp(m + d / 0.7), np.exp(m - d / 0.7)
        rows.append((lam, mu, -0.04, rng.poisson(tl), rng.poisson(tm), False))
    preds = pd.DataFrame(rows, columns=["lam", "mu", "rho", "hg", "ag", "knockout"])
    cal = fit_calibration(preds)
    assert cal["nll_after"] <= cal["nll_before"] + 1e-6
    assert cal["T"] < 1.0  # recovers the sharper-than-stated gap direction
