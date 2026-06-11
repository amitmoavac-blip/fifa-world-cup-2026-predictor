"""Layer 4: post-hoc calibration of three structural scalars (plan section 2).

Calibrating the 121-class score simplex with ~64 matches per tournament would
fit noise. Instead we adjust three interpretable scalars on the goal rates and
the draw parameter, fitted by minimizing pooled scoreline log loss on
out-of-sample (walk-forward) predictions:

    m = (log lam + log mu) / 2          # match scoring level
    d = (log lam - log mu) / 2          # team strength gap
    log lam' = m + c_cal + d / T
    log mu'  = m + c_cal - d / T
    rho      = rho_cal                  # global draw adjustment, replaces per-fit rho

  c_cal : total-goals bias (shifts both rates together)
  T     : temperature on the strength gap (T>1 = the model is over-confident)
  rho_cal : residual draw miscalibration
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc26.paths import processed_dir
from wc26.scoreline.matrix import et_convolve, score_matrix

DEFAULT = {"c_cal": 0.0, "T": 1.0, "rho_cal": None}
_RHO_BOUND = 0.18


def apply_calibration(lam: float, mu: float, rho: float, cal: dict | None) -> tuple[float, float, float]:
    if not cal:
        return lam, mu, rho
    log_lam, log_mu = np.log(lam), np.log(mu)
    m = 0.5 * (log_lam + log_mu)
    d = 0.5 * (log_lam - log_mu)
    T = cal.get("T", 1.0)
    c = cal.get("c_cal", 0.0)
    lam2 = float(np.exp(m + c + d / T))
    mu2 = float(np.exp(m + c - d / T))
    rho2 = rho if cal.get("rho_cal") is None else float(cal["rho_cal"])
    return lam2, mu2, rho2


def _pooled_nll(preds: pd.DataFrame, c: float, T: float, rho_cal: float,
                max_goals: int, kappa: float, mu_cap: float) -> float:
    total = 0.0
    for lam0, mu0, rho0, hg, ag, ko in preds[
        ["lam", "mu", "rho", "hg", "ag", "knockout"]
    ].itertuples(index=False):
        lam, mu, rho = apply_calibration(lam0, mu0, rho0, {"c_cal": c, "T": T, "rho_cal": rho_cal})
        lam, mu = min(lam, mu_cap), min(mu, mu_cap)
        m90 = score_matrix(lam, mu, rho, max_goals)
        mat = et_convolve(m90, lam, mu, kappa) if ko else m90
        total += -np.log(max(mat[min(int(hg), max_goals), min(int(ag), max_goals)], 1e-9))
    return total / len(preds)


def fit_calibration(
    preds: pd.DataFrame, max_goals: int = 10, kappa: float = 0.9, mu_cap: float = 4.5
) -> dict:
    """Fit (c_cal, T, rho_cal) by minimizing pooled scoreline NLL.

    `preds` must hold one row per out-of-sample match with columns
    lam, mu, rho, hg, ag, knockout (the raw DC-model rates).
    """
    preds = preds.dropna(subset=["lam", "mu", "rho"]).copy()

    def obj(theta):
        c, logT, rho_cal = theta
        rho_cal = float(np.clip(rho_cal, -_RHO_BOUND, _RHO_BOUND))
        return _pooled_nll(preds, c, float(np.exp(logT)), rho_cal, max_goals, kappa, mu_cap)

    rho0 = float(preds.rho.mean())
    res = minimize(obj, np.array([0.0, 0.0, rho0]), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 300})
    c, logT, rho_cal = res.x
    return {
        "c_cal": round(float(c), 4),
        "T": round(float(np.exp(logT)), 4),
        "rho_cal": round(float(np.clip(rho_cal, -_RHO_BOUND, _RHO_BOUND)), 4),
        "nll_before": round(_pooled_nll(preds, 0.0, 1.0, rho0, max_goals, kappa, mu_cap), 4),
        "nll_after": round(float(res.fun), 4),
        "n_fit": int(len(preds)),
    }


def calibration_path() -> Path:
    return processed_dir() / "calibration.json"


def save_calibration(cal: dict) -> None:
    processed_dir().mkdir(parents=True, exist_ok=True)
    calibration_path().write_text(json.dumps(cal, indent=2))


def load_calibration() -> dict | None:
    p = calibration_path()
    return json.loads(p.read_text()) if p.exists() else None
