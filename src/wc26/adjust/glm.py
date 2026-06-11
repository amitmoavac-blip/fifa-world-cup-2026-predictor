"""Layer 2: constrained Poisson GLM correction on the Layer-1 goal rates.

Offset model, one observation per team-side:

    log E[goals_side] = log(mu_L1_side) + b_rest * rest_adv_side + b_form * form_side

fitted by ridge-penalized Poisson MLE with a sign constraint (more rest cannot
lower your own scoring). Heavily regularized: only ~1.4k honest team-side
observations exist, so this stays a 2-parameter nudge, not a flexible learner
(LightGBM is deferred to v2 per the plan). The offset means the GLM can only
*move* the Layer-1 prediction, never replace it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc26.paths import processed_dir

FEATURES = ["rest_adv", "form"]


@dataclass
class GLMFit:
    coef: dict[str, float]
    ridge: float
    n_obs: int

    def adjust(self, lam: float, mu: float, features: dict) -> tuple[float, float]:
        """Apply the correction to (home, away) rates given match features."""
        b_rest = self.coef.get("rest_adv", 0.0)
        b_form = self.coef.get("form", 0.0)
        rest_adv = features.get("rest_adv", 0.0)
        d_home = b_rest * rest_adv + b_form * features.get("form_home", 0.0)
        d_away = b_rest * (-rest_adv) + b_form * features.get("form_away", 0.0)
        return float(lam * np.exp(d_home)), float(mu * np.exp(d_away))


def build_sideframe(records: pd.DataFrame) -> pd.DataFrame:
    """Long format: two rows (home/away side) per backtest match.

    Expects columns lam, mu, hg, ag, rest_adv, form_home, form_away, tournament.
    """
    r = records.dropna(subset=["lam", "mu", "rest_adv", "form_home", "form_away"]).copy()
    home = pd.DataFrame({
        "goals": r.hg.astype(float), "offset": np.log(r.lam),
        "rest_adv": r.rest_adv, "form": r.form_home, "tournament": r.tournament,
    })
    away = pd.DataFrame({
        "goals": r.ag.astype(float), "offset": np.log(r.mu),
        "rest_adv": -r.rest_adv, "form": r.form_away, "tournament": r.tournament,
    })
    return pd.concat([home, away], ignore_index=True)


def fit_glm(sideframe: pd.DataFrame, ridge: float = 50.0) -> GLMFit:
    g = sideframe.goals.to_numpy()
    off = sideframe.offset.to_numpy()
    X = sideframe[FEATURES].to_numpy()
    n, p = X.shape

    def nll_grad(theta):
        eta = off + X @ theta
        lam = np.exp(eta)
        f = -np.sum(g * eta - lam) + ridge * np.sum(theta**2)
        grad = -X.T @ (g - lam) + 2 * ridge * theta
        return f, grad

    bounds = [(0.0, None), (None, None)]  # rest advantage cannot reduce own scoring
    res = minimize(nll_grad, np.zeros(p), jac=True, method="L-BFGS-B", bounds=bounds)
    return GLMFit(coef=dict(zip(FEATURES, res.x.tolist())), ridge=ridge, n_obs=int(n))


def _poisson_deviance(g: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-9)
    ratio = np.divide(g, mu, out=np.ones_like(mu), where=g > 0)
    term = np.where(g > 0, g * np.log(ratio), 0.0) - (g - mu)
    return float(2 * np.sum(term))


def loto_cv(sideframe: pd.DataFrame, ridge: float = 50.0) -> dict:
    """Leave-one-tournament-out CV: GLM vs the offset-only Layer-1 baseline.

    Honest measure of what the Tier-A features buy, with no leakage across
    tournaments (the GLM never trains on the tournament it scores).
    """
    dev_l1 = dev_glm = mae_l1 = mae_glm = 0.0
    n = 0
    for t in sideframe.tournament.unique():
        tr = sideframe[sideframe.tournament != t]
        te = sideframe[sideframe.tournament == t]
        f = fit_glm(tr, ridge=ridge)
        beta = np.array([f.coef[k] for k in FEATURES])
        mu_l1 = np.exp(te.offset.to_numpy())
        mu_glm = np.exp(te.offset.to_numpy() + te[FEATURES].to_numpy() @ beta)
        g = te.goals.to_numpy()
        dev_l1 += _poisson_deviance(g, mu_l1)
        dev_glm += _poisson_deviance(g, mu_glm)
        mae_l1 += np.abs(g - mu_l1).sum()
        mae_glm += np.abs(g - mu_glm).sum()
        n += len(te)
    return {
        "n_obs": n,
        "deviance_l1": round(dev_l1 / n, 4),
        "deviance_glm": round(dev_glm / n, 4),
        "deviance_delta_pct": round(100 * (dev_glm - dev_l1) / dev_l1, 3),
        "mae_l1": round(mae_l1 / n, 4),
        "mae_glm": round(mae_glm / n, 4),
    }


def glm_path() -> Path:
    return processed_dir() / "glm.json"


def save_glm(fit: GLMFit) -> None:
    processed_dir().mkdir(parents=True, exist_ok=True)
    glm_path().write_text(json.dumps({"coef": fit.coef, "ridge": fit.ridge, "n_obs": fit.n_obs}, indent=2))


def load_glm() -> GLMFit | None:
    p = glm_path()
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return GLMFit(coef=d["coef"], ridge=d["ridge"], n_obs=d["n_obs"])
