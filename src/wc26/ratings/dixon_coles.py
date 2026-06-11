"""Layer 1: time-decayed Dixon-Coles attack/defense model (plan section 2).

Penalized weighted MLE:
    log lambda = c + A_i - D_j + h * H      (listed-home side)
    log mu     = c + A_j - D_i
    A_i = a_conf(i) + a_offset_i,  D_i = d_conf(i) + d_offset_i
with the Dixon-Coles low-score adjustment tau(x, y; rho), exponential time
decay, importance information-weights, and ridge shrinkage of team offsets
toward confederation means (debutants inherit their confederation's level).

Fitted with L-BFGS-B and analytic gradients; seconds per fit, warm-startable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc26.data.registry import confed_map, CONFEDERATIONS

_TAU_EPS = 1e-8


@dataclass
class DCConfig:
    halflife_years: float = 2.5
    max_age_years: float = 20.0
    min_weight: float = 1e-3
    ridge_team_offset: float = 5.0
    ridge_confed: float = 0.5
    rho_bounds: tuple[float, float] = (-0.18, 0.18)
    goal_cap_train: int = 10
    importance_weights: dict[str, float] = field(
        default_factory=lambda: {
            "world_cup": 1.30,
            "continental_finals": 1.25,
            "qualifier": 1.00,
            "nations_league": 0.70,
            "minor": 0.50,
            "friendly": 0.40,
        }
    )


@dataclass
class DCFit:
    asof: pd.Timestamp
    teams: list[str]
    c: float
    h: float
    rho: float
    attack: dict[str, float]          # total A_i (confed mean + offset)
    defense: dict[str, float]         # total D_i
    a_conf: dict[str, float]
    d_conf: dict[str, float]
    effective_matches: float
    n_matches: int

    def _ad(self, team: str) -> tuple[float, float]:
        if team in self.attack:
            return self.attack[team], self.defense[team]
        conf = confed_map().get(team)
        return self.a_conf.get(conf, 0.0), self.d_conf.get(conf, 0.0)

    def rates(self, home: str, away: str, home_at_home: bool) -> tuple[float, float]:
        """Expected goals (lambda_home, mu_away) before any Layer-2 adjustment."""
        a_h, d_h = self._ad(home)
        a_a, d_a = self._ad(away)
        lam = np.exp(self.c + a_h - d_a + (self.h if home_at_home else 0.0))
        mu = np.exp(self.c + a_a - d_h)
        return float(lam), float(mu)


def _tau_log_and_grads(x, y, lam, mu, rho):
    """log tau and d(log tau)/d(log lam), /d(log mu), /d(rho), vectorized.

    Cells where tau hits the positivity floor get zero gradient (clip mask).
    """
    tau = np.ones_like(lam)
    g_ll = np.zeros_like(lam)   # d log tau / d log lambda
    g_lm = np.zeros_like(lam)
    g_r = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho

    clipped = tau < _TAU_EPS
    tau = np.maximum(tau, _TAU_EPS)
    ok = ~clipped

    g_ll[m00 & ok] = -(lam * mu * rho / tau)[m00 & ok]
    g_lm[m00 & ok] = -(lam * mu * rho / tau)[m00 & ok]
    g_r[m00 & ok] = -(lam * mu / tau)[m00 & ok]

    g_ll[m01 & ok] = (lam * rho / tau)[m01 & ok]
    g_r[m01 & ok] = (lam / tau)[m01 & ok]

    g_lm[m10 & ok] = (mu * rho / tau)[m10 & ok]
    g_r[m10 & ok] = (mu / tau)[m10 & ok]

    g_r[m11 & ok] = (-1.0 / tau)[m11 & ok]

    return np.log(tau), g_ll, g_lm, g_r


def _build_objective(matches: pd.DataFrame, asof: pd.Timestamp, cfg: DCConfig):
    """Prepare data arrays and return (nll_grad, dims) for the weighted MLE.

    Split out of fit() so tests can verify the analytic gradient directly.
    """
    asof = pd.Timestamp(asof)

    df = matches[(matches.date < asof) & matches.played].copy()
    age_years = (asof - df.date).dt.days / 365.25
    df = df[age_years <= cfg.max_age_years]
    age_years = (asof - df.date).dt.days / 365.25

    decay = np.exp2(-age_years.to_numpy() / cfg.halflife_years)
    omega = df.importance.map(cfg.importance_weights).fillna(0.5).to_numpy()
    v = decay * omega
    keep = v >= cfg.min_weight
    df, v = df[keep], v[keep]
    if len(df) < 200:
        raise ValueError(f"only {len(df)} weighted matches before {asof.date()}")

    teams = sorted(set(df.home) | set(df.away))
    t_idx = {t: k for k, t in enumerate(teams)}
    cmap = confed_map()
    confs = list(CONFEDERATIONS)
    c_idx = {c: k for k, c in enumerate(confs)}
    team_conf = np.array([c_idx[cmap[t]] for t in teams])

    hi = df.home.map(t_idx).to_numpy()
    aj = df.away.map(t_idx).to_numpy()
    x = np.minimum(df.hg.to_numpy(dtype=float), cfg.goal_cap_train)
    y = np.minimum(df.ag.to_numpy(dtype=float), cfg.goal_cap_train)
    H = df.home_at_home.to_numpy(dtype=float)

    T, K = len(teams), len(confs)
    # theta = [c, h, rho, a_conf(K), d_conf(K), a_off(T), d_off(T)]
    n_par = 3 + 2 * K + 2 * T

    def unpack(theta):
        c, h, rho = theta[0], theta[1], theta[2]
        ac = theta[3 : 3 + K]
        dc = theta[3 + K : 3 + 2 * K]
        ao = theta[3 + 2 * K : 3 + 2 * K + T]
        do = theta[3 + 2 * K + T :]
        return c, h, rho, ac, dc, ao, do

    def nll_grad(theta):
        c, h, rho, ac, dc, ao, do = unpack(theta)
        A = ac[team_conf] + ao
        D = dc[team_conf] + do
        log_lam = c + A[hi] - D[aj] + h * H
        log_mu = c + A[aj] - D[hi]
        lam, mu = np.exp(log_lam), np.exp(log_mu)

        log_tau, g_ll_tau, g_lm_tau, g_r_tau = _tau_log_and_grads(x, y, lam, mu, rho)
        ll = v * (x * log_lam - lam + y * log_mu - mu + log_tau)

        pen = (
            cfg.ridge_team_offset * (np.sum(ao**2) + np.sum(do**2))
            + cfg.ridge_confed * (np.sum(ac**2) + np.sum(dc**2))
        )
        f = -np.sum(ll) + pen

        # d ll / d log lam, log mu (weighted), then scatter to parameters.
        g_ll = v * (x - lam + g_ll_tau)
        g_lm = v * (y - mu + g_lm_tau)

        g = np.zeros(n_par)
        g[0] = np.sum(g_ll) + np.sum(g_lm)                       # c
        g[1] = np.sum(g_ll * H)                                  # h
        g[2] = np.sum(v * g_r_tau)                               # rho

        gA = np.bincount(hi, g_ll, minlength=T) + np.bincount(aj, g_lm, minlength=T)
        gD = -np.bincount(aj, g_ll, minlength=T) - np.bincount(hi, g_lm, minlength=T)
        g[3 : 3 + K] = np.bincount(team_conf, gA, minlength=K)
        g[3 + K : 3 + 2 * K] = np.bincount(team_conf, gD, minlength=K)
        g[3 + 2 * K : 3 + 2 * K + T] = gA
        g[3 + 2 * K + T :] = gD

        grad = -g
        grad[3 : 3 + K] += 2 * cfg.ridge_confed * ac
        grad[3 + K : 3 + 2 * K] += 2 * cfg.ridge_confed * dc
        grad[3 + 2 * K : 3 + 2 * K + T] += 2 * cfg.ridge_team_offset * ao
        grad[3 + 2 * K + T :] += 2 * cfg.ridge_team_offset * do
        return f, grad

    dims = {
        "teams": teams, "t_idx": t_idx, "confs": confs, "c_idx": c_idx,
        "team_conf": team_conf, "n_par": n_par, "K": K, "T": T,
        "v": v, "x": x, "y": y, "unpack": unpack, "cmap": cmap,
    }
    return nll_grad, dims


def fit(
    matches: pd.DataFrame,
    asof: pd.Timestamp,
    cfg: DCConfig | None = None,
    warm: DCFit | None = None,
) -> DCFit:
    """Fit on played matches strictly before `asof` (point-in-time guarantee)."""
    cfg = cfg or DCConfig()
    asof = pd.Timestamp(asof)
    nll_grad, dims = _build_objective(matches, asof, cfg)
    teams, t_idx = dims["teams"], dims["t_idx"]
    confs, c_idx, team_conf = dims["confs"], dims["c_idx"], dims["team_conf"]
    n_par, K, T = dims["n_par"], dims["K"], dims["T"]
    v, x, y, unpack, cmap = dims["v"], dims["x"], dims["y"], dims["unpack"], dims["cmap"]

    theta0 = np.zeros(n_par)
    theta0[0] = np.log(max(np.average((x + y) / 2.0, weights=v), 0.3))
    if warm is not None:
        theta0[1], theta0[2] = warm.h, warm.rho
        theta0[0] = warm.c
        for k, cf in enumerate(confs):
            theta0[3 + k] = warm.a_conf.get(cf, 0.0)
            theta0[3 + K + k] = warm.d_conf.get(cf, 0.0)
        for t, k in t_idx.items():
            if t in warm.attack:
                cf = cmap[t]
                theta0[3 + 2 * K + k] = warm.attack[t] - warm.a_conf.get(cf, 0.0)
                theta0[3 + 2 * K + T + k] = warm.defense[t] - warm.d_conf.get(cf, 0.0)

    bounds = [(None, None)] * n_par
    bounds[2] = cfg.rho_bounds

    res = minimize(nll_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})
    c, h, rho, ac, dc, ao, do = unpack(res.x)
    A = ac[team_conf] + ao
    D = dc[team_conf] + do

    return DCFit(
        asof=asof,
        teams=teams,
        c=float(c),
        h=float(h),
        rho=float(rho),
        attack={t: float(A[k]) for t, k in t_idx.items()},
        defense={t: float(D[k]) for t, k in t_idx.items()},
        a_conf={cf: float(ac[k]) for cf, k in c_idx.items()},
        d_conf={cf: float(dc[k]) for cf, k in c_idx.items()},
        effective_matches=float(np.sum(v)),
        n_matches=int(len(v)),
    )
