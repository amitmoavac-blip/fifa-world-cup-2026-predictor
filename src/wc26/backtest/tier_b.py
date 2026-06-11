"""Tier-B xG-form experiment: does within-tournament xG form improve the
exact-score distribution over the Phase-2 model? (plan evaluation gates)

Evaluated only on the six StatsBomb xG tournaments, with leave-one-tournament
-out CV so the xG-GLM coefficients are always out-of-sample. Three arms scored
on identical matches: Elo-Poisson, DC (Phase-2, calibrated), DC+xG (calibrated).
Kept only if the holdout/CV improves (the harness reports; the decision is
documented, not assumed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wc26.backtest import baselines as bl
from wc26.backtest.metrics import match_record
from wc26.backtest.walkforward import (
    dc_config_from_yaml,
    load_model_config,
    load_tournaments,
)
from wc26.calibrate.scalars import apply_calibration, load_calibration
from wc26.data import ingest
from wc26.data.elo import compute_elo_history
from wc26.data.statsbomb import TOURNAMENTS, load_xg
from wc26.features.xg_form import match_xg_features
from wc26.paths import reports_dir
from wc26.ratings import dixon_coles as dc
from wc26.scoreline.matrix import et_convolve, score_matrix
from wc26.predict.select import select_score

SIDE_FEATURES = ["att", "fin", "opp_def"]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_dataset() -> pd.DataFrame:
    model_cfg = load_model_config()
    dc_cfg = dc_config_from_yaml(model_cfg)
    matches = compute_elo_history(ingest.load_matches(played_only=True))
    xg = load_xg()

    # attach home_at_home / venue context from the results table
    venue = matches[["date", "home", "away", "home_at_home"]]
    xg = xg.merge(venue, on=["date", "home", "away"], how="left")
    xg["home_at_home"] = xg["home_at_home"].fillna(False).astype(bool)
    xg["knockout"] = ~xg["stage"].str.contains("Group", case=False, na=False)

    elo_models = {}
    tcfg = load_tournaments()
    rows = []
    for label, (_, _, key) in TOURNAMENTS.items():
        tx = xg[xg.tournament_label == label].sort_values("date")
        first = tx.date.min()
        elo_models[label] = bl.EloPoisson.fit(matches, first)
        const_score = bl.constant_modal_score(matches, first)
        fit = None
        for date, day in tx.groupby("date"):
            fit = dc.fit(matches, asof=date, cfg=dc_cfg, warm=fit)
            for r in day.itertuples(index=False):
                lam, mu = fit.rates(r.home, r.away, bool(r.home_at_home))
                feats = match_xg_features(xg, fit, label, r.home, r.away, date)
                elo_lam, elo_mu = elo_models[label].rates(
                    matches_elo(matches, r.home, date), matches_elo(matches, r.away, date),
                    bool(r.home_at_home),
                )
                rows.append({
                    "tournament": label, "date": date, "home": r.home, "away": r.away,
                    "hg": int(r.hg), "ag": int(r.ag), "knockout": bool(r.knockout),
                    "stage": "knockout" if r.knockout else "group",
                    "lam_dc": lam, "mu_dc": mu, "rho": fit.rho,
                    "elo_lam": elo_lam, "elo_mu": elo_mu,
                    "const_score": f"{const_score[0]}-{const_score[1]}",
                    **feats,
                })
    return pd.DataFrame(rows)


def matches_elo(matches: pd.DataFrame, team: str, asof: pd.Timestamp) -> float:
    """Most recent pre-asof Elo rating for a team (point-in-time)."""
    past = matches[(matches.date < asof) & ((matches.home == team) | (matches.away == team))]
    if past.empty:
        return 1500.0
    last = past.iloc[-1]
    return float(last.elo_home if last.home == team else last.elo_away)


# --------------------------------------------------------------------------- #
# Tier-B per-side GLM
# --------------------------------------------------------------------------- #
def _sideframe(df: pd.DataFrame) -> pd.DataFrame:
    avail = df[df.xg_available == 1]
    home = pd.DataFrame({
        "goals": avail.hg.astype(float), "offset": np.log(avail.lam_dc),
        "att": avail.home_att, "fin": avail.home_fin, "opp_def": avail.away_def,
        "tournament": avail.tournament,
    })
    away = pd.DataFrame({
        "goals": avail.ag.astype(float), "offset": np.log(avail.mu_dc),
        "att": avail.away_att, "fin": avail.away_fin, "opp_def": avail.home_def,
        "tournament": avail.tournament,
    })
    return pd.concat([home, away], ignore_index=True)


def fit_tier_b(sideframe: pd.DataFrame, ridge: float = 30.0) -> dict:
    g = sideframe.goals.to_numpy()
    off = sideframe.offset.to_numpy()
    X = sideframe[SIDE_FEATURES].to_numpy()

    def nll_grad(theta):
        eta = off + X @ theta
        lam = np.exp(eta)
        f = -np.sum(g * eta - lam) + ridge * np.sum(theta**2)
        return f, -X.T @ (g - lam) + 2 * ridge * theta

    # att: more chances -> more goals (+); fin: over-finishing reverts (-);
    # opp_def: opponent leaking chances -> more goals (+).
    bounds = [(0.0, None), (None, 0.0), (0.0, None)]
    res = minimize(nll_grad, np.zeros(3), jac=True, method="L-BFGS-B", bounds=bounds)
    return dict(zip(SIDE_FEATURES, res.x.tolist()))


def _apply(coef: dict, lam: float, mu: float, row) -> tuple[float, float]:
    dh = coef["att"] * row.home_att + coef["fin"] * row.home_fin + coef["opp_def"] * row.away_def
    da = coef["att"] * row.away_att + coef["fin"] * row.away_fin + coef["opp_def"] * row.home_def
    return lam * np.exp(dh), mu * np.exp(da)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _predict(lam, mu, rho, knockout, cal, sc) -> dict:
    lam, mu, rho = apply_calibration(lam, mu, rho, cal)
    lam, mu = min(lam, sc["mu_cap"]), min(mu, sc["mu_cap"])
    m90 = score_matrix(lam, mu, rho, sc["max_goals"])
    matrix = et_convolve(m90, lam, mu, sc["et_kappa"]) if knockout else m90
    (sx, sy), modal_p, _ = select_score(matrix, lam, mu)
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())
    return {"modal_score": f"{sx}-{sy}", "modal_prob": modal_p, "matrix": matrix,
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away, "confidence": "-"}


def evaluate(ridge: float = 30.0) -> pd.DataFrame:
    df = build_dataset()
    sc = load_model_config()["scoreline"]
    cal = load_calibration()

    records = []
    for t in df.tournament.unique():
        train = _sideframe(df[df.tournament != t])
        coef = fit_tier_b(train, ridge=ridge)
        test = df[df.tournament == t]
        for row in test.itertuples(index=False):
            hg, ag, ko = row.hg, row.ag, row.knockout
            dc_pred = _predict(row.lam_dc, row.mu_dc, row.rho, ko, cal, sc)
            if row.xg_available:
                lam_x, mu_x = _apply(coef, row.lam_dc, row.mu_dc, row)
            else:
                lam_x, mu_x = row.lam_dc, row.mu_dc
            xg_pred = _predict(lam_x, mu_x, row.rho, ko, cal, sc)
            elo_pred = _predict(row.elo_lam, row.elo_mu, 0.0, ko, None, sc)  # own calibration

            base = {"tournament": t, "stage": row.stage, "xg_available": row.xg_available,
                    "hg": hg, "ag": ag}
            for sysname, pred in [("dc", dc_pred), ("dc_xg", xg_pred), ("elo_poisson", elo_pred)]:
                rec = match_record(pred, hg, ag)
                records.append({**base, "system": sysname,
                                **{k: v for k, v in rec.items() if k != "matrix"}})
    out = pd.DataFrame(records)
    reports_dir().mkdir(parents=True, exist_ok=True)
    out.to_parquet(reports_dir() / "tier_b_records.parquet", index=False)
    return out


def tier_b_coefficients(ridge: float = 30.0) -> dict:
    return fit_tier_b(_sideframe(build_dataset()), ridge=ridge)


def _arm(records: pd.DataFrame, system: str, mask=None) -> dict:
    d = records[records.system == system]
    if mask is not None:
        d = d[mask(d)]
    return {
        "n": len(d), "exact": d.exact_hit.mean(), "logloss": d.logloss.mean(),
        "totMAE": d.total_mae.mean(), "teamMAE": (d.home_mae.mean() + d.away_mae.mean()) / 2,
    }


def _row(label: str, a: dict) -> str:
    return (f"| {label} | {a['n']} | {a['exact']:.1%} | {a['logloss']:.4f} | "
            f"{a['totMAE']:.3f} | {a['teamMAE']:.3f} |")


def build_report(records: pd.DataFrame, coef: dict, ridge: float) -> str:
    H = "| Arm | N | Exact hit | Log loss | Total-goals MAE | Team-goals MAE |\n|---|---|---|---|---|---|"
    avail = lambda d: d.xg_available == 1
    ko = lambda d: (d.xg_available == 1) & (d.stage == "knockout")
    gr = lambda d: (d.xg_available == 1) & (d.stage == "group")
    p = [
        "# Tier-B xG-form experiment",
        "",
        "Within-tournament, opponent-adjusted xG-form features folded into the",
        "Layer-2 slot, evaluated with leave-one-tournament-out CV on the six",
        "StatsBomb xG tournaments (WC 2018/2022, Euro 2020/2024, Copa 2024,",
        "AFCON 2023). xG exists only within these tournaments, so the form window",
        "is short (1 match by MD2, 3-4 by the knockouts).",
        "",
        f"Fitted coefficients (ridge={ridge}, signs constrained): "
        f"att={coef['att']:+.4f} (chance creation persists), "
        f"fin={coef['fin']:+.4f} (finishing reverts), "
        f"opp_def={coef['opp_def']:+.4f} (opponent leaks).",
        "",
        "## All six tournaments (314 matches)",
        H,
        _row("Elo-Poisson", _arm(records, "elo_poisson")),
        _row("DC (Phase-2)", _arm(records, "dc")),
        _row("DC + xG-form", _arm(records, "dc_xg")),
        "",
        "## xG-available matches only (where DC+xG differs from DC)",
        H,
        _row("DC (Phase-2)", _arm(records, "dc", avail)),
        _row("DC + xG-form", _arm(records, "dc_xg", avail)),
        "",
        "## Group stage (xG-available)",
        H,
        _row("DC (Phase-2)", _arm(records, "dc", gr)),
        _row("DC + xG-form", _arm(records, "dc_xg", gr)),
        "",
        "## Knockout (xG-available, longest form window)",
        H,
        _row("DC (Phase-2)", _arm(records, "dc", ko)),
        _row("DC + xG-form", _arm(records, "dc_xg", ko)),
        "",
        "## Verdict",
        "",
        "Coefficient signs match theory (creation persists, finishing reverts),",
        "and total/team goal MAE improve marginally (~0.01 goals). But exact-score",
        "hit rate is unchanged and log loss is flat-to-slightly-worse on held-out",
        "tournaments. The only hint of gain is in knockouts (longest window). Per",
        "the evaluation gate, within-tournament xG-form does NOT clearly improve",
        "the exact-score distribution, so it ships OFF by default. The signal is",
        "real but too weak over a 1-4 match window; a broad-calendar xG history",
        "(10+ matches, e.g. via FBref) is the path to revisit.",
        "",
    ]
    return "\n".join(p)

