"""Walk-forward backtest harness (plan section 9).

For each tournament: train on everything strictly before each match date
(refit per date, warm-started — mirrors live operation, no leakage), predict
that date's matches with the same engine used live, and evaluate against the
recorded final scores (ET-inclusive for knockouts — the exact target).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from wc26.backtest import baselines as bl
from wc26.backtest.metrics import match_record
from wc26.data import ingest
from wc26.data.elo import compute_elo_history
from wc26.paths import configs_dir, reports_dir
from wc26.predict.engine import predict_match
from wc26.ratings import dixon_coles as dc


def load_model_config() -> dict:
    with open(configs_dir() / "model.yaml") as f:
        return yaml.safe_load(f)


def load_backtest_config() -> dict:
    with open(configs_dir() / "backtest.yaml") as f:
        return yaml.safe_load(f)


def load_tournaments() -> pd.DataFrame:
    df = pd.read_csv(configs_dir() / "tournaments.csv", parse_dates=["first_match", "knockout_start"])
    return df.set_index("key")


def dc_config_from_yaml(model_cfg: dict, halflife: float | None = None) -> dc.DCConfig:
    d = model_cfg["dixon_coles"]
    return dc.DCConfig(
        halflife_years=halflife if halflife is not None else d["halflife_years"],
        max_age_years=d["max_age_years"],
        min_weight=d["min_weight"],
        ridge_team_offset=d["ridge_team_offset"],
        ridge_confed=d["ridge_confed"],
        rho_bounds=tuple(d["rho_bounds"]),
        goal_cap_train=d["goal_cap_train"],
        importance_weights=model_cfg["importance_weights"],
    )


def tournament_matches(matches: pd.DataFrame, trow: pd.Series) -> pd.DataFrame:
    window_end = trow.first_match + pd.Timedelta(days=40)
    t = matches[
        (matches.tournament == trow.tournament)
        & (matches.date >= trow.first_match)
        & (matches.date <= window_end)
        & matches.played
    ].copy()
    t["knockout"] = t.date >= trow.knockout_start
    if len(t) != trow.expected_matches:
        raise ValueError(
            f"{trow.name}: found {len(t)} matches, expected {trow.expected_matches} "
            f"(check tournaments.csv dates)"
        )
    if int(t.knockout.sum()) != trow.expected_knockout:
        raise ValueError(
            f"{trow.name}: found {int(t.knockout.sum())} knockout matches, "
            f"expected {trow.expected_knockout}"
        )
    return t.sort_values("date")


def run_tournament(
    matches: pd.DataFrame,  # full played history WITH elo_home/elo_away columns
    trow: pd.Series,
    dc_cfg: dc.DCConfig,
    model_cfg: dict,
    refit: str = "per_date",
) -> pd.DataFrame:
    tmatches = tournament_matches(matches, trow)
    sc = model_cfg["scoreline"]

    const_score = bl.constant_modal_score(matches, trow.first_match)
    elo_model = bl.EloPoisson.fit(matches, trow.first_match)

    records = []
    fit_obj: dc.DCFit | None = None
    for date, day in tmatches.groupby("date"):
        if fit_obj is None or refit == "per_date":
            fit_obj = dc.fit(matches, asof=date, cfg=dc_cfg, warm=fit_obj)
        for _, m in day.iterrows():
            hg, ag = int(m.hg), int(m.ag)
            ko = bool(m.knockout)

            pred = predict_match(fit_obj, m.home, m.away, bool(m.home_at_home), ko, model_cfg)
            rec = {"system": "dc_model", **match_record(pred, hg, ag)}

            elo_pred = bl.elo_poisson_predict(
                elo_model, m.elo_home, m.elo_away, bool(m.home_at_home), ko,
                max_goals=sc["max_goals"], kappa=sc["et_kappa"], mu_cap=sc["mu_cap"],
            )
            rec_elo = {"system": "elo_poisson", **match_record(elo_pred, hg, ag)}

            const_pred = bl.constant_modal_predict(const_score, max_goals=sc["max_goals"])
            rec_const = {"system": "constant_modal", **match_record(const_pred, hg, ag)}

            base = {
                "tournament": trow.name,
                "date": date,
                "home": m.home,
                "away": m.away,
                "hg": hg,
                "ag": ag,
                "knockout": ko,
                "modal_score": pred["modal_score"],
            }
            for r in (rec, rec_elo, rec_const):
                records.append({**base, **{k: v for k, v in r.items() if k != "matrix"}})

    return pd.DataFrame(records)


def run_backtest(
    keys: list[str] | None = None,
    halflife: float | None = None,
    refit: str = "per_date",
    verbose: bool = True,
) -> pd.DataFrame:
    model_cfg = load_model_config()
    bt_cfg = load_backtest_config()
    tcfg = load_tournaments()
    keys = keys or (bt_cfg["selection_tournaments"] + bt_cfg["holdout_tournaments"])

    matches = ingest.load_matches(played_only=True)
    matches = compute_elo_history(matches)

    dc_cfg = dc_config_from_yaml(model_cfg, halflife)
    frames = []
    for key in keys:
        trow = tcfg.loc[key]
        trow.name = key
        df = run_tournament(matches, trow, dc_cfg, model_cfg, refit=refit)
        if verbose:
            model_rows = df[df.system == "dc_model"]
            print(
                f"{key}: n={len(model_rows)} exact={model_rows.exact_hit.mean():.3f} "
                f"logloss={model_rows.logloss.mean():.3f}"
            )
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    reports_dir().mkdir(parents=True, exist_ok=True)
    return out


def select_halflife(verbose: bool = True) -> tuple[float, pd.DataFrame]:
    """Grid-search the decay half-life on the selection tournaments only
    (holdout stays untouched, plan section 9). Single fit per tournament for
    speed; relative ordering is what matters here."""
    bt_cfg = load_backtest_config()
    rows = []
    for hl in bt_cfg["halflife_grid"]:
        df = run_backtest(
            keys=bt_cfg["selection_tournaments"], halflife=hl, refit="tournament", verbose=False
        )
        model = df[df.system == "dc_model"]
        rows.append(
            {"halflife": hl, "logloss": model.logloss.mean(),
             "exact_hit_rate": model.exact_hit.mean(), "n": len(model)}
        )
        if verbose:
            print(f"halflife={hl}: logloss={rows[-1]['logloss']:.4f} "
                  f"exact={rows[-1]['exact_hit_rate']:.4f}")
    grid = pd.DataFrame(rows)
    best = float(grid.loc[grid.logloss.idxmin(), "halflife"])
    return best, grid
