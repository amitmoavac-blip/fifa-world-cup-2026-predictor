"""Build the read-only prediction artifact the web UI serves.

Reuses the exact `predict` path (Dixon-Coles fit -> predict_match -> incentives)
so the UI never reimplements modelling. Every match becomes a small JSON record;
the artifact carries a build timestamp and model version for honest freshness.

Leakage-honest as-of: an upcoming match is predicted with today's fit; a
completed match is predicted with a fit as of the match date (what the model
knew going in), then compared to the actual result.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pandas as pd
import yaml

from wc26.adjust.glm import load_glm
from wc26.backtest.walkforward import dc_config_from_yaml, load_model_config
from wc26.calibrate.scalars import load_calibration
from wc26.data import ingest
from wc26.features import context
from wc26.features.incentives import match_incentives
from wc26.paths import configs_dir, processed_dir
from wc26.predict.engine import predict_match
from wc26.predict.snapshot import model_version, write_snapshot
from wc26.ratings import dixon_coles as dc

ARTIFACT = "ui_predictions.json"


def slugify(*parts: str) -> str:
    s = "-".join(str(p) for p in parts).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def correctness(predicted: str | None, actual: str | None) -> bool | None:
    """Exact-score correctness; None when the match has no result yet."""
    if actual is None or predicted is None:
        return None
    return predicted == actual


def _schedule_lookup(schedule: pd.DataFrame) -> dict:
    """Order-insensitive (date, {teams}) -> metadata, so home/away ordering
    differences between sources don't drop a fixture."""
    out = {}
    for r in schedule.itertuples(index=False):
        key = (pd.Timestamp(r.date).date(), frozenset({r.home, r.away}))
        out[key] = {"group": r.group, "matchday": r.matchday, "round": r.round,
                    "ground": r.ground, "time": getattr(r, "time", None)}
    return out


def _stage_label(meta: dict, knockout: bool) -> str:
    if meta.get("matchday") and not pd.isna(meta["matchday"]):
        g = meta.get("group") or ""
        return f"{g} · Matchday {int(meta['matchday'])}".strip(" ·")
    return meta.get("round") or ("Knockout" if knockout else "Group stage")


def build_artifact(asof: pd.Timestamp | None = None, write_snapshots: bool = True) -> dict:
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp(datetime.now().date())
    with open(configs_dir() / "wc2026.yaml") as fh:
        wc = yaml.safe_load(fh)
    knockout_start = pd.Timestamp(wc["knockout_start"])
    model_cfg = load_model_config()
    cal = load_calibration()
    use_glm = model_cfg.get("adjust", {}).get("use_glm", False)
    glm = load_glm() if use_glm else None

    all_matches = ingest.load_matches(played_only=False)
    history_all = all_matches[all_matches.played]
    schedule = ingest.load_schedule()
    lookup = _schedule_lookup(schedule)

    wc_fixtures = all_matches[
        (all_matches.tournament == wc["tournament"]) & (all_matches.date >= pd.Timestamp(wc["first_match"]))
    ].sort_values("date")

    fit_cache: dict[pd.Timestamp, dc.DCFit] = {}

    def get_fit(d: pd.Timestamp) -> dc.DCFit:
        if d not in fit_cache:
            hist = all_matches[all_matches.played & (all_matches.date < d)]
            fit_cache[d] = dc.fit(hist, asof=d, cfg=dc_config_from_yaml(model_cfg))
        return fit_cache[d]

    matches = []
    n_predicted = 0
    for m in wc_fixtures.itertuples(index=False):
        played = bool(m.played)
        pred_asof = pd.Timestamp(m.date) if played else asof
        meta = lookup.get((pd.Timestamp(m.date).date(), frozenset({m.home, m.away})), {})
        knockout = pd.Timestamp(m.date) >= knockout_start

        fit = get_fit(pred_asof)
        history = all_matches[all_matches.played & (all_matches.date < pred_asof)]
        feats = context.match_features(history, fit, m.home, m.away, pred_asof) if glm else None
        pred = predict_match(fit, m.home, m.away, bool(m.home_at_home), knockout, model_cfg, cal, glm, feats)

        inc = match_incentives(history, schedule, m.home, m.away, pred_asof)
        explanation = pred["explanation"]
        if inc.get("dead_rubber"):
            explanation += " Dead rubber: both sides' group fate is settled, so intensity may drop."

        actual = f"{int(m.hg)}-{int(m.ag)}" if played else None
        exact_hit = correctness(pred["modal_score"], actual)
        status = "completed" if played else "upcoming"

        rec = {
            "id": slugify(str(pd.Timestamp(m.date).date()), m.home, m.away),
            "date": str(pd.Timestamp(m.date).date()),
            "time": meta.get("time"),
            "home": m.home, "away": m.away, "city": m.city,
            "ground": meta.get("ground") or m.city,
            "group": meta.get("group"),
            "stage": _stage_label(meta, knockout),
            "is_knockout": knockout,
            "status": status,
            "predicted_score": pred["modal_score"],
            "confidence": pred["confidence"],
            "modal_prob": pred["modal_prob"],
            "explanation": explanation,
            "lam": pred["lam"], "mu": pred["mu"],
            "p_home": pred["p_home"], "p_draw": pred["p_draw"], "p_away": pred["p_away"],
            "top_scores": pred["top_scores"],
            "p_draw_90": pred["p_draw_90"], "et_xg": pred["et_xg"],
            "stale": pred["stale"], "mu_capped": pred["mu_capped"],
            "dead_rubber": inc.get("dead_rubber", False),
            "status_home": inc.get("status_home"), "status_away": inc.get("status_away"),
            "pred_asof": str(pred_asof.date()),
            "actual": actual, "exact_hit": exact_hit,
            "prediction_mode": "pre-match",
        }
        matches.append(rec)
        n_predicted += 1
        if write_snapshots and not played:
            snap = {k: pred[k] for k in pred if k != "matrix"}
            snap["explanation"] = explanation
            write_snapshot(snap, epoch="pre_lineup")

    # Knockout placeholder fixtures (teams not yet known): show, don't predict.
    for r in schedule[schedule.matchday.isna()].itertuples(index=False):
        matches.append({
            "id": slugify(str(pd.Timestamp(r.date).date()), r.home, r.away),
            "date": str(pd.Timestamp(r.date).date()), "time": getattr(r, "time", None),
            "home": r.home, "away": r.away, "city": r.ground, "ground": r.ground,
            "group": None, "stage": r.round, "is_knockout": True,
            "status": "pending", "predicted_score": None, "confidence": None,
            "explanation": "Teams to be decided by group results; prediction available once both sides are known.",
            "actual": None, "exact_hit": None, "prediction_mode": "pending",
        })

    artifact = {
        "build_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asof": str(asof.date()),
        "model_version": model_version(),
        "n_predicted": n_predicted,
        "glm_active": use_glm,
        "calibration_active": cal is not None,
        "matches": matches,
    }
    processed_dir().mkdir(parents=True, exist_ok=True)
    (processed_dir() / ARTIFACT).write_text(json.dumps(artifact, indent=2))
    return artifact


def artifact_path():
    return processed_dir() / ARTIFACT
