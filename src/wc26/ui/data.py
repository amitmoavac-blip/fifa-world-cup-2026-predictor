"""Read-only views the web app renders: artifact, evaluation, history, health.

Thin wrappers over the artifact and the existing reports/snapshots; all numbers
come from the validated engine outputs, never recomputed here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wc26.backtest.metrics import aggregate
from wc26.backtest.walkforward import load_backtest_config
from wc26.paths import processed_dir, raw_dir, reports_dir, snapshots_dir
from wc26.predict.snapshot import model_version
from wc26.ui.build import ARTIFACT, build_artifact

SYSTEM_LABELS = {
    "dc_model": "Dixon-Coles model",
    "elo_poisson": "Elo-Poisson baseline",
    "constant_modal": "Constant modal score",
}


def load_artifact(rebuild_if_missing: bool = True) -> dict:
    p = processed_dir() / ARTIFACT
    if not p.exists():
        if rebuild_if_missing:
            return build_artifact()
        raise FileNotFoundError("UI artifact missing; run `wc26 build-ui-data`")
    return json.loads(p.read_text())


def match_by_id(artifact: dict, match_id: str) -> dict | None:
    return next((m for m in artifact["matches"] if m["id"] == match_id), None)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _slice_table(records: pd.DataFrame) -> list[dict]:
    rows = []
    for sys_key, label in SYSTEM_LABELS.items():
        sub = records[records.system == sys_key]
        if sub.empty:
            continue
        a = aggregate(sub)
        rows.append({
            "label": label, "n": a["n"], "exact": a["exact_hit_rate"],
            "logloss": a["logloss"], "total_mae": a["total_goals_mae"],
            "team_mae": round((a["home_goals_mae"] + a["away_goals_mae"]) / 2, 3),
            "outcome": a["outcome_acc"],
        })
    return rows


def evaluation_summary() -> dict:
    path = reports_dir() / "backtest_records.parquet"
    if not path.exists():
        return {"available": False}
    rec = pd.read_parquet(path)
    bt = load_backtest_config()
    sel, hold = set(bt["selection_tournaments"]), set(bt["holdout_tournaments"])
    model = rec[rec.system == "dc_model"]

    slices = {
        "All tournaments": _slice_table(rec),
        "Holdout (untouched by tuning)": _slice_table(rec[rec.tournament.isin(hold)]),
        "Selection (tuning) set": _slice_table(rec[rec.tournament.isin(sel)]),
        "Group stage": _slice_table(rec[~rec.knockout]),
        "Knockout (120-minute target)": _slice_table(rec[rec.knockout]),
    }

    by_tournament = []
    for t, sub in model.groupby("tournament"):
        a = aggregate(sub)
        by_tournament.append({"tournament": t, "n": a["n"], "exact": a["exact_hit_rate"],
                              "logloss": a["logloss"], "outcome": a["outcome_acc"]})

    conf = []
    for label in ("high", "medium", "low"):
        s = model[model.confidence == label]
        if len(s):
            conf.append({"label": label, "n": int(len(s)), "hit": round(s.exact_hit.mean(), 4)})

    return {
        "available": True,
        "slices": slices,
        "by_tournament": by_tournament,
        "confidence": conf,
        "experimental": _experimental_layers(),
        "note": ("Exact-score prediction is inherently hard: the best constant guess "
                 "(historical modal score) lands ~10% here. A common rule of thumb puts "
                 "strong models/markets in the low-to-mid teens, but we have no market-odds "
                 "data in-repo to verify it — treat it as context, not a measured ceiling."),
    }


def _experimental_layers() -> list[dict]:
    layers = [{
        "name": "Layer-2 context GLM (rest, form-residual)",
        "status": "disabled",
        "finding": "Leave-one-tournament-out CV: -0.05% per-side deviance. Goal-based "
                   "context features add nothing for international football.",
    }]
    tb = reports_dir() / "tier_b_records.parquet"
    if tb.exists():
        df = pd.read_parquet(tb)
        av = df[df.xg_available == 1]
        dc_hit = av[av.system == "dc"].exact_hit.mean()
        xg_hit = av[av.system == "dc_xg"].exact_hit.mean()
        dc_ll = av[av.system == "dc"].logloss.mean()
        xg_ll = av[av.system == "dc_xg"].logloss.mean()
        layers.append({
            "name": "Tier-B xG-form (StatsBomb, 6 tournaments)",
            "status": "disabled",
            "finding": f"LOTO CV on {len(av[av.system=='dc'])} xG-available matches: exact-hit "
                       f"{dc_hit:.1%} -> {xg_hit:.1%}, log loss {dc_ll:.3f} -> {xg_ll:.3f}. "
                       f"Correct coefficient signs but no improvement to the exact-score "
                       f"distribution over the short within-tournament window.",
        })
    return layers


# --------------------------------------------------------------------------- #
# Per-match prediction history
# --------------------------------------------------------------------------- #
def match_history(home: str, away: str) -> list[dict]:
    path = snapshots_dir() / "predictions.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("home") == home and r.get("away") == away:
            out.append({
                "epoch": r.get("epoch"), "snapshot_ts": r.get("snapshot_ts"),
                "asof": r.get("asof"), "modal_score": r.get("modal_score"),
                "confidence": r.get("confidence"), "modal_prob": r.get("modal_prob"),
                "model_version": r.get("model_version"),
            })
    out.sort(key=lambda e: e["snapshot_ts"] or "")
    return out


# --------------------------------------------------------------------------- #
# Data health
# --------------------------------------------------------------------------- #
def _mtime(p: Path) -> str | None:
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def data_health() -> dict:
    raw = {
        "results.csv": _mtime(raw_dir() / "results.csv"),
        "shootouts.csv": _mtime(raw_dir() / "shootouts.csv"),
        "goalscorers.csv": _mtime(raw_dir() / "goalscorers.csv"),
        "worldcup2026.json": _mtime(raw_dir() / "worldcup2026.json"),
    }
    sources = [
        {"name": "martj42 international results (CC0)", "file": "results.csv",
         "last_sync": raw["results.csv"], "status": "ok" if raw["results.csv"] else "missing"},
        {"name": "openfootball WC2026 schedule (CC0)", "file": "worldcup2026.json",
         "last_sync": raw["worldcup2026.json"], "status": "ok" if raw["worldcup2026.json"] else "missing"},
        {"name": "StatsBomb open xG (research-only)", "file": "statsbomb_xg.parquet",
         "last_sync": _mtime(processed_dir() / "statsbomb_xg.parquet"),
         "status": "ok" if (processed_dir() / "statsbomb_xg.parquet").exists() else "not ingested (Tier-B only)"},
    ]

    snap_path = snapshots_dir() / "predictions.jsonl"
    n_snap, last_snap = 0, None
    if snap_path.exists():
        lines = [l for l in snap_path.read_text().splitlines() if l.strip()]
        n_snap = len(lines)
        if lines:
            last_snap = json.loads(lines[-1]).get("snapshot_ts")

    art_path = processed_dir() / ARTIFACT
    artifact_built = None
    if art_path.exists():
        artifact_built = json.loads(art_path.read_text()).get("build_ts")

    warnings = []
    now = datetime.now(timezone.utc)
    for name, ts in raw.items():
        if ts is None:
            warnings.append(f"{name} not downloaded — run `wc26 ingest --force`.")
        else:
            age_days = (now - datetime.fromisoformat(ts)).days
            if age_days > 3:
                warnings.append(f"{name} is {age_days} days old — re-sync before relying on results.")

    return {
        "model_version": model_version(),
        "mode": "historical / cached (free sources only)",
        "live_data_configured": False,
        "lineup_data_configured": False,
        "calibration_active": (processed_dir() / "calibration.json").exists(),
        "glm_active": (processed_dir() / "glm.json").exists(),
        "raw": raw,
        "sources": sources,
        "matches_processed": _mtime(processed_dir() / "matches.parquet"),
        "artifact_built": artifact_built,
        "snapshots": n_snap,
        "last_snapshot": last_snap,
        "warnings": warnings,
    }
