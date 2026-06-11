"""Markdown backtest report: model vs baselines, sliced honestly."""
from __future__ import annotations

import pandas as pd

from wc26.backtest.metrics import aggregate
from wc26.backtest.walkforward import load_backtest_config
from wc26.paths import reports_dir

SYSTEM_LABELS = {
    "dc_model": "Dixon-Coles model",
    "elo_poisson": "Elo-Poisson baseline",
    "constant_modal": "Constant modal score",
}


def _table(records: pd.DataFrame) -> str:
    rows = []
    for sys_key, label in SYSTEM_LABELS.items():
        sub = records[records.system == sys_key]
        if sub.empty:
            continue
        a = aggregate(sub)
        rows.append(
            f"| {label} | {a['n']} | {a['exact_hit_rate']:.1%} | {a['logloss']:.3f} | "
            f"{a['rps']:.4f} | {a['outcome_acc']:.1%} | {a['total_goals_mae']:.2f} | {a['gd_mae']:.2f} |"
        )
    header = (
        "| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    return header + "\n" + "\n".join(rows)


def _calibration_table(model: pd.DataFrame) -> str:
    bins = pd.cut(model.modal_prob, [0, 0.07, 0.09, 0.11, 0.13, 1.0])
    g = model.groupby(bins, observed=True).agg(n=("exact_hit", "size"), hit=("exact_hit", "mean"),
                                               predicted=("modal_prob", "mean"))
    lines = ["| Modal-prob bin | N | Predicted | Realized hit rate |", "|---|---|---|---|"]
    for idx, r in g.iterrows():
        lines.append(f"| {idx} | {int(r.n)} | {r.predicted:.1%} | {r.hit:.1%} |")
    return "\n".join(lines)


def build_report(records: pd.DataFrame, halflife_grid: pd.DataFrame | None = None) -> str:
    bt_cfg = load_backtest_config()
    sel, hold = set(bt_cfg["selection_tournaments"]), set(bt_cfg["holdout_tournaments"])
    model = records[records.system == "dc_model"]

    parts = [
        "# Walk-forward backtest report",
        "",
        "Target: exact final score (90' group stage; 120' knockouts, shootouts excluded).",
        "Exact-score prediction has a hard ceiling — betting markets hit ~11-12%. Read",
        "every number against the baselines below, not against intuition (plan section 9).",
        "",
        "## All tournaments",
        _table(records),
        "",
        "## Selection set (hyperparameters tuned here)",
        _table(records[records.tournament.isin(sel)]),
        "",
        "## Holdout set (untouched by tuning)",
        _table(records[records.tournament.isin(hold)]),
        "",
        "## Group stage only",
        _table(records[~records.knockout]),
        "",
        "## Knockout only (120-minute target)",
        _table(records[records.knockout]),
        "",
        "## By tournament (model only)",
        "| Tournament | N | Exact hit | Logloss | Outcome acc |",
        "|---|---|---|---|---|",
    ]
    for t, sub in model.groupby("tournament"):
        a = aggregate(sub)
        parts.append(f"| {t} | {a['n']} | {a['exact_hit_rate']:.1%} | {a['logloss']:.3f} | {a['outcome_acc']:.1%} |")

    parts += ["", "## Modal-probability calibration (model)", _calibration_table(model)]

    conf = model.groupby("confidence").agg(n=("exact_hit", "size"), hit=("exact_hit", "mean"))
    parts += ["", "## Confidence label vs realized exact-hit rate",
              "| Label | N | Hit rate |", "|---|---|---|"]
    for label in ("high", "medium", "low"):
        if label in conf.index:
            r = conf.loc[label]
            parts.append(f"| {label} | {int(r.n)} | {r.hit:.1%} |")

    if halflife_grid is not None:
        parts += ["", "## Half-life grid (selection set, single-fit mode)",
                  "| Half-life (years) | Logloss | Exact hit |", "|---|---|---|"]
        for _, r in halflife_grid.iterrows():
            parts.append(f"| {r.halflife} | {r.logloss:.4f} | {r.exact_hit_rate:.1%} |")

    return "\n".join(parts) + "\n"


def write_report(records: pd.DataFrame, halflife_grid: pd.DataFrame | None = None) -> str:
    reports_dir().mkdir(parents=True, exist_ok=True)
    records.drop(columns=[c for c in records.columns if c == "matrix"], errors="ignore").to_parquet(
        reports_dir() / "backtest_records.parquet", index=False
    )
    text = build_report(records, halflife_grid)
    (reports_dir() / "backtest_report.md").write_text(text)
    return text
