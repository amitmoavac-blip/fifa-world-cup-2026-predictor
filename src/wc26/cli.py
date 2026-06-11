"""wc26 command-line interface: ingest | fit | predict | backtest | report."""
from __future__ import annotations

from datetime import date as _date

import pandas as pd
import typer
import yaml

from wc26.paths import configs_dir, reports_dir

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _resolve_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(_date.today()) if value == "today" else pd.Timestamp(value)


@app.command()
def ingest(force: bool = typer.Option(False, help="re-download raw files")):
    """Download raw sources and build processed parquet tables."""
    from wc26.data import ingest as ing

    counts = ing.run(force_download=force)
    typer.echo(f"ingested: {counts}")


@app.command()
def fit(asof: str = typer.Option("today")):
    """Fit Layer-1 ratings as of a date and print the top of the table."""
    from wc26.backtest.walkforward import dc_config_from_yaml, load_model_config
    from wc26.data import ingest as ing
    from wc26.ratings import dixon_coles as dc

    ts = _resolve_date(asof)
    matches = ing.load_matches()
    f = dc.fit(matches, asof=ts, cfg=dc_config_from_yaml(load_model_config()))
    strength = {t: f.attack[t] + f.defense[t] for t in f.attack}
    top = sorted(strength.items(), key=lambda e: -e[1])[:20]
    typer.echo(f"fit asof {ts.date()}  (n={f.n_matches}, eff={f.effective_matches:.0f}, "
               f"home_adv={f.h:.3f}, rho={f.rho:.3f})")
    for i, (team, s) in enumerate(top, 1):
        typer.echo(f"{i:>2}. {team:<22} attack {f.attack[team]:+.3f}  defense {f.defense[team]:+.3f}")


@app.command()
def predict(
    date: str = typer.Option("today", help="match date to predict"),
    snapshot: bool = typer.Option(True, help="append predictions to the snapshot log"),
):
    """Predict scheduled World Cup 2026 fixtures on a date."""
    from wc26.backtest.walkforward import dc_config_from_yaml, load_model_config
    from wc26.data import ingest as ing
    from wc26.predict.engine import predict_match
    from wc26.predict.snapshot import write_snapshot
    from wc26.ratings import dixon_coles as dc

    ts = _resolve_date(date)
    with open(configs_dir() / "wc2026.yaml") as fh:
        wc = yaml.safe_load(fh)
    model_cfg = load_model_config()

    all_matches = ing.load_matches(played_only=False)
    history = all_matches[all_matches.played]
    fixtures = all_matches[
        (~all_matches.played)
        & (all_matches.tournament == wc["tournament"])
        & (all_matches.date == ts)
    ]
    if fixtures.empty:
        typer.echo(f"no scheduled WC 2026 fixtures on {ts.date()} (re-run `wc26 ingest --force`?)")
        raise typer.Exit(1)

    f = dc.fit(history, asof=ts, cfg=dc_config_from_yaml(model_cfg))
    knockout = ts >= pd.Timestamp(wc["knockout_start"])
    for _, m in fixtures.iterrows():
        pred = predict_match(f, m.home, m.away, bool(m.home_at_home), knockout, model_cfg)
        typer.echo(
            f"\n{m.home} vs {m.away}  ({m.city})\n"
            f"  Predicted final score : {m.home} {pred['modal_score'].replace('-', ' - ')} {m.away}\n"
            f"  Confidence            : {pred['confidence']} "
            f"(modal probability {pred['modal_prob']:.1%})\n"
            f"  Why                   : {pred['explanation']}"
        )
        if snapshot:
            write_snapshot(pred, epoch="pre_lineup")


@app.command(name="fit-kappa")
def fit_kappa(knockout_rate: float = typer.Option(2.5, help="assumed regulation rate for ET-bound matches")):
    """Estimate the extra-time intensity multiplier from goal-minute data."""
    from wc26.data import ingest as ing
    from wc26.scoreline.extra_time import fit_kappa as _fit

    res = _fit(ing.load_matches(), ing.load_goalscorers(), ing.load_shootouts(), knockout_rate)
    typer.echo(
        f"ET kappa from {res.n_matches} complete extra-time matches "
        f"({res.et_goals_per_match:.3f} ET goals/match):\n"
        f"  self-normalized (upper) : {res.kappa_self_normalized:.3f}\n"
        f"  population baseline (lower): {res.kappa_population:.3f}\n"
        f"  match-rate synthesis (used): {res.kappa_match_rate:.3f}  "
        f"(assumed knockout rate {res.assumed_knockout_rate})\n"
        f"  -> configs/model.yaml uses scoreline.et_kappa = 0.90"
    )


@app.command()
def backtest(
    tournaments: str = typer.Option("", help="comma-separated keys; default = all configured"),
    tune: bool = typer.Option(True, help="grid-search half-life on the selection set first"),
    refit: str = typer.Option("per_date", help="per_date | tournament"),
):
    """Run the walk-forward backtest and write reports/backtest_report.md."""
    from wc26.backtest import walkforward as wf
    from wc26.backtest.report import write_report

    grid = None
    halflife = None
    if tune:
        halflife, grid = wf.select_halflife()
        typer.echo(f"selected halflife={halflife}")
    keys = [k.strip() for k in tournaments.split(",") if k.strip()] or None
    records = wf.run_backtest(keys=keys, halflife=halflife, refit=refit)
    text = write_report(records, grid)
    typer.echo(text)


@app.command()
def report():
    """Re-render the markdown report from stored backtest records."""
    from wc26.backtest.report import write_report

    records = pd.read_parquet(reports_dir() / "backtest_records.parquet")
    typer.echo(write_report(records))


if __name__ == "__main__":
    app()
