"""Ingestion: download raw sources, clean, canonicalize, write parquet.

Output schema (data/processed/matches.parquet), one row per match:
    date (datetime), home, away, hg, ag (nullable Int for future fixtures),
    tournament, importance (class string), city, country, neutral (bool),
    home_at_home (bool: home team plays in its own country, not neutral),
    played (bool)

Scores in results.csv include extra time and exclude penalty shootouts —
exactly the prediction target (plan section 1). shootouts.parquet identifies
matches decided on penalties (whose recorded score is therefore a draw).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from wc26.data.importance import importance_class
from wc26.data.registry import canonical, is_registered
from wc26.paths import processed_dir, raw_dir

MARTJ42 = "https://raw.githubusercontent.com/martj42/international_results/master/"
OPENFOOTBALL_2026 = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)
RAW_FILES = {
    "results.csv": MARTJ42 + "results.csv",
    "shootouts.csv": MARTJ42 + "shootouts.csv",
    "goalscorers.csv": MARTJ42 + "goalscorers.csv",
    "worldcup2026.json": OPENFOOTBALL_2026,
}


def download(force: bool = False) -> None:
    raw_dir().mkdir(parents=True, exist_ok=True)
    for fname, url in RAW_FILES.items():
        dest = raw_dir() / fname
        if dest.exists() and not force:
            continue
        urllib.request.urlretrieve(url, dest)


def _clean_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("home_team", "away_team"):
        df[col] = df[col].map(canonical)
    # FIFA-member (plus historical/associate) registry filter: drops CONIFA
    # sides, Island Games teams etc., which never meet FIFA teams.
    mask = df.home_team.map(is_registered) & df.away_team.map(is_registered)
    df = df[mask].copy()

    df = df.rename(
        columns={"home_team": "home", "away_team": "away", "home_score": "hg", "away_score": "ag"}
    )
    df["played"] = df.hg.notna() & df.ag.notna()
    df["hg"] = df.hg.astype("Int64")
    df["ag"] = df.ag.astype("Int64")
    df["neutral"] = df.neutral.astype(bool)
    df["importance"] = df.tournament.map(importance_class)
    df["home_at_home"] = ~df.neutral

    # Exact duplicates would double-weight matches in every model downstream.
    df = df.drop_duplicates(subset=["date", "home", "away"], keep="first")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _clean_shootouts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("home_team", "away_team", "winner"):
        df[col] = df[col].map(canonical)
    return df.rename(columns={"home_team": "home", "away_team": "away"})


def _clean_goalscorers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("home_team", "away_team", "team"):
        df[col] = df[col].map(canonical)
    return df.rename(columns={"home_team": "home", "away_team": "away"})


def run(force_download: bool = False) -> dict[str, int]:
    download(force=force_download)
    out = processed_dir()
    out.mkdir(parents=True, exist_ok=True)

    matches = _clean_results(raw_dir() / "results.csv")
    shootouts = _clean_shootouts(raw_dir() / "shootouts.csv")
    scorers = _clean_goalscorers(raw_dir() / "goalscorers.csv")

    matches.to_parquet(out / "matches.parquet", index=False)
    shootouts.to_parquet(out / "shootouts.parquet", index=False)
    scorers.to_parquet(out / "goalscorers.parquet", index=False)
    return {"matches": len(matches), "shootouts": len(shootouts), "goalscorers": len(scorers)}


def load_matches(played_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(processed_dir() / "matches.parquet")
    return df[df.played].reset_index(drop=True) if played_only else df


def load_shootouts() -> pd.DataFrame:
    return pd.read_parquet(processed_dir() / "shootouts.parquet")
