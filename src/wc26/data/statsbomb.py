"""StatsBomb open-data ingestion: match-level xG for international tournaments.

The only free, research-licensed source of international xG. Coverage is the
six modern men's national-team tournaments below; xG exists ONLY within these
tournaments (not the qualifiers/friendlies between them), so any xG-form
feature built on this data is necessarily within-tournament (plan: Tier-B).

We download each match's event JSON once, sum shot xG per team, cache a small
table, and discard the bulky raw events. Attribution: data provided free by
Hudl StatsBomb (https://github.com/statsbomb/open-data) under their licence.
"""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from wc26.data.registry import canonical
from wc26.paths import processed_dir

SB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# competition_id, season_id, and the matching key in configs/tournaments.csv
# (AFCON 2023 is not in the main backtest list; key left as None.)
TOURNAMENTS = {
    "WC2018": (43, 3, "wc2018"),
    "WC2022": (43, 106, "wc2022"),
    "Euro2020": (55, 43, "euro2020"),
    "Euro2024": (55, 282, "euro2024"),
    "Copa2024": (223, 282, "copa2024"),
    "AFCON2023": (1267, 107, None),
}


def _fetch_json(url: str, retries: int = 3):
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - transient network/ratelimit
            last = e
    raise last


def _match_xg(match_id: int) -> tuple[float, float, dict[str, float]]:
    events = _fetch_json(f"{SB}/events/{match_id}.json")
    per_team: dict[str, float] = {}
    for e in events:
        if e.get("type", {}).get("name") == "Shot":
            t = e["team"]["name"]
            per_team[t] = per_team.get(t, 0.0) + e["shot"].get("statsbomb_xg", 0.0)
    return match_id, per_team


def _tournament_rows(label: str, comp: int, season: int, key: str | None) -> list[dict]:
    matches = _fetch_json(f"{SB}/matches/{comp}/{season}.json")
    by_id = {m["match_id"]: m for m in matches}
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for match_id, per_team in ex.map(lambda m: _match_xg(m["match_id"]), matches):
            m = by_id[match_id]
            home = canonical(m["home_team"]["home_team_name"])
            away = canonical(m["away_team"]["away_team_name"])
            rows.append({
                "date": pd.Timestamp(m["match_date"]),
                "home": home,
                "away": away,
                "hg": int(m["home_score"]),
                "ag": int(m["away_score"]),
                "xg_home": round(per_team.get(m["home_team"]["home_team_name"], 0.0), 4),
                "xg_away": round(per_team.get(m["away_team"]["away_team_name"], 0.0), 4),
                "stage": m.get("competition_stage", {}).get("name"),
                "tournament_label": label,
                "tournament_key": key,
            })
    return rows


def run(force: bool = False) -> int:
    out = processed_dir() / "statsbomb_xg.parquet"
    if out.exists() and not force:
        return len(pd.read_parquet(out))
    rows: list[dict] = []
    for label, (comp, season, key) in TOURNAMENTS.items():
        rows.extend(_tournament_rows(label, comp, season, key))
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    processed_dir().mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return len(df)


def load_xg() -> pd.DataFrame:
    return pd.read_parquet(processed_dir() / "statsbomb_xg.parquet")
