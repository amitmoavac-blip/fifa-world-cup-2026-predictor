"""Team registry: canonical national-team names and confederations.

Filters out non-FIFA entities (CONIFA sides, Island Games teams, ...) and is
the firewall against silently wrong ratings from name drift (plan section 3).
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from wc26.paths import configs_dir

CONFEDERATIONS = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]

# Name variants occasionally seen in external sources, mapped to the
# martj42 canonical spelling used throughout the repo.
ALIASES = {
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
}


@lru_cache(maxsize=1)
def load_registry() -> pd.DataFrame:
    df = pd.read_csv(configs_dir() / "teams.csv")
    assert df.name.is_unique, "duplicate team names in configs/teams.csv"
    bad = set(df.confed) - set(CONFEDERATIONS)
    assert not bad, f"unknown confederations: {bad}"
    return df


def canonical(name: str) -> str:
    return ALIASES.get(name.strip(), name.strip())


@lru_cache(maxsize=1)
def confed_map() -> dict[str, str]:
    df = load_registry()
    return dict(zip(df.name, df.confed))


def is_registered(name: str) -> bool:
    return canonical(name) in confed_map()
