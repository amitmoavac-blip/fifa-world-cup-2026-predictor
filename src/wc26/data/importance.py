"""Match-importance classification from martj42 tournament strings.

The source data has no importance column (plan section 3); classes here drive
both the Layer-1 information weights and the Elo K-factors.
"""
from __future__ import annotations

WORLD_CUP = "world_cup"
CONTINENTAL_FINALS = "continental_finals"
QUALIFIER = "qualifier"
NATIONS_LEAGUE = "nations_league"
MINOR = "minor"
FRIENDLY = "friendly"

_CONTINENTAL_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Oceania Nations Cup",
    "Confederations Cup",
    "FIFA Confederations Cup",
}

_FRIENDLY_LIKE = {
    "Friendly",
    "FIFA Series",
    "Cyprus International Tournament",
    "King's Cup",
    "Kirin Cup",
    "Kirin Challenge Cup",
    "Nehru Cup",
    "Lunar New Year Cup",
    "Merdeka Tournament",
    "USA Cup",
    "Rous Cup",
    "Tournoi de France",
}


def importance_class(tournament: str) -> str:
    t = tournament.strip()
    if t == "FIFA World Cup":
        return WORLD_CUP
    if t in _CONTINENTAL_FINALS:
        return CONTINENTAL_FINALS
    if t.endswith("qualification"):
        base = t[: -len("qualification")].strip()
        if base.startswith("FIFA World Cup") or base in _CONTINENTAL_FINALS:
            return QUALIFIER
        if "Nations League" in base:
            return NATIONS_LEAGUE
        return MINOR
    if "Nations League" in t:
        return NATIONS_LEAGUE
    if t in _FRIENDLY_LIKE:
        return FRIENDLY
    return MINOR
