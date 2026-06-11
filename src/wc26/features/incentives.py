"""Qualification-scenario engine for the 2026 group stage (plan section 4).

Format: 12 groups of 4; the top 2 of each group plus the 8 best third-placed
teams reach the Round of 32. That makes matchday-3 incentives unusually
asymmetric — a team already through, a team mathematically out, and a third
side needing a narrow result can share a group on the final day.

This module reconstructs group standings from results as of any date and
classifies each team's matchday-3 situation. The third-place cutoff is a
cross-group quantity, so "third-place contention" is judged against a
configurable points threshold (4 points has historically been enough in
32-team World Cups; the 48-team edition is expected to be similar). Status is
used for the analyst explanation and a dead-rubber flag, not as a fitted goal
feature — the Layer-2 ablation showed goal-based context features add little,
so we do not overclaim a scoring effect here.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

THIRD_PLACE_SAFE_POINTS = 4   # points that have reliably secured a best-third place
WIN, DRAW = 3, 1


@dataclass
class TeamStanding:
    team: str
    played: int
    points: int
    gf: int
    ga: int

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def group_standings(
    results: pd.DataFrame, fixtures: pd.DataFrame, teams: list[str], asof: pd.Timestamp
) -> list[TeamStanding]:
    """Standings among `teams` counting ONLY this group's tournament fixtures
    that have been played before `asof`.

    `fixtures` is the schedule slice for the group (its scheduled matches);
    we count a fixture only once its result appears in `results`. This keeps
    standings tournament-scoped — counting all historical head-to-head matches
    between the four teams would be a serious bug.
    """
    fx_keys = set(map(tuple, fixtures[["date", "home", "away"]].values))
    played = results[(results.date < asof) & results.played].copy()
    played = played[played[["date", "home", "away"]].apply(tuple, axis=1).isin(fx_keys)]
    tbl = {t: TeamStanding(t, 0, 0, 0, 0) for t in teams}
    for home, away, hg, ag in played[["home", "away", "hg", "ag"]].itertuples(index=False):
        hg, ag = int(hg), int(ag)
        h, a = tbl[home], tbl[away]
        h.played += 1; a.played += 1
        h.gf += hg; h.ga += ag; a.gf += ag; a.ga += hg
        if hg > ag:
            h.points += WIN
        elif hg < ag:
            a.points += WIN
        else:
            h.points += DRAW; a.points += DRAW
    return sorted(tbl.values(), key=lambda s: (-s.points, -s.gd, -s.gf, s.team))


def _max_possible_points(points: int, games_left: int) -> int:
    return points + WIN * games_left


def qualification_status(
    standings: list[TeamStanding], games_total: int = 3,
    third_safe: int = THIRD_PLACE_SAFE_POINTS,
) -> dict[str, str]:
    """Per-team status before a remaining matchday.

    Returns one of: secured_top2, contention, third_contention, eliminated.
    Guarantees are points-based and deliberately conservative — we flag a team
    as secured or eliminated only when it is true regardless of remaining
    results (goal difference is not used to claim certainty). The best-third
    judgement is threshold-based (cross-group cutoff is unknowable in-group).
    """
    out: dict[str, str] = {}
    for s in standings:
        my_max = _max_possible_points(s.points, games_total - s.played)
        # Teams that could still finish strictly above this team's floor.
        can_exceed = sum(
            1 for o in standings if o.team != s.team
            and _max_possible_points(o.points, games_total - o.played) > s.points
        )
        # Teams whose worst case still beats this team's best case (uncatchable).
        locked_above = sum(
            1 for o in standings if o.team != s.team and o.points > my_max
        )
        if locked_above >= 2 and my_max < third_safe:
            out[s.team] = "eliminated"            # can't reach top 2 nor a safe third
        elif can_exceed <= 1:
            out[s.team] = "secured_top2"          # at most one team can pass -> top 2 locked
        elif locked_above >= 2:
            out[s.team] = "third_contention"      # top 2 gone, but a best-third is alive
        else:
            out[s.team] = "contention"
    return out


def is_dead_rubber(status_home: str, status_away: str) -> bool:
    """A match where neither side's group outcome can still change in a way
    they can influence: both already secured top-2 or both eliminated."""
    sealed = {"secured_top2", "eliminated"}
    return status_home in sealed and status_away in sealed


def match_incentives(results, schedule, home: str, away: str, asof: pd.Timestamp) -> dict:
    """Stage + incentive context for a WC-2026 fixture, from the schedule."""
    pair = {home, away}
    fx = schedule[schedule.apply(lambda r: {r.home, r.away} == pair, axis=1)]
    if fx.empty:
        return {"stage": "unknown", "dead_rubber": False}
    row = fx.iloc[0]
    group, md = row.group, row.matchday
    if group is None or pd.isna(md):
        return {"stage": "knockout", "dead_rubber": False}

    group_fx = schedule[schedule.group == group]
    teams = sorted(set(group_fx.home) | set(group_fx.away))
    standings = group_standings(results, group_fx, teams, asof)
    status = qualification_status(standings)
    dead = int(md) == 3 and is_dead_rubber(status.get(home, ""), status.get(away, ""))
    return {
        "stage": f"group_md{int(md)}",
        "group": group,
        "dead_rubber": dead,
        "status_home": status.get(home),
        "status_away": status.get(away),
    }
