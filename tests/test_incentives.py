import pandas as pd

from wc26.features.incentives import (
    group_standings,
    is_dead_rubber,
    qualification_status,
)


def _results(rows):
    df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
    df["date"] = pd.to_datetime(df["date"])
    df["played"] = True
    return df


def _fixtures(results):
    # In these tests the scheduled group fixtures are exactly the played rows.
    return results[["date", "home", "away"]].copy()


def test_standings_points_and_gd():
    # After two matchdays in a group of four.
    res = _results([
        ("2026-06-11", "A", "B", 2, 0),
        ("2026-06-11", "C", "D", 1, 1),
        ("2026-06-18", "A", "C", 1, 0),
        ("2026-06-18", "B", "D", 0, 0),
    ])
    table = group_standings(res, _fixtures(res), ["A", "B", "C", "D"], pd.Timestamp("2026-06-24"))
    assert table[0].team == "A" and table[0].points == 6 and table[0].gd == 3
    names = [s.team for s in table]
    assert names[0] == "A"  # A clear top
    assert {s.team for s in table} == {"A", "B", "C", "D"}


def test_secured_and_eliminated_before_md3():
    # Standings before MD3: A 6, B 4, C 1, D 0 (each played 2).
    res = _results([
        ("2026-06-11", "A", "C", 1, 0),   # A3 C0
        ("2026-06-11", "B", "D", 1, 0),   # B3 D0
        ("2026-06-18", "A", "D", 1, 0),   # A6 D0
        ("2026-06-18", "B", "C", 1, 1),   # B4 C1
    ])
    table = group_standings(res, _fixtures(res), ["A", "B", "C", "D"], pd.Timestamp("2026-06-24"))
    status = qualification_status(table)
    # A on 6: only B (max 7) can pass -> top 2 mathematically locked.
    assert status["A"] == "secured_top2"
    # D on 0, max 3, with A(6) and B(4) both uncatchable and 3 < safe-third(4).
    assert status["D"] == "eliminated"
    # C on 1 can still reach 4 and isn't locked below two teams for a third place.
    assert status["C"] in {"third_contention", "contention"}


def test_dead_rubber_detection():
    assert is_dead_rubber("secured_top2", "eliminated")
    assert is_dead_rubber("eliminated", "eliminated")
    assert not is_dead_rubber("secured_top2", "contention")
    assert not is_dead_rubber("contention", "contention")


def test_runs_on_real_schedule(matches):
    from wc26.data import ingest as ing
    from wc26.features.incentives import match_incentives

    sched = ing.load_schedule()
    assert len(sched) >= 72
    assert sched.group.nunique() == 12
    # Opening match: MD1, no one eliminated yet, not a dead rubber.
    inc = match_incentives(matches, sched, "Mexico", "South Africa", pd.Timestamp("2026-06-11"))
    assert inc["stage"] == "group_md1"
    assert inc["dead_rubber"] is False
