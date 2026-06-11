"""Regression tests on hand-checked extra-time / shootout matches.

The prediction target is the 120' score excluding shootouts; if the source
ever records post-shootout or 90' scores inconsistently, training and
evaluation are silently corrupted (plan section 14, risk 1).
"""
import pandas as pd

# (date, home, away, hg, ag) — scores after extra time, shootout goals excluded.
KNOWN_ET_SCORES = [
    ("2022-12-18", "Argentina", "France", 3, 3),        # WC22 final, pens
    ("2014-07-13", "Germany", "Argentina", 1, 0),        # WC14 final, ET winner
    ("2010-07-11", "Netherlands", "Spain", 0, 1),        # WC10 final, ET winner Spain
    ("2006-07-09", "Italy", "France", 1, 1),             # WC06 final, pens
    ("2021-07-11", "England", "Italy", 1, 1),            # Euro 2020 final, pens
    ("2016-07-10", "France", "Portugal", 0, 1),          # Euro 2016 final, ET winner Portugal
    ("2018-07-11", "Croatia", "England", 2, 1),          # WC18 SF, ET winner
    ("2022-12-09", "Croatia", "Brazil", 1, 1),           # WC22 QF, pens
    ("2022-12-09", "Netherlands", "Argentina", 2, 2),    # WC22 QF, pens
    ("2022-12-05", "Japan", "Croatia", 1, 1),            # WC22 R16, pens
    ("2018-07-01", "Russia", "Spain", 1, 1),             # WC18 R16, pens
    ("2018-07-01", "Croatia", "Denmark", 1, 1),          # WC18 R16, pens
    ("2024-07-06", "England", "Switzerland", 1, 1),      # Euro 2024 QF, pens
    ("2016-06-26", "Argentina", "Chile", 0, 0),          # Copa 2016 final, pens
    ("2022-12-10", "England", "France", 1, 2),           # WC22 QF: no ET, 90' score
]

KNOWN_SHOOTOUT_WINNERS = [
    ("2022-12-18", "Argentina", "France", "Argentina"),
    ("2006-07-09", "Italy", "France", "Italy"),
    ("2021-07-11", "England", "Italy", "Italy"),
    ("2022-12-09", "Croatia", "Brazil", "Croatia"),
    ("2022-12-09", "Netherlands", "Argentina", "Argentina"),
    ("2016-06-26", "Argentina", "Chile", "Chile"),
]


def test_known_et_scores(matches):
    for date, home, away, hg, ag in KNOWN_ET_SCORES:
        row = matches[
            (matches.date == pd.Timestamp(date)) & (matches.home == home) & (matches.away == away)
        ]
        assert len(row) == 1, f"missing match {date} {home}-{away}"
        r = row.iloc[0]
        assert (int(r.hg), int(r.ag)) == (hg, ag), (
            f"{date} {home}-{away}: recorded {r.hg}-{r.ag}, expected {hg}-{ag}"
        )


def test_known_shootouts(shootouts):
    for date, home, away, winner in KNOWN_SHOOTOUT_WINNERS:
        row = shootouts[
            (shootouts.date == pd.Timestamp(date))
            & (shootouts.home == home)
            & (shootouts.away == away)
        ]
        assert len(row) == 1, f"missing shootout {date} {home}-{away}"
        assert row.iloc[0].winner == winner


def test_et_matches_recorded_as_draw_iff_pens(matches, shootouts):
    # In single-match knockouts (all WC/continental finals), a shootout implies
    # the recorded 120' score is a draw. Two-legged qualifiers/regional ties can
    # legitimately have non-draw match scores with the shootout settling the
    # aggregate, so the invariant is asserted on major tournaments only.
    merged = shootouts.merge(matches, on=["date", "home", "away"], how="inner")
    majors = merged[merged.importance.isin(["world_cup", "continental_finals"])]
    assert len(majors) > 150
    assert (majors.hg == majors.ag).all(), "major-tournament shootouts must be recorded as draws"
