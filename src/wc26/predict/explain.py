"""Analyst-grade explanations assembled from exact model quantities.

No post-hoc ML attribution: every sentence reports a number the model
actually used (plan section 11).
"""
from __future__ import annotations


def build_explanation(p: dict) -> str:
    home, away = p["home"], p["away"]
    lam, mu = p["lam"], p["mu"]
    parts = []

    gap = lam - mu
    if abs(gap) < 0.25:
        parts.append(
            f"Evenly matched: ratings imply {lam:.2f} vs {mu:.2f} expected goals."
        )
    else:
        fav = home if gap > 0 else away
        parts.append(
            f"{fav} are favourites on opponent-adjusted attack/defense ratings "
            f"({lam:.2f} vs {mu:.2f} expected goals)."
        )

    total = lam + mu
    if total < 2.2:
        parts.append("A low-scoring game is expected, which boosts compact scorelines and the draw.")
    elif total > 3.2:
        parts.append("Both attacks rate well above defenses; a high-scoring game is expected.")

    if p.get("home_at_home"):
        parts.append(f"{home} benefit from a home-country venue (+{p['host_boost']:.0%} on goal rate).")

    if p.get("knockout"):
        parts.append(
            f"Knockout match: {p['p_draw_90']:.0%} chance it is level after 90', in which case "
            f"extra time adds {p['et_xg']:.2f} expected goals (shootout excluded from the score)."
        )

    if p.get("mu_capped"):
        parts.append("Severe mismatch: expected goals were capped; exact margin is highly uncertain.")
    if p.get("stale"):
        parts.append("Caution: limited recent competitive data for one side; rating is partly shrunk to confederation level.")

    parts.append(
        f"Most likely score {p['modal_score']} at {p['modal_prob']:.1%}; "
        f"outcome split {p['p_home']:.0%}/{p['p_draw']:.0%}/{p['p_away']:.0%} (win/draw/loss)."
    )
    return " ".join(parts)
