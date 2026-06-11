import pandas as pd
import pytest

from wc26.ui.build import _schedule_lookup, _stage_label, correctness, slugify


def test_slugify_stable_and_clean():
    assert slugify("2026-06-11", "Mexico", "South Africa") == "2026-06-11-mexico-south-africa"
    assert slugify("2026-06-24", "Bosnia and Herzegovina", "Canada") == "2026-06-24-bosnia-and-herzegovina-canada"


def test_correctness_rule():
    assert correctness("1-0", "1-0") is True
    assert correctness("1-0", "2-1") is False
    assert correctness("1-0", None) is None      # not played yet
    assert correctness(None, None) is None        # pending knockout slot


def test_schedule_lookup_is_order_insensitive():
    sch = pd.DataFrame([
        {"date": pd.Timestamp("2026-06-24"), "home": "Switzerland", "away": "Canada",
         "group": "Group B", "matchday": 3.0, "round": "Matchday 3", "ground": "Vancouver", "time": "12:00"},
    ])
    look = _schedule_lookup(sch)
    key_swapped = (pd.Timestamp("2026-06-24").date(), frozenset({"Canada", "Switzerland"}))
    assert key_swapped in look
    assert look[key_swapped]["group"] == "Group B"


def test_stage_label():
    assert _stage_label({"group": "Group A", "matchday": 1.0}, knockout=False) == "Group A · Matchday 1"
    assert _stage_label({"round": "Round of 32"}, knockout=True) == "Round of 32"


def test_build_artifact_structure(matches):
    # Full build against real data (ingest already run by the fixtures).
    from wc26.ui.build import build_artifact

    art = build_artifact(asof=pd.Timestamp("2026-06-11"), write_snapshots=False)
    assert {"build_ts", "model_version", "n_predicted", "matches"} <= art.keys()
    assert art["n_predicted"] == 72                      # group fixtures
    predicted = [m for m in art["matches"] if m["status"] != "pending"]
    pending = [m for m in art["matches"] if m["status"] == "pending"]
    assert len(predicted) == 72 and len(pending) == 32   # + knockout placeholders

    ids = [m["id"] for m in art["matches"]]
    assert len(ids) == len(set(ids)), "match ids must be unique"

    sample = predicted[0]
    required = {"id", "home", "away", "predicted_score", "confidence", "modal_prob",
                "explanation", "lam", "mu", "top_scores", "stage", "status", "prediction_mode"}
    assert required <= sample.keys()
    sx, sy = sample["predicted_score"].split("-")
    assert sx.isdigit() and sy.isdigit()
    assert sample["confidence"] in {"high", "medium", "low"}
    assert len(sample["top_scores"]) == 5
    assert sample["prediction_mode"] == "pre-match"
    # No WC2026 match is played yet -> all upcoming, none marked completed.
    assert all(m["actual"] is None for m in predicted)
