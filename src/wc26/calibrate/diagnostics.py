"""Calibration diagnostics: reliability of the model's probabilistic claims.

These are checks, not fitting targets (plan section 10). Pooled across
tournaments because per-tournament curves are noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(preds: pd.DataFrame, prob_col: str, hit_col: str, bins: list[float]) -> pd.DataFrame:
    """Predicted vs realized rate within probability bins, with counts."""
    b = pd.cut(preds[prob_col], bins)
    g = preds.groupby(b, observed=True).agg(
        n=(hit_col, "size"), predicted=(prob_col, "mean"), realized=(hit_col, "mean")
    )
    return g.reset_index()


def draw_calibration(preds: pd.DataFrame) -> dict:
    """Predicted draw probability vs realized draw frequency (model claim P_draw)."""
    is_draw = (preds.hg == preds.ag).astype(float)
    return {
        "predicted_draw_rate": round(float(preds.p_draw.mean()), 4),
        "realized_draw_rate": round(float(is_draw.mean()), 4),
        "n": int(len(preds)),
    }
