"""Shared helpers for benchmark models.

The interval rule lives here so that a comparison between two models is a comparison of
their *point* forecasts and not an accident of how each one happened to widen its bands.
Any model is free to ignore this and emit its own intervals -- coverage and Winkler are on
the leaderboard precisely so that a better uncertainty model can show up as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import traffic_pipeline as T


def band_offsets(resid_log: np.ndarray, pred_log: np.ndarray, target: float = 0.90) -> dict:
    """Symmetric-in-quantile additive offsets per volume band, fitted in log space.

    Split-conformal in spirit: take the empirical quantiles of the fit residuals inside
    each volume band. Additive rather than multiplicative because a multiplicative scale
    on a collapsed interval explodes -- measured at widths of 16,000 on a 622-vehicle
    forecast.
    """
    if len(resid_log) == 0:
        raise ValueError("no residuals to fit intervals on -- the fit window was empty")
    lo_q, hi_q = (1 - target) / 2, 1 - (1 - target) / 2
    bands = T.volume_band(pred_log)
    out = {}
    for b in np.unique(bands):
        r = resid_log[bands == b]
        if len(r) >= 200:
            out[b] = (float(np.quantile(r, lo_q)), float(np.quantile(r, hi_q)))
    out["__global__"] = (float(np.quantile(resid_log, lo_q)), float(np.quantile(resid_log, hi_q)))
    return out


def apply_offsets(pred_log: np.ndarray, offs: dict) -> np.ndarray:
    """-> (n, 3) volumes: lower_90, forecast, upper_90."""
    bands = T.volume_band(pred_log)
    lo = np.empty_like(pred_log)
    hi = np.empty_like(pred_log)
    gl, gh = offs["__global__"]
    for i, b in enumerate(bands):
        a, c = offs.get(b, (gl, gh))
        lo[i], hi[i] = a, c
    out = np.column_stack([
        np.expm1(pred_log + lo),
        np.expm1(pred_log),
        np.expm1(pred_log + hi),
    ])
    return np.clip(out, 0.0, None)


def cold_pairs_of(rows: pd.DataFrame, known: set) -> np.ndarray:
    """Boolean mask of rows whose station-direction pair has no fitted history."""
    return ~rows.pair.isin(known).to_numpy()
