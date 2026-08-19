"""
Evaluation metrics for the day-1 traffic forecast.
==================================================
Point accuracy (MAE, RMSE, R2, WAPE, %within), the traffic-engineering
standard (GEH), interval quality (coverage, mean width, Winkler), and the
gap-to-ceiling framing that reports results as the share of the naive->oracle
gap that the model closes.
"""
from __future__ import annotations
import numpy as np


def point_metrics(actual, pred):
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    err = a - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2)); ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot else float("nan")
    wape = float(np.sum(np.abs(err)) / np.sum(a)) if np.sum(a) else float("nan")
    denom = np.maximum(a, 1.0)
    within20 = float(np.mean(np.abs(err) / denom <= 0.20) * 100)
    within30 = float(np.mean(np.abs(err) / denom <= 0.30) * 100)
    return dict(MAE=mae, RMSE=rmse, R2=r2, WAPE=wape,
                pct_within_20=within20, pct_within_30=within30)


def geh_metrics(actual, pred):
    """GEH statistic: the DOT acceptance standard. GEH<5 = good, <10 = acceptable."""
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    geh = np.sqrt(2 * (p - a) ** 2 / np.maximum(p + a, 1e-9))
    return dict(GEH_mean=float(np.mean(geh)),
                pct_GEH_under_5=float(np.mean(geh < 5) * 100),
                pct_GEH_under_10=float(np.mean(geh < 10) * 100))


def interval_metrics(actual, lo, hi, alpha=0.10):
    a = np.asarray(actual, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    covered = (a >= lo) & (a <= hi)
    coverage = float(np.mean(covered) * 100)
    width = hi - lo
    # Winkler / interval score (lower is better)
    w = width.copy()
    below = a < lo; above = a > hi
    w[below] += (2 / alpha) * (lo[below] - a[below])
    w[above] += (2 / alpha) * (a[above] - hi[above])
    return dict(coverage_pct=coverage, target_pct=(1 - alpha) * 100,
                mean_width=float(np.mean(width)), winkler=float(np.mean(w)))


def gap_to_ceiling(actual, pred, naive, oracle):
    """
    Report the model as the fraction of the naive->oracle WAPE gap it closes.
    naive  = train-derived seasonal profile (a floor any entrant can hit)
    oracle = same-period profile computed from the scored data itself (the
             best a pure profile model could ever do; a reference ceiling only)
    """
    def wape(p):
        return float(np.sum(np.abs(np.asarray(actual, float) - p)) / np.sum(actual))
    wn, wm, wo = wape(naive), wape(pred), wape(oracle)
    closed = (wn - wm) / (wn - wo) if (wn - wo) else float("nan")
    return dict(wape_naive=wn, wape_model=wm, wape_oracle=wo,
                gap_closed_pct=float(closed * 100))


def all_metrics(actual, pred, lo, hi, naive=None, oracle=None):
    m = {}
    m.update(point_metrics(actual, pred))
    m.update(geh_metrics(actual, pred))
    m.update(interval_metrics(actual, lo, hi))
    if naive is not None and oracle is not None:
        m.update(gap_to_ceiling(actual, pred, naive, oracle))
    return m
