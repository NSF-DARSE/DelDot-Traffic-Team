"""
Better per-pair profiles for warm stations -- the largest measured lever in the problem.

`warm_decomp.py` shows a perfect hour x dow shape is worth +9.3 GEH points on warm-future
(68.7% -> 78.0%), and a perfect pair x hour x dow x month profile reaches 88.4% against the
72.3% we deploy. The backbone already estimates that profile from 2024; the gap is therefore
estimation error, not missing information.

A per-pair profile has 168 cells (24 x 7) estimated from a partial year with whole-day
outages, so each cell rests on few observations and is noisy. Three ways to denoise, all
scored against the pair's TRUE 2025 H1 profile:

    raw            the per-pair empirical profile (what Backbone uses)
    shrunk         empirical-Bayes shrinkage toward the direction-level profile
    low-rank       SVD across pairs, keep k components  -- profiles live on a low-dimensional
                   manifold (commuter / rural / recreational shapes), so the trailing
                   components are mostly noise
    low-rank+shrunk  both

This is the same collaborative-filtering machinery as `cf_shape.py`, pointed at denoising
observed profiles rather than transferring to unobserved ones.

    python warm_shape.py --folds 4
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import traffic_pipeline as T
import run_pipeline as R
import validate_protocol as V

T0 = time.time()
HOURS, DOWS = 24, 7
CELLS = pd.MultiIndex.from_product([range(HOURS), range(DOWS)])


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def profile_and_counts(d: pd.DataFrame):
    """Per-pair (168-cell) mean deviation from the pair median, plus per-cell counts."""
    med = d.groupby("pair").ly.transform("median")
    dev = (d.ly - med)
    g = dev.groupby([d.pair, d.hour, d.dow])
    P = g.mean().unstack(level=[1, 2]).reindex(columns=CELLS)
    N = g.size().unstack(level=[1, 2]).reindex(columns=CELLS).fillna(0.0)
    return P, N


def direction_profile(P: pd.DataFrame) -> pd.DataFrame:
    dc = pd.Series([p.split("|")[1] for p in P.index], index=P.index)
    grp = P.groupby(dc).transform("mean")
    return grp


def shrink(P, N, prior, k):
    """Empirical Bayes: each cell moves toward the prior by its own sample size."""
    w = N / (N + k)
    pr = prior.fillna(0.0)
    return (w * P.fillna(pr) + (1 - w) * pr)


def low_rank(P: pd.DataFrame, k: int) -> pd.DataFrame:
    A = P.to_numpy(copy=True).astype(float)
    # some (hour, dow) cells are empty for every pair -- train hours 0/1 never occur -- so
    # the column mean is itself NaN and must fall back to the global mean, then to zero.
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(np.isfinite(A), A, np.nan), axis=0)
    gm = float(np.nanmean(mu)) if np.isfinite(mu).any() else 0.0
    mu = np.where(np.isfinite(mu), mu, gm)
    A = np.where(np.isfinite(A), A, mu)
    Ac = np.nan_to_num(A - mu, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
    rec = (U[:, :k] * S[:k]) @ Vt[:k] + mu
    return pd.DataFrame(rec, index=P.index, columns=P.columns)


def score(est: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Every candidate is scored on exactly the same cells.

    The raw profile is undefined for (hour, dow) cells with no 2024 observation -- true
    hours 0 and 1 are absent from the training year entirely -- and the direction prior is
    undefined there too. Leaving those NaN and calling `nanmean` scores the raw profile on
    the easy 92% of cells while scoring every filled candidate on all of them, which made
    the deployed baseline look far better than it is. Fill first, then score everything.
    """
    common = est.index.intersection(truth.index)
    e = est.reindex(index=common, columns=truth.columns).fillna(0.0)
    t = truth.reindex(index=common)
    m = t.notna().to_numpy()
    return float(np.abs(e.to_numpy()[m] - t.to_numpy()[m]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="warm_shape.json")
    a = ap.parse_args()

    D = T.load_all(); lab, net = D["labeled"], D["network"]
    G = T.build_graph(D["edges"], net)
    _, folds = V.make_splits(net, G, D["warm"], a.seed, V.N_REPORT, 4)
    pool = set().union(*folds)

    res: dict = {}
    for fi in range(a.folds):
        seen = pool - folds[fi]
        hist = lab[(lab.timestamp < V.HIST_END) & (lab.station_key.isin(seen))]
        fut = lab[(lab.timestamp >= V.HIST_END) & (lab.station_key.isin(seen))]
        P, N = profile_and_counts(hist)
        Ptrue, Ntrue = profile_and_counts(fut)
        Ptrue = Ptrue[Ntrue.sum(axis=1) > 1000]          # only pairs with a solid truth
        prior = direction_profile(P)

        # control: the same cell-mean imputation low_rank uses, WITHOUT any factorisation.
        # Without this, a gain from better filling of the empty cells would be misread as
        # a gain from denoising.
        # fillna with a Series aligns on COLUMNS, so this fills each (hour, dow) cell with
        # that cell's average across pairs -- the network-wide profile value.
        colmean = P.mean(axis=0)
        filled = P.fillna(prior).fillna(colmean).fillna(0.0)

        cand = {"raw (deployed, zero-filled)": P.fillna(prior),
                "raw + cell-mean fill (control)": filled}
        for k in (2, 5, 10, 20):
            cand[f"shrunk k={k}"] = shrink(P, N, prior, k)
        for r in (5, 10, 20, 30, 45, 60, 80, 120):
            cand[f"low-rank r={r}"] = low_rank(filled, r)
        for r in (20, 45):
            for k in (2, 5):
                cand[f"low-rank r={r} + shrunk k={k}"] = low_rank(shrink(P, N, prior, k), r)

        for name, est in cand.items():
            res.setdefault(name, []).append(score(est, Ptrue))
        log(f"fold {fi+1}/{a.folds}: {len(P)} pairs, {len(Ptrue)} scored")

    print("\n" + "=" * 72)
    print("mean |error| of the estimated profile against the TRUE 2025 H1 profile")
    print("-" * 72)
    base = float(np.mean(res["raw (deployed, zero-filled)"]))
    for name, v in sorted(res.items(), key=lambda kv: np.mean(kv[1])):
        v = np.array(v)
        print(f"{name:30s} {v.mean():8.4f}  sd {v.std():6.4f}  ({100*(base-v.mean())/base:+5.1f}%)")
    print("-" * 72)
    Path(a.out).write_text(json.dumps({k: list(map(float, v)) for k, v in res.items()}, indent=2))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
