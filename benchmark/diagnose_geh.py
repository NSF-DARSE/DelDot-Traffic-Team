"""
Three cheap diagnostics that decide where the remaining GEH error actually lives.
No model is refitted; every number comes from one fold's held-out predictions.

Motivating algebra (verified numerically in `_selftest`):

    GEH = |M-C| * sqrt(2/(M+C))   and   2|sqrt(M) - sqrt(C)| = 2|M-C|/(sqrt M + sqrt C)

agree to <1.5% over the whole relevant range, so

    GEH < 5   <=>   |sqrt(M) - sqrt(C)| < 2.5

i.e. GEH is an L1 metric in SQUARE-ROOT space. The pipeline trains everything in log1p
space, which is a *relative* error geometry: at 2000 veh/h GEH<5 allows 11% error, at
50 veh/h it allows 71%. Log space treats those as equally important; GEH does not. That
mismatch is the suspected reason the GNN scores ~98% GEH<5 in the lowest volume decile
and ~40% in the highest.

    1. global multiplier   -- is the model biased as a whole? (a scalar fixes it if so)
    2. per-pair oracle     -- is the residual error per-series LEVEL, or shape/noise?
    3. GEH<5 by decile     -- which end of the volume distribution is being under-served?

Reading it: if (2) lands far above the achieved number, the error is level-shaped and
per-series recentring / reconciliation pays. If (2) barely moves, the error is shape or
irreducible noise and level work is wasted.

    python diagnose_geh.py --fold 0
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


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def geh(pred_vol, y_vol):
    return np.sqrt(2 * (pred_vol - y_vol) ** 2 / np.maximum(pred_vol + y_vol, 1e-9))


def geh5(pred_log, y_log):
    return 100 * float((geh(np.expm1(pred_log).clip(0), np.expm1(y_log)) < 5).mean())


def global_multiplier(pred, y_log, grid=np.round(np.arange(0.70, 1.4001, 0.01), 3)):
    """Best single scalar applied to the point forecast, in volume space."""
    y = np.expm1(y_log)
    m = np.expm1(pred[:, 1]).clip(0)
    # many multipliers can tie; prefer the least aggressive correction of the tied set
    scores = [(100 * float((geh(c * m, y) < 5).mean()), -abs(c - 1.0), c) for c in grid]
    best, _, c = max(scores)
    at1 = 100 * float((geh(m, y) < 5).mean())
    return {"achieved_%": round(at1, 2),
            "best_%": round(best, 2), "best_multiplier": round(float(c), 3)}


def per_pair_oracle(pred, y_log, pairs):
    """Upper bound on anything that corrects a per-series LEVEL bias.

    Each station-direction gets the single multiplier that best matches its own mean.
    This is an oracle -- it uses held-out truth -- so it is a ceiling, not a model.
    """
    y = np.expm1(y_log)
    m = np.expm1(pred[:, 1]).clip(0)
    df = pd.DataFrame({"pair": pairs, "y": y, "m": m})
    g = df.groupby("pair", observed=True)[["y", "m"]].mean()
    scale = (g.y / g.m.clip(lower=1e-6)).clip(0.2, 5.0)
    m2 = m * df.pair.map(scale).to_numpy()
    return {"achieved_%": round(100 * float((geh(m, y) < 5).mean()), 2),
            "per_pair_oracle_%": round(100 * float((geh(m2, y) < 5).mean()), 2)}


def by_volume_decile(pred, y_log):
    y = np.expm1(y_log)
    m = np.expm1(pred[:, 1]).clip(0)
    g = geh(m, y) < 5
    q = pd.qcut(y, 10, labels=False, duplicates="drop")
    return {int(d): round(100 * float(g[q == d].mean()), 1) for d in sorted(set(q[~pd.isna(q)]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--out", default="diagnose_geh.json")
    a = ap.parse_args()

    D = T.load_all()
    G = T.build_graph(D["edges"], D["network"])
    axis = T.station_axis(D["labeled"], D["targets"])
    statics = R.static_table(D["network"], G)
    report, folds = V.make_splits(D["network"], G, D["warm"], a.seed, V.N_REPORT, 4)
    pool = set().union(*folds)
    unseen = folds[a.fold]

    preds: dict = {}
    log(f"fitting fold {a.fold}")
    V.run_fold(D, statics, G, axis, pool - unseen, unseen, a.seed,
               use_gnn=not a.no_gnn, preds_out=preds)

    out = {}
    for name, (rows, pred) in sorted(preds.items()):
        y = rows.ly.to_numpy()
        out[name] = {
            "n": int(len(rows)),
            "global_multiplier": global_multiplier(pred, y),
            "per_pair_level": per_pair_oracle(pred, y, rows.pair.to_numpy()),
            "GEH<5_by_volume_decile": by_volume_decile(pred, y),
        }
        gm, pp = out[name]["global_multiplier"], out[name]["per_pair_level"]
        print(f"\n--- {name}  (n={len(rows):,})")
        print(f"    achieved GEH<5            {pp['achieved_%']:6.2f}%")
        print(f"    best global multiplier    {gm['best_%']:6.2f}%  at c={gm['best_multiplier']}"
              f"   (+{gm['best_%']-pp['achieved_%']:.2f})")
        print(f"    per-pair level ORACLE     {pp['per_pair_oracle_%']:6.2f}%"
              f"   (+{pp['per_pair_oracle_%']-pp['achieved_%']:.2f})")
        print(f"    GEH<5 by volume decile    {out[name]['GEH<5_by_volume_decile']}")

    Path(a.out).write_text(json.dumps(out, indent=2))
    log(f"wrote {a.out}")


def _selftest():
    rng = np.random.default_rng(0)
    M = rng.uniform(1, 3000, 200_000)
    C = M * rng.uniform(0.5, 1.6, 200_000)
    g = geh(M, C)
    approx = 2 * np.abs(np.sqrt(M) - np.sqrt(C))
    assert np.abs(g - approx).max() / g.max() < 0.02
    assert float(((g < 5) == (np.abs(np.sqrt(M) - np.sqrt(C)) < 2.5)).mean()) > 0.99

    # a deliberately 20%-low forecast must be corrected by a ~1.25 multiplier
    y = np.log1p(np.full(5000, 800.0))
    pred = np.stack([np.log1p(np.full(5000, 640.0))] * 3, 1)
    # |sqrt(640c) - sqrt(800)| < 2.5  =>  c in (1.038, 1.480); the tie-break must return
    # the least aggressive multiplier in that band, not the largest one on the grid.
    r = global_multiplier(pred, y)
    assert r["best_%"] > 99.0, r
    assert 1.03 < r["best_multiplier"] < 1.10, r
    assert r["achieved_%"] == 0.0, r
    print("selftest ok")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
