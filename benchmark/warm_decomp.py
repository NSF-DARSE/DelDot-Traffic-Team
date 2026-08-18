"""
Where does warm-future error actually live, and what is the ceiling?

Warm-future is 70.5% of the scored rows and sits at 72.3% GEH<5 against an 85% industry
target, but nobody has decomposed it. This hands the model perfect information one component
at a time -- all measured on the protocol's held-out future window, never on training rows.

    level   the pair's true mean log volume over the scored period (captures YoY drift)
    shape   the pair's true hour x dow deviation
    month   the true month factor
    shock   the true hour-by-hour deviation left after level+shape+month

If level+shape+month lands near 85%, the remaining work is drift and calendar structure and
the target is reachable. If it lands well below, warm-future is noise-bound and 85% is not
attainable by any model on this data.

    python warm_decomp.py --folds 4
"""
from __future__ import annotations

import argparse, itertools, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import traffic_pipeline as T
import run_pipeline as R
import validate_protocol as V

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def geh5(pred_log, y_log):
    y, m = np.expm1(y_log), np.expm1(pred_log).clip(0)
    g = np.sqrt(2 * (m - y) ** 2 / np.maximum(m + y, 1e-9))
    return 100 * float((g < 5).mean()), float(np.abs(m - y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="warm_decomp.json")
    a = ap.parse_args()

    D = T.load_all(); lab, net = D["labeled"], D["network"]
    G = T.build_graph(D["edges"], net); axis = T.station_axis(lab, D["targets"])
    statics = R.static_table(net, G)
    _, folds = V.make_splits(net, G, D["warm"], a.seed, V.N_REPORT, 4)
    pool = set().union(*folds)

    combos = list(itertools.product([0, 1], repeat=3))
    res = {}
    for fi in range(a.folds):
        seen = pool - folds[fi]
        hist = lab[(lab.timestamp < V.HIST_END) & (lab.station_key.isin(seen))]
        fut = lab[(lab.timestamp >= V.HIST_END) & (lab.station_key.isin(seen))]
        bb = T.Backbone().fit(hist, axis)
        base = bb.predict(fut)
        y = fut.ly.to_numpy()

        # truth components, computed on the scored window itself (oracle by construction)
        # MEDIAN, not mean: GEH and MAE are volume-space and want the median. Adding a
        # mean log residual over-predicts on a right-skewed residual -- the same Jensen
        # trap recorded as the Duan-smearing failure in the brief (+25 MAE).
        f = fut.assign(base=base, resid=y - base)
        lvl_t = f.groupby("pair").resid.transform("median")                  # level/drift gap
        shp_t = f.groupby(["pair", "hour", "dow"]).resid.transform("median") - lvl_t
        mon_t = f.groupby(["pair", "month"]).resid.transform("median") - lvl_t

        for use_l, use_s, use_m in combos:
            p = base.copy()
            if use_l: p = p + lvl_t.to_numpy()
            if use_s: p = p + shp_t.to_numpy()
            if use_m: p = p + mon_t.to_numpy()
            name = "backbone" + ("+level" if use_l else "") + ("+shape" if use_s else "") \
                   + ("+month" if use_m else "")
            g, m = geh5(p, y)
            res.setdefault(name, []).append((g, m))

        # everything: the deterministic pair x hour x dow x month oracle
        pfull = base + f.groupby(["pair", "hour", "dow", "month"]).resid.transform("median").to_numpy()
        g, m = geh5(pfull, y)
        res.setdefault("ORACLE pair x hour x dow x month", []).append((g, m))
        # plus the true network-median shock on top of that oracle
        rem = pd.Series(y - pfull, index=fut.index)
        netmed = rem.groupby(fut.timestamp).transform("median")
        g, m = geh5(pfull + netmed.to_numpy(), y)
        res.setdefault("  + true network shock", []).append((g, m))
        log(f"fold {fi+1}/{a.folds} done")

    print("\n" + "=" * 74)
    print(f"{'information given to the model':38s} {'GEH<5':>8s} {'sd':>6s} {'MAE':>9s}")
    print("-" * 74)
    for k, v in sorted(res.items(), key=lambda kv: np.mean([x[0] for x in kv[1]])):
        g = np.array([x[0] for x in v]); m = np.array([x[1] for x in v])
        print(f"{k:38s} {g.mean():7.1f}% {g.std():6.2f} {m.mean():9.1f}")
    print("-" * 74)
    print("deployed warm-future GNN: 72.3% / MAE 167.5   industry target: 85%")
    Path(a.out).write_text(json.dumps(
        {k: [[float(x[0]), float(x[1])] for x in v] for k, v in res.items()}, indent=2))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
