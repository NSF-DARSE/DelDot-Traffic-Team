"""
Capacity sweep for the v2 R-GCN, scored on the real protocol folds.

Why this exists: capacity was badly misjudged once already (hidden 96 -> 192 moved
warm-future GEH<5 from 68.2% to 78.9%), and the criterion that hid it -- validation
pinball -- is now replaced by GEH<5. On this GPU an epoch costs ~1s, so a grid over
width/depth/regularisation is minutes, not hours.

Selection runs on folds 0..k-1 of the frozen protocol split; the winner must then be
confirmed by `validate_protocol.py --folds 4`. Set C is never touched.

    python sweep_gnn.py --folds 2 --out sweep_gnn.json
"""
from __future__ import annotations

import argparse, itertools, json, time
from pathlib import Path

import numpy as np

import traffic_pipeline as T
import run_pipeline as R
import validate_protocol as V
import gnn_v2 as G2
from gnn_baseline import pick_device

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


GRID = dict(
    hidden=[192, 384, 768],
    head=[384, 768],
    layers=[3],            # depth already measured as unhelpful (5 layers < 3); width is what pays
    node_dropout=[0.2, 0.4],
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=2, help="how many protocol folds to score on")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sweep_gnn.json")
    a = ap.parse_args()

    D = T.load_all()
    G = T.build_graph(D["edges"], D["network"])
    axis = T.station_axis(D["labeled"], D["targets"])
    statics = R.static_table(D["network"], G)
    P = G2.prepare(D, statics)
    dev = pick_device("auto")

    report, folds = V.make_splits(D["network"], G, D["warm"], a.seed, V.N_REPORT, 4)
    pool = set().union(*folds)
    lab = D["labeled"]
    ts = P["ts"]

    keys = sorted(GRID)
    combos = [dict(zip(keys, c)) for c in itertools.product(*(GRID[k] for k in keys))]
    log(f"{len(combos)} configs x {a.folds} folds on {dev}")

    results = []
    for ci, cfg in enumerate(combos, 1):
        per_fold = []
        for fi in range(a.folds):
            unseen = folds[fi]
            seen = pool - unseen
            fut_seen = lab[(lab.timestamp >= V.HIST_END) & (lab.station_key.isin(seen))]
            fut_unseen = lab[(lab.timestamp >= V.HIST_END) & (lab.station_key.isin(unseen))]

            keep = np.isin(P["station"], list(seen))
            rows = np.flatnonzero((ts < np.datetime64(V.HIST_END)) & keep)
            order = rows[np.argsort(ts[rows])]
            inner = int(len(order) * 0.9)
            dead = set(np.unique(P["node"][~keep]).tolist())

            m, hist, idav = G2.train(P, order[:inner], order[inner:], dead, dev,
                                     max_epochs=a.epochs, patience=a.patience,
                                     seed=a.seed, verbose=False, select="geh", **cfg)
            r = {"epochs": len(hist),
                 "val_geh5_%": max(h["val_geh5_%"] for h in hist)}
            for nm, rdf in (("warm-future", fut_seen), ("cold-future", fut_unseen)):
                pr = G2.predict_rows(m, P, rdf, dev, idav).astype(np.float64)
                r[nm] = V.score(rdf.ly.to_numpy(), pr)
            per_fold.append(r)

        agg = {nm: {"GEH<5_%": round(float(np.mean([f[nm]["GEH<5_%"] for f in per_fold])), 2),
                    "sd": round(float(np.std([f[nm]["GEH<5_%"] for f in per_fold])), 2),
                    "MAE": round(float(np.mean([f[nm]["MAE"] for f in per_fold])), 1)}
               for nm in ("warm-future", "cold-future")}
        results.append({"cfg": cfg, "folds": per_fold, "agg": agg})
        log(f"{ci:2d}/{len(combos)} {cfg} -> warm {agg['warm-future']['GEH<5_%']:5.2f}% "
            f"(±{agg['warm-future']['sd']:.2f}, MAE {agg['warm-future']['MAE']:.1f})  "
            f"cold {agg['cold-future']['GEH<5_%']:5.2f}%")
        Path(a.out).write_text(json.dumps(results, indent=2))

    best = max(results, key=lambda r: r["agg"]["warm-future"]["GEH<5_%"])
    print("\n" + "=" * 78)
    print("BEST on warm-future:", json.dumps(best["cfg"]), "->", json.dumps(best["agg"]))
    print("confirm with: validate_protocol.py --folds 4 --gnn-hidden ... --gnn-head ...")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
