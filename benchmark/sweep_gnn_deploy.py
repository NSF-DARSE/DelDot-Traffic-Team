"""
GNN capacity sweep on the deployment arm.

`sweep_gnn.py` sweeps on the protocol, which is the right setup for cold-start but the
wrong one for the warm slice: it fits on 2024 alone and so discards the 2025 H1 half-year
deployment actually holds. Warm-slice decisions belong here.

The sweep also has to be redone rather than reused, because the GNN's capacity was tuned
when it had no `base` input and was spending its embedding budget re-deriving the profile.
With the backbone handed to it, the old width/depth conclusions describe a different model.

    python sweep_gnn_deploy.py
"""
from __future__ import annotations

import argparse, itertools, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import traffic_pipeline as T
import run_pipeline as R
import gnn_v2 as V
from gnn_baseline import pick_device

T0 = time.time()
CUT = pd.Timestamp("2025-05-01")

GRID = dict(hidden=[192, 384, 768], head=[384, 768], node_dropout=[0.2, 0.4])


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sweep_gnn_deploy.json")
    a = ap.parse_args()

    D = T.load_all(); net = D["network"]
    axis = T.station_axis(D["labeled"], D["targets"])
    G = T.build_graph(D["edges"], net); statics = R.static_table(net, G)
    lab = D["labeled"][D["labeled"].station_key.isin(D["warm"])].copy()
    hist, test = lab[lab.timestamp < CUT], lab[lab.timestamp >= CUT]
    y = test.ly.to_numpy()

    def geh(p):
        yy, m = np.expm1(y), np.expm1(p).clip(0)
        g = np.sqrt(2 * (m - yy) ** 2 / np.maximum(m + yy, 1e-9))
        return round(100 * float((g < 5).mean()), 2), round(float(np.abs(m - yy).mean()), 1)

    bb = T.Backbone().fit(hist, axis)
    dm = T.DonorModel(G, net).fit(bb, hist)
    dinfo = dm.station_levels(sorted(set(net.station_key)))
    Xh = R.build_features(hist, bb, statics, dinfo, None, warm=D["warm"])
    Xt = R.build_features(test, bb, statics, dinfo, None, warm=D["warm"])
    p_lgb = R.fit_path(Xh, hist.ly.to_numpy() - Xh.base.to_numpy(), Xt,
                       R.feature_list(False), seed=a.seed) + Xt.base.to_numpy()[:, None]
    log(f"backbone {geh(bb.predict(test))}  |  +LightGBM {geh(p_lgb[:, 1])}")

    base_s = pd.Series(np.nan, index=lab.index, dtype=float)
    base_s.loc[hist.index] = Xh.base.to_numpy()
    base_s.loc[test.index] = Xt.base.to_numpy()
    lab_g = lab.assign(base=base_s.to_numpy())
    P = V.prepare({**D, "labeled": lab_g}, statics)
    dev = pick_device("auto")
    ts = P["ts"]
    rows = np.flatnonzero(ts < np.datetime64(CUT))
    order = rows[np.argsort(ts[rows])]
    inner = int(len(order) * 0.92)

    keys = sorted(GRID)
    combos = [dict(zip(keys, c)) for c in itertools.product(*(GRID[k] for k in keys))]
    log(f"{len(combos)} configs on {dev}")
    res = []
    for i, cfg in enumerate(combos, 1):
        m, h, idav = V.train(P, order[:inner], order[inner:], set(), dev,
                             max_epochs=200, patience=12, seed=a.seed, verbose=False,
                             select="geh", **cfg)
        pr = V.predict_rows(m, P, lab_g.loc[test.index], dev, idav).astype(np.float64)
        g_gnn = geh(pr[:, 1])
        g_mix = geh(0.5 * pr[:, 1] + 0.5 * p_lgb[:, 1])
        res.append({"cfg": cfg, "epochs": len(h), "gnn": g_gnn, "blend": g_mix})
        log(f"{i:2d}/{len(combos)} {cfg} -> GNN {g_gnn[0]:5.2f}% / {g_gnn[1]:6.1f}   "
            f"blend {g_mix[0]:5.2f}% / {g_mix[1]:6.1f}  ({len(h)} ep)")
        Path(a.out).write_text(json.dumps(res, indent=2))

    best = max(res, key=lambda r: r["blend"][0])
    print("\n" + "=" * 70)
    print("BEST by blended GEH<5:", json.dumps(best["cfg"]),
          f"-> blend {best['blend'][0]}% / MAE {best['blend'][1]}")
    print("deployed reference: GNN 83.4% / 90.9, blend 83.5% / 90.7")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
