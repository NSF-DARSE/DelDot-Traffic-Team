"""
Can a cold station's *contemporaneous shock* be predicted from the warm network?

This is the one cold-start channel that is not bounded by station attributes. Cold-interp
targets sit INSIDE the labelled window, so at each target timestamp every warm station is
recording. The oracle decomposition says a perfect deterministic profile reaches 80.5%
GEH<5 on cold-interp, and adding the observed network-median residual reaches 85.8% -- so
the shock channel is worth about +5.3 points on top of a perfect profile, and nothing about
`aadt_band` or the graph caps it.

`traffic_pipeline.neighbour_features` already reads this channel, but only through the 186
graph edges: corridor mean, proximity mean, network median. This asks whether *learned*
similarity over all warm pairs does better -- collaborative filtering, where the neighbourhood
is inferred from data rather than declared by the edge list.

Target definition matters. The shock is measured against each cold pair's OWN fitted profile,
so what is scored is purely the time-varying deviation, with level and shape already removed.
That is the same decomposition the oracle above uses.

    python cf_residual.py --folds 4
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import traffic_pipeline as T
import run_pipeline as R
import validate_protocol as V
from cf_shape import pair_features

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def residual_panel(d: pd.DataFrame) -> pd.DataFrame:
    """(timestamp x pair) deviation from each pair's own hour x dow profile."""
    med = d.groupby("pair").ly.transform("median")
    prof = (d.ly - med).groupby([d.pair, d.hour, d.dow]).transform("mean")
    r = d.ly - med - prof
    return pd.DataFrame({"timestamp": d.timestamp, "pair": d.pair, "r": r}) \
             .pivot_table(index="timestamp", columns="pair", values="r", aggfunc="mean")


def pair_similarity_model(Rw: pd.DataFrame, Xw: pd.DataFrame, seed=0):
    """Learn corr(r_i, r_j) from the two pairs' features. Trained on warm-warm pairs only."""
    C = Rw.corr(min_periods=200)
    pairs = list(Rw.columns)
    rows, ys = [], []
    for i, pi in enumerate(pairs):
        for pj in pairs[i + 1:]:
            c = C.loc[pi, pj]
            if np.isnan(c):
                continue
            a, b = Xw.loc[pi].to_numpy(), Xw.loc[pj].to_numpy()
            rows.append(np.concatenate([np.abs(a - b), (a + b) / 2]))
            ys.append(c)
    if not rows:
        return None
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    m.fit(np.array(rows), np.array(ys))
    return m


def predict_similarity(model, Xw, xc):
    a = Xw.to_numpy(); b = xc.to_numpy()
    F = np.hstack([np.abs(a - b), (a + b) / 2])
    return model.predict(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--out", default="cf_residual_results.json")
    a = ap.parse_args()

    D = T.load_all(); lab, net = D["labeled"], D["network"]
    G = T.build_graph(D["edges"], net)
    statics = R.static_table(net, G)
    hist = lab[lab.timestamp < V.HIST_END]
    _, folds = V.make_splits(net, G, D["warm"], a.seed, V.N_REPORT, 4)
    pool = set().union(*folds)

    out = {k: [] for k in ("network median", "graph neighbours", "CF learned similarity",
                           "ORACLE best single warm pair")}
    for fi in range(a.folds):
        unseen = folds[fi]; seen = pool - unseen
        Rw = residual_panel(hist[hist.station_key.isin(seen)])
        Rc = residual_panel(hist[hist.station_key.isin(unseen)])
        Rw = Rw.loc[:, Rw.notna().sum() > 2000]
        Rc = Rc.loc[:, Rc.notna().sum() > 2000]
        if Rc.shape[1] == 0:
            continue
        Xw = pair_features(list(Rw.columns), statics, G, net)
        sim = pair_similarity_model(Rw, Xw, seed=a.seed)
        netmed = Rw.median(axis=1)

        sc = {k: [] for k in out}
        for pc in Rc.columns:
            y = Rc[pc].dropna()
            idx = y.index.intersection(Rw.index)
            if len(idx) < 500:
                continue
            y = y.loc[idx]
            base = float(np.abs(y).mean())          # predicting zero shock

            def gain(pred):                          # % of |shock| explained
                p = pd.Series(pred, index=idx).fillna(0.0)
                return 100 * (base - float(np.abs(y - p).mean())) / base

            sc["network median"].append(gain(netmed.reindex(idx).to_numpy()))

            st = pc.split("|")[0]
            nb = [c for c in Rw.columns if c.split("|")[0] in
                  (set(G.neighbors(st)) if st in G else set())]
            sc["graph neighbours"].append(
                gain(Rw[nb].reindex(idx).mean(axis=1).to_numpy() if nb
                     else netmed.reindex(idx).to_numpy()))

            xc = pair_features([pc], statics, G, net).iloc[0]
            s = predict_similarity(sim, Xw, xc)
            top = np.argsort(s)[::-1][:a.topk]
            w = np.clip(s[top], 0, None)
            if w.sum() <= 0:
                w = np.ones_like(w)
            pred = (Rw.iloc[:, top].reindex(idx).to_numpy() * w).sum(1) / w.sum()
            sc["CF learned similarity"].append(gain(pred))

            gains = [gain(Rw[c].reindex(idx).to_numpy()) for c in Rw.columns[:80]]
            sc["ORACLE best single warm pair"].append(max(gains))

        for k in out:
            if sc[k]:
                out[k].append(float(np.mean(sc[k])))
        log(f"fold {fi+1}/{a.folds}: {Rw.shape[1]} warm pairs -> {len(sc['network median'])} cold pairs")

    print("\n" + "=" * 72)
    print("% of the cold pair's contemporaneous shock explained (higher is better)")
    print("-" * 72)
    for k, v in out.items():
        v = np.array(v)
        print(f"{k:32s} {v.mean():7.2f}%  sd {v.std():5.2f}   {np.round(v, 1)}")
    print("-" * 72)
    print("0% = predicting no shock at all. ORACLE picks the best warm pair using truth.")
    Path(a.out).write_text(json.dumps({k: list(map(float, v)) for k, v in out.items()}, indent=2))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
