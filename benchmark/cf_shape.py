"""
Collaborative filtering for cold-start *shape*, scored against the measured ceiling.

Framing: station-direction pairs are users, the 168 (hour x dow) cells are items, the
normalised log profile is the rating. A zero-history station is a cold-start user, so the
usable family is hybrid matrix factorisation -- learn item structure from the warm pairs,
then map a cold pair's attributes and graph position onto that structure. DOT practice
calls the discrete version "factor groups" (FHWA TMG); it is the same idea.

Baselines and ceiling, both previously measured on the 4 protocol folds:

    GraphShapeTransfer (deployed)                     0.2795
    oracle assignment to 8 k-means profile clusters   0.2117   <- ceiling for this family

The ceiling matters: an oracle *shape* is worth only about +2 GEH points on cold-interp,
against +18.5 for an oracle level. So this is a bounded prize and the honest question is
whether any of it is reachable, not whether the idea is elegant.

    python cf_shape.py --folds 4
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

import traffic_pipeline as T
import run_pipeline as R
import validate_protocol as V

T0 = time.time()
HOURS, DOWS = 24, 7
NCELL = HOURS * DOWS


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def profile_matrix(lab: pd.DataFrame, pairs=None) -> pd.DataFrame:
    """(pair x 168) normalised log profile: mean deviation from the pair's own median."""
    d = lab if pairs is None else lab[lab.pair.isin(pairs)]
    med = d.groupby("pair").ly.transform("median")
    dev = (d.ly - med).groupby([d.pair, d.hour, d.dow]).mean()
    M = dev.unstack(level=[1, 2])
    full = pd.MultiIndex.from_product([range(HOURS), range(DOWS)])
    return M.reindex(columns=full)


def pair_features(pairs, statics, G, net) -> pd.DataFrame:
    """Everything a cold pair knows about itself: statics, direction, graph position."""
    st = [p.split("|")[0] for p in pairs]
    dc = [p.split("|")[1] for p in pairs]
    X = statics.reindex(st).reset_index(drop=True)
    X.index = pairs
    for c in ("1", "3", "5", "7"):
        X[f"dir_{c}"] = [int(d == c) for d in dc]
    X["n_corr"] = [sum(1 for n in G.neighbors(s) if G.edges[s, n]["etype"] == "same_corridor")
                   if s in G else 0 for s in st]
    X["n_prox"] = [sum(1 for n in G.neighbors(s) if G.edges[s, n]["etype"] == "proximity")
                   if s in G else 0 for s in st]
    return X.astype(float).fillna(-1.0)


# --------------------------------------------------------------------------- #
# estimators: each maps (warm profiles, features) -> profiles for cold pairs
# --------------------------------------------------------------------------- #

def est_global_mean(Mw, Xw, Xc, **kw):
    return pd.DataFrame(np.tile(Mw.mean(0).to_numpy(), (len(Xc), 1)),
                        index=Xc.index, columns=Mw.columns)


def est_direction_mean(Mw, Xw, Xc, **kw):
    """The hierarchy's fallback: average profile of the same direction_code."""
    dcw = np.array([p.split("|")[1] for p in Mw.index])
    out = {}
    for pr in Xc.index:
        dc = pr.split("|")[1]
        sel = Mw[dcw == dc]
        out[pr] = (sel.mean(0) if len(sel) else Mw.mean(0)).to_numpy()
    return pd.DataFrame(out, index=Mw.columns).T


def est_svd_regress(Mw, Xw, Xc, k=8, seed=0, **kw):
    """Hybrid MF: SVD the warm profiles, regress the factors on attributes+graph."""
    svd = TruncatedSVD(n_components=k, random_state=seed)
    Z = svd.fit_transform(Mw.to_numpy())
    pred = np.zeros((len(Xc), k))
    for j in range(k):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1)
        rf.fit(Xw.to_numpy(), Z[:, j])
        pred[:, j] = rf.predict(Xc.to_numpy())
    return pd.DataFrame(svd.inverse_transform(pred), index=Xc.index, columns=Mw.columns)


def est_cluster_soft(Mw, Xw, Xc, k=8, seed=0, **kw):
    """Soft factor groups: classify into profile clusters, return posterior-weighted centroid.

    Kept soft deliberately -- NCHRP warns that mis-assigning a site to the wrong factor
    group can triple the error, so a confident-but-wrong hard assignment is the failure
    mode to avoid.
    """
    from sklearn.ensemble import RandomForestClassifier
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Mw.to_numpy())
    if len(np.unique(km.labels_)) < 2:
        return est_global_mean(Mw, Xw, Xc)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                 random_state=seed, n_jobs=-1)
    clf.fit(Xw.to_numpy(), km.labels_)
    P = clf.predict_proba(Xc.to_numpy())
    cent = np.vstack([Mw.to_numpy()[km.labels_ == c].mean(0) for c in clf.classes_])
    return pd.DataFrame(P @ cent, index=Xc.index, columns=Mw.columns)


def est_feature_knn(Mw, Xw, Xc, k=10, seed=0, **kw):
    """CF by learned neighbourhood: nearest warm pairs in standardised feature space."""
    A = Xw.to_numpy(); B = Xc.to_numpy()
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A, B = (A - mu) / sd, (B - mu) / sd
    out = {}
    for i, pr in enumerate(Xc.index):
        d = np.linalg.norm(A - B[i], axis=1)
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 1e-6)
        out[pr] = (Mw.to_numpy()[idx] * w[:, None]).sum(0) / w.sum()
    return pd.DataFrame(out, index=Mw.columns).T


def est_graph_transfer(Mw, Xw, Xc, G=None, lab_seen=None, **kw):
    """The deployed GraphShapeTransfer, for reference."""
    gst = T.GraphShapeTransfer(G).fit(lab_seen)
    s = gst.transfer(list(Xc.index))
    out = {}
    for pr in Xc.index:
        v = s.loc[pr]
        out[pr] = v.reindex(pd.MultiIndex.from_product([range(HOURS), range(DOWS)])).to_numpy()
    return pd.DataFrame(out, index=Mw.columns).T


def est_graph_plus_cluster(Mw, Xw, Xc, G=None, lab_seen=None, k=8, seed=0, w=0.5, **kw):
    """Average of the graph transfer and the soft cluster centroid."""
    a = est_graph_transfer(Mw, Xw, Xc, G=G, lab_seen=lab_seen)
    b = est_cluster_soft(Mw, Xw, Xc, k=k, seed=seed)
    return w * a + (1 - w) * b


def est_oracle_cluster(Mw, Xw, Xc, Mc=None, k=8, seed=0, **kw):
    """CEILING, not a model: assign each cold pair to its own best cluster using truth."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Mw.to_numpy())
    cent = np.vstack([Mw.to_numpy()[km.labels_ == c].mean(0) for c in sorted(set(km.labels_))])
    out = {}
    for pr in Xc.index:
        t = Mc.loc[pr].to_numpy()
        err = np.nanmean(np.abs(cent - t), axis=1)
        out[pr] = cent[int(np.argmin(err))]
    return pd.DataFrame(out, index=Mw.columns).T


ESTIMATORS = {
    "global mean": est_global_mean,
    "direction mean": est_direction_mean,
    "GraphShapeTransfer (deployed)": est_graph_transfer,
    "CF: feature kNN": est_feature_knn,
    "CF: SVD + factor regression": est_svd_regress,
    "CF: soft factor groups": est_cluster_soft,
    "graph + soft factor groups": est_graph_plus_cluster,
    "ORACLE cluster assignment": est_oracle_cluster,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="cf_shape_results.json")
    a = ap.parse_args()

    D = T.load_all(); lab, net = D["labeled"], D["network"]
    G = T.build_graph(D["edges"], net)
    statics = R.static_table(net, G)
    hist = lab[lab.timestamp < V.HIST_END]
    _, folds = V.make_splits(net, G, D["warm"], a.seed, V.N_REPORT, 4)
    pool = set().union(*folds)

    res = {n: [] for n in ESTIMATORS}
    for fi in range(a.folds):
        unseen = folds[fi]
        seen = pool - unseen
        h_seen = hist[hist.station_key.isin(seen)]
        h_cold = hist[hist.station_key.isin(unseen)]
        Mw = profile_matrix(h_seen).dropna(thresh=int(0.8 * NCELL))
        Mc = profile_matrix(h_cold).reindex(columns=Mw.columns)
        Mw = Mw.fillna(0.0)
        Mc = Mc.dropna(thresh=int(0.8 * NCELL))
        if not len(Mc):
            continue
        Xw = pair_features(list(Mw.index), statics, G, net)
        Xc = pair_features(list(Mc.index), statics, G, net)

        for name, fn in ESTIMATORS.items():
            P = fn(Mw, Xw, Xc, G=G, lab_seen=h_seen, Mc=Mc, k=a.k, seed=a.seed)
            P = P.reindex(index=Mc.index, columns=Mc.columns)
            err = float(np.nanmean(np.abs(P.to_numpy() - Mc.to_numpy())))
            res[name].append(err)
        log(f"fold {fi+1}/{a.folds}: {len(Mw)} warm pairs -> {len(Mc)} cold pairs")

    print("\n" + "=" * 74)
    print(f"{'shape estimator':36s} {'mean |err|':>11s} {'sd':>7s}   per-fold")
    print("-" * 74)
    base = float(np.mean(res["GraphShapeTransfer (deployed)"]))
    for name in ESTIMATORS:
        v = np.array(res[name])
        tag = "" if name.startswith("ORACLE") else f"  ({100*(base-v.mean())/base:+.1f}%)"
        print(f"{name:36s} {v.mean():11.4f} {v.std():7.4f}   {np.round(v, 3)}{tag}")
    print("-" * 74)
    print("deployed = GraphShapeTransfer; % is improvement over it. ORACLE is a ceiling.")
    Path(a.out).write_text(json.dumps({k: list(map(float, v)) for k, v in res.items()}, indent=2))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
