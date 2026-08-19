"""Ablation on the cold-start holdout to find which change regressed cold_sim_WAPE.

Toggles, independently:
  leak     : class shapes/levels fitted on ALL history (old, leaky) vs in-train only
  weights  : weighted adjacency (new) vs unweighted (old)
  peers    : peer-group shapes with aadt fallback (new) vs functional_class only (old)
  feats    : 8 node features (new) vs the original 6
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"d:\files_deldottraffic_joeal\Day1_Traffic_Solution")
sys.path.insert(0, str(ROOT / "src"))
import pipeline as P, gnn as G

train, val, net, edges, template = P.load_all(ROOT / "day1", apply_offset=False)
hist = pd.concat([train, val], ignore_index=True)
FLOOR = 20.0


def old_node_features(net, stations, idx):
    s = net.drop_duplicates("station_key").set_index(net.drop_duplicates("station_key")["station_key"].astype(str))
    F = np.zeros((len(stations), 6), float)
    for st, i in idx.items():
        r = s.loc[st]
        F[i, 0] = pd.to_numeric(r["through_lanes"], errors="coerce")
        F[i, 1] = pd.to_numeric(r["speed_limit"], errors="coerce")
        F[i, 2] = G._FC.get(r["functional_class"], 3)
        F[i, 3] = G._AADT.get(r["aadt_band"], 2)
        F[i, 4] = G._W.get(r["roadway_width_band"], 2)
        F[i, 5] = pd.to_numeric(r["lane_width"], errors="coerce")
    mu = np.nanmean(F, 0); sd = np.nanstd(F, 0) + 1e-9
    return np.nan_to_num((F - mu) / sd, nan=0.0)


def old_class_shapes(history, net):
    h = history.merge(net[["station_key", "functional_class"]], on="station_key", how="left")
    h["how"] = h["day_of_week"] * 24 + h["hour"]
    shapes = {}
    for fc, grp in h.groupby("functional_class"):
        prof = grp.groupby("how")["volume"].mean().reindex(range(168))
        prof = prof.fillna(prof.mean())
        shapes[fc] = (prof / prof.mean()).to_numpy()
    glob = h.groupby("how")["volume"].mean().reindex(range(168))
    shapes["__global__"] = (glob / glob.mean()).fillna(1.0).to_numpy()
    return shapes


def old_class_levels(history, net):
    lv = history.groupby("station_key")["volume"].mean().rename("lvl").reset_index()
    lv = lv.merge(net[["station_key", "functional_class"]], on="station_key", how="left")
    d = lv.groupby("functional_class")["lvl"].median().to_dict()
    d["__global__"] = float(lv["lvl"].median())
    return d


def old_blend(gnn_map, class_level, net, weight=0.5):
    fc = net.set_index("station_key")["functional_class"].to_dict()
    out = {}
    for stn, g in gnn_map.items():
        prior = class_level.get(fc.get(stn), class_level["__global__"])
        if not np.isfinite(prior):
            prior = class_level["__global__"]
        out[stn] = float(np.exp(weight * np.log(max(g, 1)) + (1 - weight) * np.log(max(prior, 1))))
    return out


def cold_forecast(levels, shapes, keymap, rows):
    r = P.add_calendar(rows)
    glob = shapes["__global__"]
    vals = [v for v in levels.values() if np.isfinite(v)]
    fb = float(np.mean(vals)) if vals else 1.0
    how = pd.to_numeric(r["how"], errors="coerce").fillna(0).to_numpy().astype(int) % 168
    stn = r["station_key"].astype(str).to_numpy()
    lvl = np.array([levels.get(s, np.nan) for s in stn], float)
    lvl = np.where(np.isfinite(lvl), lvl, fb)
    shp = np.array([shapes.get(keymap.get(s), glob)[h] for s, h in zip(stn, how)], float)
    return np.maximum(lvl * np.nan_to_num(shp, nan=1.0), 0.0)


def wape(a, p):
    a = np.asarray(a, float); return float(np.sum(np.abs(a - p)) / np.sum(a))


unw = edges[["from_station_key", "to_station_key"]]
stations, idx, A_new = G.build_graph(net, edges)
_, _, A_old = G.build_graph(net, unw)
X_new = G.node_features(net, stations, idx)
X_old = old_node_features(net, stations, idx)

lv = G.station_levels(hist)
y = np.full(len(stations), np.nan)
for st, i in idx.items():
    if st in lv.index:
        y[i] = np.log1p(lv[st])
known = ~np.isnan(y)
rng = np.random.default_rng(7); kn = np.where(known)[0]
hold = rng.choice(kn, size=int(0.2 * len(kn)), replace=False)
in_train = known.copy(); in_train[hold] = False
held = [stations[i] for i in hold]
sr = val[val["station_key"].isin(held)].copy()
act = sr[P.TARGET].to_numpy()
hist_sim = hist[~hist["station_key"].isin(held)]

peer_map = G.peer_groups(net)
fc_map = {str(k): v for k, v in net.set_index("station_key")["functional_class"].to_dict().items()}

print(f"held {len(held)} stations, {len(sr):,} rows, actual mean {act.mean():.1f}\n")
print(f"{'leak':>5} {'wts':>5} {'peers':>6} {'feats':>6} | {'lvl mean':>9} {'pred mean':>9} {'WAPE':>8}")
print("-" * 62)
rows = []
for leak, wts, peers, feats in itertools.product([1, 0], [0, 1], [0, 1], [0, 1]):
    h_use = hist if leak else hist_sim
    A = A_new if wts else A_old
    Xb = X_new if feats else X_old
    if peers:
        shapes = G.class_shapes(h_use, net); clv = G.class_levels(h_use, net); km = peer_map
        blend = G.blend_levels
    else:
        shapes = old_class_shapes(h_use, net); clv = old_class_levels(h_use, net); km = fc_map
        blend = old_blend
    Xa = G.augment_with_levels(Xb, np.nan_to_num(y), in_train)
    yf = np.nan_to_num(y, nan=float(np.nanmean(y[in_train])))
    g = G.train_gcn(A, Xa, yf, in_train, hid=16, epochs=500, lr=0.02, seed=0)
    sim_lv = np.expm1(g.forward(A, Xa))
    raw = {stations[i]: max(sim_lv[i], 1.0) for i in hold}
    lm = blend(raw, clv, net, weight=0.5)
    p = cold_forecast(lm, shapes, km, sr)
    w = wape(act, p)
    rows.append((leak, wts, peers, feats, w))
    print(f"{leak:>5} {wts:>5} {peers:>6} {feats:>6} | "
          f"{np.mean(list(raw.values())):9.1f} {p.mean():9.1f} {w:8.3f}")

best = min(rows, key=lambda r: r[-1])
print(f"\nbest: leak={best[0]} wts={best[1]} peers={best[2]} feats={best[3]} WAPE={best[4]:.3f}")
print("true station levels (held):", np.round(np.expm1(y[hold]), 0).astype(int).tolist())
