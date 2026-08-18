"""
Unified validation protocol  (v2 -- nested station CV, information-matched tasks).

One time split, mirroring the real data:

    HISTORICAL = calendar 2024        (what traffic_train.parquet is)
    FUTURE     = Jan-Jun 2025         (what traffic_validation.parquet is)

Station split -- nested, because 75 stations is not many:

    set C   15 stations, FIXED, never touched until the very end
    pool    the remaining 60, k-folded 4 ways for fit / select

    each fold:  45 stations "seen"  ->  15 stations "unseen"

The three tasks are separated by *what information is permitted*, not by which
period they are scored on. That is what the real slices actually differ by:

    warm-future   fit 2024 seen    -> predict 2025 H1, seen
                  a station we know, a period we do not.

    cold-interp   fit 2024 seen    -> predict 2024, unseen
                  a station never observed, INSIDE the labelled window, so its
                  neighbours are recording at the target hour. Interpolation.

    cold-future   fit 2024 seen    -> predict 2025 H1, unseen
                  a station never observed, AFTER the labelled window, so nothing
                  anywhere is recording. Forecasting.

The period relationship is what mirrors deployment: there the labelled window runs
2024-01..2025-06, the real backfill targets sit *inside* it and the real future
targets sit *after* it. Scoring cold-interp inside the training window and
cold-future after it reproduces exactly that.

cold-future is run as two arms so the chain can be measured rather than assumed:

    direct    attributes + graph donors + calendar
    chained   + pseudo-history reconstructed for the unseen stations by cold-interp

DelDOT permit the chained arm explicitly, with the caveat that model-generated
history is "an estimate rather than observed truth" -- so intervals are always
fitted on real held-out data, never on reconstructions.

Prediction-time rule: only information available at or before the target timestamp.

    ~/.virtualenvs/data_science/bin/python validate_protocol.py --folds 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import traffic_pipeline as T
import run_pipeline as R

T0 = time.time()
HIST_END = pd.Timestamp("2025-01-01")
N_REPORT = 15          # set C, fixed and untouched
N_FOLDS = 4            # over the remaining 60


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def score(y_log, pred_log) -> dict:
    """GEH first -- MAE alone says nothing about whether the model is usable."""
    y = np.expm1(np.asarray(y_log, float))
    lo, mid, hi = (np.expm1(pred_log[:, i]).clip(0) for i in range(3))
    geh = np.sqrt(2 * (mid - y) ** 2 / np.maximum(mid + y, 1e-9))
    ape = np.abs(mid - y) / np.maximum(y, 1e-9)
    a = 0.10
    return {
        "GEH<5_%": round(100 * float((geh < 5).mean()), 1),
        "GEH<10_%": round(100 * float((geh < 10).mean()), 1),
        "within20_%": round(100 * float((ape <= 0.20).mean()), 1),
        "MAE": round(float(np.abs(y - mid).mean()), 1),
        "sMAPE": round(float((2 * np.abs(y - mid) / (y + mid + 1e-9)).mean() * 100), 1),
        "R2": round(float(1 - np.sum((y - mid) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
        "n": int(len(y)),
    }


def stratified_pick(net, G, pool, n, seed):
    """Sample stations resembling the 43 real cold ones (corridor degree, AADT
    reliability, functional_class missingness -- the real cold set differs from the
    warm set mainly on the last)."""
    s = net.set_index("station_key")
    d = pd.DataFrame([{
        "station_key": st,
        "corr_deg": sum(1 for nb in G.neighbors(st)
                        if G.edges[st, nb]["etype"] == "same_corridor") if st in G else 0,
        "aadt_bad": int(s.aadt_band.get(st) in T.AADT_BAD),
        "fclass_missing": int(pd.isna(s.functional_class.get(st))),
    } for st in sorted(pool)])
    d["stratum"] = (d.corr_deg.clip(0, 2).astype(str) + d.aadt_bad.astype(str)
                    + d.fclass_missing.astype(str))
    rng = np.random.default_rng(seed)
    picked = []
    for _, g in d.groupby("stratum"):
        k = max(1, int(round(n * len(g) / len(d))))
        picked += list(rng.choice(g.station_key.to_numpy(), min(k, len(g)), replace=False))
    if len(picked) > n:
        picked = list(rng.choice(picked, n, replace=False))
    return set(picked)


FROZEN_SPLITS = Path(__file__).resolve().parent / "splits_frozen.json"


def make_splits(net, G, warm, seed=0, n_report=N_REPORT, n_folds=N_FOLDS):
    """Fixed report set, then a k-fold over everything else.

    Reads `splits_frozen.json` when it matches the requested configuration, so every model
    scores on byte-identical folds. Regenerating from the seed is the fallback only, and it
    is not reproducible across numpy versions -- `np.random.default_rng().permutation` is
    not a stability guarantee. `splits.json` in the repo root is an older, *different*
    partition (A_fit / B_select / C_report) that no code ever read; it does not describe
    these folds and must not be used to reproduce any published number.
    """
    if FROZEN_SPLITS.exists():
        js = json.loads(FROZEN_SPLITS.read_text())
        if js.get("seed") == seed and js.get("n_folds") == n_folds \
                and len(js.get("report_set", [])) == n_report:
            report = set(js["report_set"])
            folds = [set(f) for f in js["folds"]]
            if report | set().union(*folds) == set(warm):
                return report, folds
        log(f"   WARNING: {FROZEN_SPLITS.name} does not match this configuration "
            f"(seed={seed}, folds={n_folds}, report={n_report}); regenerating from the seed. "
            f"Results will NOT be comparable to the published numbers.")
    report = stratified_pick(net, G, warm, n_report, seed + 999)
    pool = sorted(warm - report)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(pool), n_folds)
    return report, [set(f) for f in folds]


# --------------------------------------------------------------------------- #
# one fold
# --------------------------------------------------------------------------- #

def _keep(store, name, rows_df, pred):
    if store is not None:
        store[name] = (rows_df, pred)


def volume_route(p_gnn, p_lgb, y_log, cut_log=np.log1p(250.0)):
    """GNN and LightGBM are complementary by volume -- the GNN wins on quiet links, LGBM
    on busy ones; per-row oracle selection reaches 84% GEH<5 on warm-future. Route on the
    *predicted* level (never the truth) so the rule is usable at inference time."""
    use_gnn = p_gnn[:, 1] < cut_log
    return np.where(use_gnn[:, None], p_gnn, p_lgb)


def run_fold(D, statics, G, axis, seen, unseen, seed=0, use_gnn=True, gnn_kw=None,
             preds_out=None, cold_arm=False):
    """Score one fold. If `preds_out` is a dict it is filled with
    {model_name: (rows_df, pred_array)} so callers can re-score on stress slices
    without refitting anything."""
    lab, net = D["labeled"], D["network"]
    hist_seen = lab[(lab.timestamp < HIST_END) & (lab.station_key.isin(seen))]
    fut_seen = lab[(lab.timestamp >= HIST_END) & (lab.station_key.isin(seen))]
    hist_unseen = lab[(lab.timestamp < HIST_END) & (lab.station_key.isin(unseen))]
    fut_unseen = lab[(lab.timestamp >= HIST_END) & (lab.station_key.isin(unseen))]

    # every model below is fitted on hist_seen only
    bb = T.Backbone().fit(hist_seen, axis)
    dm = T.DonorModel(G, net).fit(bb, hist_seen)
    dinfo = dm.station_levels(sorted(set(net.station_key)))
    lm, _ = R.fit_level_model(hist_seen, net, G, axis, statics, seed=seed)
    slev = R.predict_levels(lm, dinfo, statics, sorted(set(net.station_key)))
    gst = T.GraphShapeTransfer(G).fit(hist_seen)
    plt_ = T.PairLevelTransfer(G, net, statics=statics).fit(bb)      # direction-matched cold level
    cfs = T.CFShapeTransfer().fit(hist_seen, statics, G)   # factorised cold shape
    bb.transfer_season(list(hist_unseen.pair.unique()) + list(fut_unseen.pair.unique()), G)

    f, fc = R.feature_list(False), R.feature_list(True)
    Xtr = R.build_features(hist_seen, bb, statics, dinfo, None, warm=seen)
    ytr = hist_seen.ly.to_numpy() - Xtr.base.to_numpy()
    out = {}

    # ---- warm-future -------------------------------------------------------
    Xte = R.build_features(fut_seen, bb, statics, dinfo, None, warm=seen)
    out["warm-future_backbone"] = score(fut_seen.ly.to_numpy(), np.stack([Xte.base.to_numpy()] * 3, 1))
    p_warm_lgbm = R.fit_path(Xtr, ytr, Xte, f, seed=seed) + Xte.base.to_numpy()[:, None]
    out["warm-future_lgbm"] = score(fut_seen.ly.to_numpy(), p_warm_lgbm)
    _keep(preds_out, "warm-future_lgbm", fut_seen, p_warm_lgbm)

    # ---- cold-interp: INSIDE the labelled window, neighbours recording ------
    lvl_h = pd.Series({q: slev.get(q.split("|")[0], np.nan) for q in hist_unseen.pair.unique()})
    shp_h = gst.transfer(hist_unseen.pair.unique())
    hist_res = hist_seen[["timestamp", "station_key"]].copy()
    hist_res["res"] = hist_seen.ly.to_numpy() - bb.predict(hist_seen)

    Xtr_c = R.build_features(hist_seen, bb, statics, dinfo, None, hist_res, G, warm=seen)
    Xci = R.build_features(hist_unseen, bb, statics, dinfo, lvl_h, hist_res, G,
                           warm=seen, shape_override=shp_h)
    out["cold-interp_backbone"] = score(hist_unseen.ly.to_numpy(),
                                        np.stack([Xci.base.to_numpy()] * 3, 1))
    p_interp = R.fit_path(Xtr_c, ytr, Xci, fc, seed=seed) + Xci.base.to_numpy()[:, None]
    out["cold-interp_full"] = score(hist_unseen.ly.to_numpy(), p_interp)
    _keep(preds_out, "cold-interp_full", hist_unseen, p_interp)

    # same path, but the level comes from direction-matched pair donors instead of the
    # direction-averaged station level + LightGBM level model
    lvl_h2 = plt_.transfer(hist_unseen.pair.unique())
    Xci2 = R.build_features(hist_unseen, bb, statics, dinfo, lvl_h2, hist_res, G,
                            warm=seen, shape_override=shp_h)
    p_interp2 = R.fit_path(Xtr_c, ytr, Xci2, fc, seed=seed) + Xci2.base.to_numpy()[:, None]
    out["cold-interp_pairlevel"] = score(hist_unseen.ly.to_numpy(), p_interp2)
    _keep(preds_out, "cold-interp_pairlevel", hist_unseen, p_interp2)

    # ... and with the factorised shape instead of the graph-averaged one
    shp_cf = cfs.transfer(hist_unseen.pair.unique(), statics, G)
    Xci3 = R.build_features(hist_unseen, bb, statics, dinfo, lvl_h2, hist_res, G,
                            warm=seen, shape_override=shp_cf)
    p_interp3 = R.fit_path(Xtr_c, ytr, Xci3, fc, seed=seed) + Xci3.base.to_numpy()[:, None]
    out["cold-interp_pairlevel_cfshape"] = score(hist_unseen.ly.to_numpy(), p_interp3)
    _keep(preds_out, "cold-interp_pairlevel_cfshape", hist_unseen, p_interp3)

    # ---- cold-future: AFTER the labelled window, nothing recording ---------
    lvl = pd.Series({q: slev.get(q.split("|")[0], np.nan) for q in fut_unseen.pair.unique()})
    shp = gst.transfer(fut_unseen.pair.unique())
    Xcf = R.build_features(fut_unseen, bb, statics, dinfo, lvl, warm=seen, shape_override=shp)
    out["cold-future_backbone"] = score(fut_unseen.ly.to_numpy(),
                                        np.stack([Xcf.base.to_numpy()] * 3, 1))
    p = R.fit_path(Xtr, ytr, Xcf, f, seed=seed) + Xcf.base.to_numpy()[:, None]
    out["cold-future_direct"] = score(fut_unseen.ly.to_numpy(), p)
    _keep(preds_out, "cold-future_direct", fut_unseen, p)

    lvl2 = plt_.transfer(fut_unseen.pair.unique())
    Xcf2 = R.build_features(fut_unseen, bb, statics, dinfo, lvl2, warm=seen, shape_override=shp)
    out["cold-future_backbone_pairlevel"] = score(fut_unseen.ly.to_numpy(),
                                                  np.stack([Xcf2.base.to_numpy()] * 3, 1))
    p2 = R.fit_path(Xtr, ytr, Xcf2, f, seed=seed) + Xcf2.base.to_numpy()[:, None]
    out["cold-future_pairlevel"] = score(fut_unseen.ly.to_numpy(), p2)
    _keep(preds_out, "cold-future_pairlevel", fut_unseen, p2)

    shp_cf_f = cfs.transfer(fut_unseen.pair.unique(), statics, G)
    Xcf3 = R.build_features(fut_unseen, bb, statics, dinfo, lvl2, warm=seen,
                            shape_override=shp_cf_f)
    p3 = R.fit_path(Xtr, ytr, Xcf3, f, seed=seed) + Xcf3.base.to_numpy()[:, None]
    out["cold-future_pairlevel_cfshape"] = score(fut_unseen.ly.to_numpy(), p3)
    _keep(preds_out, "cold-future_pairlevel_cfshape", fut_unseen, p3)

    # chained arm: give the forecaster the reconstructed 2024 history as if it were
    # real training rows, then forecast 2025 H1. This is the KITS / MoGERNN pattern
    # and DelDOT permit it; whether it helps is measured, not assumed.
    pseudo = hist_unseen.copy()
    pseudo["ly"] = p_interp[:, 1]                       # reconstructed, not observed
    aug = pd.concat([hist_seen, pseudo], ignore_index=True)
    Xaug = R.build_features(aug, bb, statics, dinfo, lvl_h, warm=seen)
    yaug = aug.ly.to_numpy() - Xaug.base.to_numpy()
    p = R.fit_path(Xaug, yaug, Xcf, f, seed=seed) + Xcf.base.to_numpy()[:, None]
    out["cold-future_chained"] = score(fut_unseen.ly.to_numpy(), p)

    # ---- GNN on both regimes ----------------------------------------------
    if use_gnn:
        import gnn_v2 as V
        from gnn_baseline import pick_device
        dev = pick_device("auto")
        # Hand the GNN the same profile backbone LightGBM gets, computed per regime with
        # the same level overrides, so it predicts a residual instead of re-deriving the
        # level and profile from scratch inside its node-ID embedding.
        base_full = pd.Series(np.nan, index=lab.index, dtype=float)
        base_full.loc[hist_seen.index] = Xtr.base.to_numpy()
        base_full.loc[fut_seen.index] = Xte.base.to_numpy()
        base_full.loc[fut_unseen.index] = Xcf.base.to_numpy()
        base_full.loc[hist_unseen.index] = Xci.base.to_numpy()
        lab_g = lab.assign(base=base_full.fillna(0.0).to_numpy())
        P = V.prepare({**D, "labeled": lab_g}, statics)
        ts = P["ts"]
        keep = np.isin(P["station"], list(seen))
        rows = np.flatnonzero((ts < np.datetime64(HIST_END)) & keep)
        order = rows[np.argsort(ts[rows])]
        inner = int(len(order) * 0.9)
        dead = set(np.unique(P["node"][~keep]).tolist())
        m, hist, idav = V.train(P, order[:inner], order[inner:], dead, dev,
                                max_epochs=200, patience=12, seed=seed, verbose=False,
                                **(gnn_kw or {}))
        gnn_pred = {}
        for nm, rdf in (("warm-future", fut_seen), ("cold-future", fut_unseen)):
            pr = V.predict_rows(m, P, lab_g.loc[rdf.index], dev, idav).astype(np.float64)
            gnn_pred[nm] = pr
            out[f"{nm}_gnn"] = score(rdf.ly.to_numpy(), pr)
            _keep(preds_out, f"{nm}_gnn", rdf, pr)
        out["_gnn_epochs"] = len(hist)
        out["_gnn_val_geh5_%"] = max(h["val_geh5_%"] for h in hist)

        # Control arm in the ORIGINAL geometry: raw `ly` target (no backbone input),
        # unweighted log-space pinball, pinball-based early stopping. The residual target
        # and the sqrt-volume loss weighting both help warm-future and both appear to hurt
        # the cold slices -- for a zero-history station `base` rests on a donor-transferred
        # level, so anchoring to it propagates the donor's level error, and the weighting
        # concentrates loss exactly where cold level error is worst. The slices are routed
        # separately downstream, so each can take the configuration that suits it; this arm
        # measures the difference instead of assuming it.
        if cold_arm:
            P0 = V.prepare(D, statics)
            m0, h0, id0 = V.train(P0, order[:inner], order[inner:], dead, dev,
                                  max_epochs=200, patience=12, seed=seed, verbose=False,
                                  select="pinball", geh_loss_weight=False,
                                  **{k: v for k, v in (gnn_kw or {}).items()
                                     if k not in ("select", "geh_loss_weight")})
            for nm, rdf in (("warm-future", fut_seen), ("cold-future", fut_unseen)):
                pr0 = V.predict_rows(m0, P0, rdf, dev, id0).astype(np.float64)
                out[f"{nm}_gnn_v0"] = score(rdf.ly.to_numpy(), pr0)
                _keep(preds_out, f"{nm}_gnn_v0", rdf, pr0)

        # ---- ensembles on warm-future (70.5% of the scored rows) ------------
        pg, pl = gnn_pred["warm-future"], p_warm_lgbm
        yw = fut_seen.ly.to_numpy()
        out["warm-future_ens_mean"] = score(yw, 0.5 * pg + 0.5 * pl)
        p_routed = volume_route(pg, pl, yw)
        out["warm-future_ens_routed"] = score(yw, p_routed)
        _keep(preds_out, "warm-future_ens_routed", fut_seen, p_routed)
        # oracle: not a usable model, a ceiling on what any router could reach
        gg = np.sqrt(2 * (np.expm1(pg[:, 1]) - np.expm1(yw)) ** 2 / np.maximum(np.expm1(pg[:, 1]) + np.expm1(yw), 1e-9))
        gl = np.sqrt(2 * (np.expm1(pl[:, 1]) - np.expm1(yw)) ** 2 / np.maximum(np.expm1(pl[:, 1]) + np.expm1(yw), 1e-9))
        out["warm-future_ens_ORACLE"] = score(yw, np.where((gg < gl)[:, None], pg, pl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--gnn-hidden", type=int, default=96)
    ap.add_argument("--gnn-head", type=int, default=192)
    ap.add_argument("--gnn-layers", type=int, default=3)
    ap.add_argument("--gnn-select", default="geh", choices=["geh", "pinball"])
    ap.add_argument("--gnn-cold-arm", action="store_true",
                    help="also train a control GNN in the original geometry (no base "
                         "input, unweighted log-space pinball) to isolate the cold regression")
    ap.add_argument("--out", default="protocol_results.json")
    ap.add_argument("--report-set", action="store_true",
                    help="ALSO evaluate on the untouched set C. Use once, at the very end.")
    a = ap.parse_args()

    log("loading")
    D = T.load_all()
    G = T.build_graph(D["edges"], D["network"])
    axis = T.station_axis(D["labeled"], D["targets"])
    statics = R.static_table(D["network"], G)

    report, folds = make_splits(D["network"], G, D["warm"], a.seed, N_REPORT, a.folds)
    pool = set().union(*folds)
    log(f"set C (untouched): {len(report)} stations | CV pool: {len(pool)} in {a.folds} folds")

    runs = []
    for i, unseen in enumerate(folds):
        seen = pool - unseen
        log(f"--- fold {i+1}/{a.folds}: {len(seen)} seen -> {len(unseen)} unseen ---")
        r = run_fold(D, statics, G, axis, seen, unseen, a.seed, use_gnn=not a.no_gnn,
                     gnn_kw=dict(hidden=a.gnn_hidden, head=a.gnn_head,
                                 layers=a.gnn_layers, select=a.gnn_select),
                     cold_arm=a.gnn_cold_arm)
        runs.append(r)
        for k in sorted(k for k in r if not k.startswith("_")):
            log(f"   {k:26s} GEH<5 {r[k]['GEH<5_%']:5.1f}%  MAE {r[k]['MAE']:6.1f}")

    keys = sorted(k for k in runs[0] if not k.startswith("_"))
    print("\n" + "=" * 92)
    print(f"NESTED STATION CV  —  fit on 2024, predict Jan–Jun 2025, {a.folds} folds")
    print("=" * 92)
    print(f"{'task / model':28s} {'GEH<5':>8s} {'±sd':>6s} {'GEH<10':>8s} {'±20%':>7s} {'MAE':>8s} {'sMAPE':>7s} {'R2':>7s}")
    print("-" * 92)
    summary = {}
    for k in keys:
        v = {m: float(np.mean([r[k][m] for r in runs])) for m in ("GEH<5_%", "GEH<10_%", "within20_%", "MAE", "sMAPE", "R2")}
        v["GEH<5_sd"] = float(np.std([r[k]["GEH<5_%"] for r in runs]))
        summary[k] = {kk: round(vv, 2) for kk, vv in v.items()}
        print(f"{k:28s} {v['GEH<5_%']:7.1f}% {v['GEH<5_sd']:6.2f} {v['GEH<10_%']:7.1f}% "
              f"{v['within20_%']:6.1f}% {v['MAE']:8.1f} {v['sMAPE']:6.1f}% {v['R2']:7.3f}")
    print("-" * 92)
    print("industry calibration target: >=85% of links below GEH 5")
    print("cold-future: direct vs chained = does reconstructed history actually help?")
    Path(a.out).write_text(json.dumps(
        {"config": vars(a), "summary": summary, "folds": runs,
         "report_set": sorted(report)}, indent=2))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
