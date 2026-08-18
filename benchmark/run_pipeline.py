"""
Orchestration for the slice-routed pipeline.

    validate -- run the temporal / cold-future / cold-backfill folds and print a
                per-slice dashboard (research doc section 9)
    submit   -- fit on everything, calibrate, write a validated submission

    ~/.virtualenvs/data_science/bin/python run_pipeline.py validate
    ~/.virtualenvs/data_science/bin/python run_pipeline.py submit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import traffic_pipeline as T

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

STATIC_COLS = ["through_lanes", "speed_limit", "lane_width", "aadt_ord", "aadt_bad",
               "fclass", "surface", "rwidth", "degree", "corr_degree"]

CAL_COLS = ["hour", "dow", "month", "doy", "is_weekend", "is_holiday", "holiday_adj",
            "t_years", "horizon_days"]

DONOR_COLS = ["donor_n", "donor_spread", "corridor_w", "nbr_w", "has_history"]

NBR_COLS = ["nbr_net", "nbr_corr", "nbr_prox", "nbr_n", "nbr_spread"]


def static_table(net: pd.DataFrame, G) -> pd.DataFrame:
    s = net.set_index("station_key").copy()
    s["aadt_ord"] = s.aadt_band.map(T.AADT_ORDER).astype(float)
    s["aadt_bad"] = s.aadt_band.isin(T.AADT_BAD).astype(int)
    s.loc[s.aadt_bad == 1, "aadt_ord"] = np.nan          # unreliable -> missing
    s["fclass"] = s.functional_class.astype("category").cat.codes
    s["surface"] = s.surface_type.astype("category").cat.codes
    s["rwidth"] = s.roadway_width_band.map({"<24": 0, "24-35": 1, "36-47": 2, "48-59": 3, "60+": 4}).astype(float)
    s["degree"] = [G.degree(i) if i in G else 0 for i in s.index]
    s["corr_degree"] = [
        sum(1 for n in G.neighbors(i) if G.edges[i, n]["etype"] == "same_corridor") if i in G else 0
        for i in s.index
    ]
    return s[STATIC_COLS]


def build_features(rows, bb, statics, donor_info, level_override, lab_res=None, G=None, warm=None,
                   shape_override=None):
    X = rows[["hour", "dow", "month", "doy", "is_weekend", "is_holiday", "holiday_adj", "t_years"]].copy()
    X["horizon_days"] = ((rows.timestamp - T.LAST_LABEL).dt.total_seconds() / 86400).clip(lower=0).to_numpy()
    X["base"] = bb.predict(rows, level_override=level_override, shape_override=shape_override)

    st = statics.reindex(rows.station_key).reset_index(drop=True)
    st.index = rows.index
    X = pd.concat([X, st], axis=1)

    if donor_info is not None:
        d = donor_info.reindex(rows.station_key)[["donor_n", "donor_spread", "corridor_w", "nbr_w"]]
        d.index = rows.index
        X = pd.concat([X, d], axis=1)
    else:
        for c in ["donor_n", "donor_spread", "corridor_w", "nbr_w"]:
            X[c] = np.nan
    X["has_history"] = rows.station_key.isin(warm).astype(int).to_numpy() if warm is not None else 1

    if lab_res is not None and G is not None:
        nb = T.neighbour_features(rows, lab_res, G)
        X = pd.concat([X, nb], axis=1)
    else:
        for c in NBR_COLS:
            X[c] = np.nan
    return X


def feature_list(use_nbr: bool) -> list[str]:
    f = CAL_COLS + ["base"] + STATIC_COLS + DONOR_COLS
    return f + NBR_COLS if use_nbr else f


# --------------------------------------------------------------------------- #
# learned cold-start level model
# --------------------------------------------------------------------------- #

LEVEL_FEATS = ["donor_level", "donor_n", "donor_spread", "corridor_w", "nbr_w"] + STATIC_COLS


def _station_levels_from(bb) -> pd.Series:
    pl = bb.level.rename("lv").reset_index()
    pl["station_key"] = pl.pair.str.split("|").str[0]
    return pl.groupby("station_key").lv.mean()


def fit_level_model(lab, net, G, axis, statics, n_folds=5, seed=0):
    """Predict a cold station's log level from graph donors + static attributes.

    The hand-weighted donor blend reaches log-MAE 0.372 on held-out stations
    (a typical x1.32 multiplicative level error), and that level error -- not the
    temporal deviation -- is what dominates cold-slice MAE. So the blend becomes a
    *feature* and a gradient-boosted model learns the combination instead.

    Cross-fitted over stations: donor features for a station are always computed
    from a DonorModel that never saw that station, so the training features match
    what a genuinely unseen station will get.
    """
    import lightgbm as lgb

    stations = np.array(sorted(set(lab.station_key)))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(stations), n_folds)

    parts = []
    for f in folds:
        held = set(f)
        sub = lab[~lab.station_key.isin(held)]
        bb_f = T.Backbone().fit(sub, axis)
        dm_f = T.DonorModel(G, net).fit(bb_f, sub)
        di = dm_f.station_levels(sorted(held)).rename(columns={"level": "donor_level"})
        parts.append(di)
    Xs = pd.concat(parts)
    Xs = Xs.join(statics)
    y = _station_levels_from(T.Backbone().fit(lab, axis)).reindex(Xs.index)
    ok = y.notna()
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_child_samples=5, subsample=0.9, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(Xs.loc[ok, LEVEL_FEATS], y[ok])
    oof_mae = float(np.abs(m.predict(Xs.loc[ok, LEVEL_FEATS]) - y[ok]).mean())
    return m, oof_mae


def predict_levels(level_model, donor_info, statics, stations):
    d = donor_info.reindex(stations).rename(columns={"level": "donor_level"}).join(statics)
    pred = level_model.predict(d[LEVEL_FEATS])
    # Blend with the transparent donor estimate; the doc recommends keeping the
    # interpretable path alive as a fallback and their disagreement is informative.
    return pd.Series(0.5 * pred + 0.5 * d.donor_level.to_numpy(), index=stations)


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #

def fit_path(Xtr, ytr, Xte, feats, n_est=120, seed=0):
    """Residual quantile learner on top of the backbone.

    350 trees was tuned against a backbone with no seasonal shape term. Once the backbone
    models seasonality (`Backbone.month_kernel`) the residual is smaller and 350 trees
    overfit it: on the deployment arm the deployed setting adds nothing over the backbone
    on the May-Jun window (82.2% both). Retuned across two windows, 120 trees is best on
    GEH and MAE alike -- mean 83.25% / 92.4 against 82.95% / 93.6 at n=350 and 82.15% /
    95.0 for the backbone alone. The margin over 350 is inside noise; the margin over
    doing nothing is not.
    """
    return T.lgb_quantiles(Xtr, ytr, Xte, feats, n_estimators=n_est, seed=seed)


def run_validation(D, seed=0, n_cold=18, use_gnn=True):
    lab, tgt, net, edges, warm = D["labeled"], D["targets"], D["network"], D["edges"], D["warm"]
    G = T.build_graph(edges, net)
    axis = T.station_axis(lab, tgt)
    statics = static_table(net, G)
    rng = np.random.default_rng(seed)
    results = {}

    # ---------------- fold 1: warm temporal (slice A analogue) ----------------
    log("fold A: warm temporal (train < 2025-05-01, test May-Jun 2025)")
    tr = lab[lab.timestamp < "2025-05-01"]
    te = lab[lab.timestamp >= "2025-05-01"]
    bb = T.Backbone().fit(tr, axis)
    dm = T.DonorModel(G, net).fit(bb, tr)
    dinfo = dm.station_levels(sorted(set(net.station_key)))
    f = feature_list(False)
    Xtr = build_features(tr, bb, statics, dinfo, None, warm=warm)
    Xte = build_features(te, bb, statics, dinfo, None, warm=warm)
    base_mae = float(np.abs(te.volume.to_numpy() - np.expm1(Xte.base)).mean())
    pred = fit_path(Xtr, tr.ly.to_numpy() - Xtr.base.to_numpy(), Xte, f, seed=seed)
    pred = pred + Xte.base.to_numpy()[:, None]
    # Slice A ships mean(GNN, LightGBM), so calibrate on that same blend -- conformal
    # offsets and the reliability model fitted on LightGBM alone would be calibrated for a
    # model we do not ship, and the blend has smaller residuals, so the intervals would
    # come out systematically too wide.
    if use_gnn:
        import gnn_v2 as V
        from gnn_baseline import pick_device
        dev_v = pick_device("auto")
        lab_v = lab.assign(base=np.nan)
        lab_v.loc[tr.index, "base"] = Xtr.base.to_numpy()
        lab_v.loc[te.index, "base"] = Xte.base.to_numpy()
        lab_v["base"] = lab_v["base"].fillna(0.0)
        Pv = V.prepare({**D, "labeled": lab_v}, statics)
        tsv = Pv["ts"]
        rowsv = np.flatnonzero(tsv < np.datetime64(pd.Timestamp("2025-05-01")))
        orderv = rowsv[np.argsort(tsv[rowsv])]
        innerv = int(len(orderv) * 0.92)
        mv, hv, idv = V.train(Pv, orderv[:innerv], orderv[innerv:], set(), dev_v,
                              hidden=192, head=384, max_epochs=200, patience=12,
                              seed=seed, verbose=False, select="geh")
        pg = V.predict_rows(mv, Pv, lab_v.loc[te.index], dev_v, idv).astype(np.float64)
        pred = 0.5 * pred + 0.5 * pg
        log(f"   fold-A GNN converged in {len(hv)} epochs; calibrating on the blend")
    results["A_warm"] = {"backbone_only_MAE": round(base_mae, 1), **T.metrics(te.ly.to_numpy(), pred)}
    A_pred, A_y = pred, te.ly.to_numpy()
    log(f"   backbone {base_mae:.1f} -> +LGBM {results['A_warm']['MAE']:.1f}  cov {results['A_warm']['cov90']:.3f}")

    # ---------------- fold 2: cold future (slice B analogue) ----------------
    log("fold B: cold future (whole stations held out, future window)")
    stations = np.array(sorted(warm))
    hold = set(rng.choice(stations, size=n_cold, replace=False))
    tr2 = lab[(lab.timestamp < "2025-05-01") & (~lab.station_key.isin(hold))]
    te2 = lab[(lab.timestamp >= "2025-05-01") & (lab.station_key.isin(hold))]
    bb2 = T.Backbone().fit(tr2, axis)
    dm2 = T.DonorModel(G, net).fit(bb2, tr2)
    dinfo2 = dm2.station_levels(sorted(set(net.station_key)))
    # same cold components the submission uses, so the conformal offsets fitted here
    # describe the residuals of the model we actually ship
    lvl2 = T.PairLevelTransfer(G, net, statics=statics).fit(bb2).transfer(te2.pair.unique())
    shp2 = T.CFShapeTransfer().fit(tr2, statics, G).transfer(te2.pair.unique(), statics, G)
    bb2.transfer_season(te2.pair.unique(), G)
    Xtr2 = build_features(tr2, bb2, statics, dinfo2, None, warm=set(tr2.station_key))
    Xte2 = build_features(te2, bb2, statics, dinfo2, lvl2, warm=set(tr2.station_key), shape_override=shp2)
    base_mae2 = float(np.abs(te2.volume.to_numpy() - np.expm1(Xte2.base)).mean())
    pred2 = fit_path(Xtr2, tr2.ly.to_numpy() - Xtr2.base.to_numpy(), Xte2, f, seed=seed)
    pred2 = pred2 + Xte2.base.to_numpy()[:, None]
    results["B_cold_future"] = {"donor_only_MAE": round(base_mae2, 1), **T.metrics(te2.ly.to_numpy(), pred2)}
    B_pred, B_y = pred2, te2.ly.to_numpy()
    log(f"   donor baseline {base_mae2:.1f} -> +LGBM {results['B_cold_future']['MAE']:.1f}  cov {results['B_cold_future']['cov90']:.3f}")

    # ---------------- fold 3: cold backfill (slice C analogue) ----------------
    log("fold C: cold backfill (held-out stations, neighbours observed at same hour)")
    tr3 = lab[~lab.station_key.isin(hold)]
    te3 = lab[lab.station_key.isin(hold)]
    bb3 = T.Backbone().fit(tr3, axis)
    dm3 = T.DonorModel(G, net).fit(bb3, tr3)
    dinfo3 = dm3.station_levels(sorted(set(net.station_key)))
    lvl3 = T.PairLevelTransfer(G, net, statics=statics).fit(bb3).transfer(te3.pair.unique())
    shp3 = T.CFShapeTransfer().fit(tr3, statics, G).transfer(te3.pair.unique(), statics, G)
    bb3.transfer_season(te3.pair.unique(), G)

    lab_res = tr3[["timestamp", "station_key"]].copy()
    lab_res["res"] = tr3.ly.to_numpy() - bb3.predict(tr3)
    log("   building neighbour residual features...")
    fc = feature_list(True)
    Xtr3 = build_features(tr3, bb3, statics, dinfo3, None, lab_res, G, warm=set(tr3.station_key))
    Xte3 = build_features(te3, bb3, statics, dinfo3, lvl3, lab_res, G, warm=set(tr3.station_key),
                          shape_override=shp3)
    base_mae3 = float(np.abs(te3.volume.to_numpy() - np.expm1(Xte3.base)).mean())

    # with and without the contemporaneous neighbour block, to isolate its value
    p_no = fit_path(Xtr3, tr3.ly.to_numpy() - Xtr3.base.to_numpy(), Xte3, f, seed=seed) + Xte3.base.to_numpy()[:, None]
    p_yes = fit_path(Xtr3, tr3.ly.to_numpy() - Xtr3.base.to_numpy(), Xte3, fc, seed=seed) + Xte3.base.to_numpy()[:, None]
    results["C_cold_backfill_calendar_only"] = {"donor_only_MAE": round(base_mae3, 1), **T.metrics(te3.ly.to_numpy(), p_no)}
    results["C_cold_backfill_with_neighbours"] = T.metrics(te3.ly.to_numpy(), p_yes)
    C_pred, C_y = p_yes, te3.ly.to_numpy()
    log(f"   donor {base_mae3:.1f} | calendar-only {results['C_cold_backfill_calendar_only']['MAE']:.1f}"
        f" | +neighbours {results['C_cold_backfill_with_neighbours']['MAE']:.1f}")

    # ---------------- conformal ----------------
    log("fitting Mondrian conformal scales")
    scales = {}
    for name, pr, yy in (("A", A_pred, A_y), ("B", B_pred, B_y), ("C", C_pred, C_y)):
        grp = np.char.add(name + "_", T.volume_band(pr[:, 1]))
        # Additive split-conformal everywhere: bounded by the observed residuals,
        # so a collapsed raw interval cannot explode into a useless width.
        sc = T.fit_conformal_additive(pr, yy, grp)
        scales.update({k: v for k, v in sc.items() if k != "__global__"})
        scales[f"{name}__global__"] = sc["__global__"]
        after = T.metrics(yy, T.apply_conformal_additive(pr, grp, sc))
        results[f"{name}_conformal"] = {"cov90": after["cov90"], "width": round(after["width"], 1),
                                        "winkler": round(after["winkler"], 1)}
        log(f"   slice {name}: cov {T.metrics(yy, pr)['cov90']:.3f} -> {after['cov90']:.3f}"
            f" (offsets {sc['__global__'][0]:+.2f}/{sc['__global__'][1]:+.2f} log)")

    # Per-fold history counts. The held-out stations in folds B and C stand in for
    # zero-history stations, so their `pair_n` must be counted from that fold's TRAINING
    # rows only -- which makes it 0, exactly as it will be for a real cold pair. Counting
    # from the full panel instead gives them thousands of rows at fit time and zero at
    # serve time, and the reliability model then extrapolates wildly (it collapsed cold
    # reliability to 0.06 against a realistic 0.3-0.5).
    n_a = tr.groupby("pair", observed=True).size()
    n_b = tr2.groupby("pair", observed=True).size()
    n_c = tr3.groupby("pair", observed=True).size()
    oof = {"A": (A_pred, A_y, te, n_a), "B": (B_pred, B_y, te2, n_b),
           "C": (C_pred, C_y, te3, n_c)}
    return results, scales, oof, {"hold": hold, "G": G, "axis": axis, "statics": statics}


# --------------------------------------------------------------------------- #
# reliability meta-model
# --------------------------------------------------------------------------- #

RELIABILITY_FEATS = ["slice_A", "slice_B", "slice_C", "pred", "rel_width", "log_pred",
                     "hour", "dow", "month", "is_weekend", "is_holiday", "night", "pair_n"]


def reliability_frame(pr, rows, sl, hist_n=None) -> pd.DataFrame:
    """Features for the reliability model.

    The original version saw only the slice, the prediction and the interval width. The
    robustness slices say that is the wrong feature set: the model's accuracy varies
    enormously by hour (night 60.4% GEH<5 vs 72.3% overall), by holiday (65.2%), by season
    (winter 67.2%) and by how much history the pair has (65.3%) -- all knowable in advance
    and none of it visible to the old feature set.
    """
    lo, mid, hi = (np.expm1(pr[:, i]) for i in range(3))
    n = pd.Series(rows.pair.to_numpy()).map(hist_n).to_numpy() if hist_n is not None \
        else np.zeros(len(rows))
    return pd.DataFrame({
        "slice_A": int(sl == "A"), "slice_B": int(sl == "B"), "slice_C": int(sl == "C"),
        "pred": mid, "rel_width": (hi - lo) / (mid + 10.0), "log_pred": pr[:, 1],
        "hour": rows.hour.to_numpy(), "dow": rows.dow.to_numpy(),
        "month": rows.month.to_numpy(), "is_weekend": rows.is_weekend.to_numpy(),
        "is_holiday": rows.is_holiday.to_numpy(),
        "night": np.isin(rows.hour.to_numpy(), (0, 1, 2, 3, 4, 5)).astype(int),
        "pair_n": np.nan_to_num(n),
    })


def fit_reliability(oof: dict, threshold=0.30):
    """P(sAPE <= threshold), learned from out-of-fold predictions (research doc s.8)."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    rows = []
    for sl, (pr, yy, rf, hn) in oof.items():
        y = np.expm1(yy)
        mid = np.expm1(pr[:, 1])
        sape = 2 * np.abs(y - mid) / (np.abs(y) + np.abs(mid) + 1e-9)
        fr = reliability_frame(pr, rf, sl, hn)
        fr["ok"] = (sape <= threshold).astype(int)
        rows.append(fr)
    d = pd.concat(rows, ignore_index=True)
    feats = RELIABILITY_FEATS
    n = len(d)
    idx = np.random.default_rng(0).permutation(n)
    tr_i, ca_i = idx[: int(n * 0.7)], idx[int(n * 0.7):]
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.07, num_leaves=31,
                           min_child_samples=100, random_state=0, verbose=-1, n_jobs=-1)
    m.fit(d.iloc[tr_i][feats], d.iloc[tr_i].ok)
    raw = m.predict_proba(d.iloc[ca_i][feats])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, d.iloc[ca_i].ok)

    from sklearn.metrics import brier_score_loss, roc_auc_score
    cal = iso.predict(raw)
    stats = {"base_rate": float(d.ok.mean()), "AUC": float(roc_auc_score(d.iloc[ca_i].ok, raw)),
             "Brier": float(brier_score_loss(d.iloc[ca_i].ok, cal))}
    return (m, iso, feats), stats


def apply_reliability(model, pred_log, slices, rows=None, hist_n=None):
    m, iso, feats = model
    parts = []
    for sl in ("A", "B", "C"):
        sel = slices == sl
        if not sel.any():
            continue
        fr = reliability_frame(pred_log[sel], rows[sel], sl, hist_n)
        fr["_i"] = np.flatnonzero(sel)
        parts.append(fr)
    d = pd.concat(parts).sort_values("_i")
    out = np.clip(iso.predict(m.predict_proba(d[feats])[:, 1]), 0.0, 1.0)
    res = np.zeros(len(pred_log))
    res[d["_i"].to_numpy()] = out
    return res


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

COLD_BLEND = 0.5     # weight on the LightGBM residual model for slices B/C


def main(args):
    log("loading data")
    D = T.load_all()
    results, scales, oof, aux = run_validation(D, seed=args.seed)

    log("fitting reliability meta-model")
    rel_model, rel_stats = fit_reliability(oof)
    # At serve time a warm pair's history is everything we hold; a cold pair has none,
    # which is what the fold-B/C training rows encoded.
    hist_n = D["labeled"].groupby("pair", observed=True).size()
    results["reliability"] = {k: round(v, 4) for k, v in rel_stats.items()}
    log(f"   base rate {rel_stats['base_rate']:.3f} | AUC {rel_stats['AUC']:.3f} | Brier {rel_stats['Brier']:.4f}")

    print("\n" + "=" * 78)
    print("VALIDATION DASHBOARD")
    print("=" * 78)
    print(json.dumps(results, indent=2))
    Path("validation_results.json").write_text(json.dumps(results, indent=2))

    if args.mode == "validate":
        return

    # ------------------------------------------------------------------ submit
    from gnn_baseline import pick_device
    device_gnn = pick_device("auto")
    log("\nrefitting on all labeled data for submission")
    lab, tgt, net, edges, warm = D["labeled"], D["targets"], D["network"], D["edges"], D["warm"]
    G, axis, statics = aux["G"], aux["axis"], aux["statics"]

    bb = T.Backbone().fit(lab, axis)
    dm = T.DonorModel(G, net).fit(bb, lab)
    cold_st = sorted(set(tgt.station_key) - warm)
    dinfo = dm.station_levels(sorted(set(net.station_key)))
    # Cold level comes from direction-matched pair donors. The old path (station-level
    # donor blend + LightGBM level model) averaged a station's two directions into one
    # number; on the 4 protocol folds this change is worth +5.7 GEH<5 on cold-interp and
    # +3.2 on cold-future, positive on every fold, with MAE down on both.
    plt_ = T.PairLevelTransfer(G, net, statics=statics).fit(bb)
    lvl = plt_.transfer(tgt[tgt.slice.isin(["B", "C"])].pair.unique())
    log(f"   pair-level transfer for {len(lvl)} cold pairs "
        f"(held-out log-MAE 0.338 vs 0.394 for the old level path)")

    # Cold shape by hybrid matrix factorisation rather than graph averaging. On the
    # 4-fold protocol it is better on the mean and markedly steadier across folds:
    # cold-interp 43.1% (sd 6.88) -> 43.6% (sd 5.11), cold-future 37.0% (sd 6.32) ->
    # 37.2% (sd 5.24). The mean gain is small because an oracle shape is only worth about
    # +2 GEH points on these slices; the variance reduction is the real prize, since
    # cold-slice fold spread is what makes every result there hard to read.
    cold_pairs = tgt[tgt.slice.isin(["B", "C"])].pair.unique()
    shp = T.CFShapeTransfer().fit(lab, statics, G).transfer(cold_pairs, statics, G)
    # Cold pairs have no `shape_month` entry, so they were getting a zero seasonal
    # correction while warm pairs got one worth about +4.9 GEH points. Borrow it from
    # direction-matched donors, the same way the hour x dow shape is borrowed.
    bb.transfer_season(cold_pairs, G)
    log(f"   factorised shape transferred for {len(cold_pairs)} cold pairs")

    lab_res = lab[["timestamp", "station_key"]].copy()
    lab_res["res"] = lab.ly.to_numpy() - bb.predict(lab)

    log("building training features")
    f, fc = feature_list(False), feature_list(True)
    Xtr = build_features(lab, bb, statics, dinfo, None, warm=warm)
    ytr = lab.ly.to_numpy() - Xtr.base.to_numpy()
    Xtr_c = build_features(lab, bb, statics, dinfo, None, lab_res, G, warm=warm)

    # Slice A goes to the GNN: measured 95.3 MAE vs 102.6 for the LightGBM path.
    # Slices B/C stay on the pipeline: the GNN scores 409.9 there vs 285.7, because
    # its node-identity pathway is undefined for stations it never saw.
    log("training GNN v2 for slice A")
    import gnn_v2 as V
    # Give the GNN the same backbone LightGBM gets, so it predicts a residual. This is the
    # configuration measured at 72.3% GEH<5 on the 4-fold protocol; without `base` it is a
    # different model than the one we validated.
    lab_g = lab.assign(base=Xtr.base.to_numpy())
    P = V.prepare({**D, "labeled": lab_g}, statics)
    ts_all = P["ts"]
    order = np.argsort(ts_all)
    inner = int(len(order) * 0.92)
    gm, ghist, gid = V.train(P, order[:inner], order[inner:], set(), device_gnn,
                             max_epochs=200, patience=12, seed=args.seed, verbose=False)
    log(f"   GNN converged in {len(ghist)} epochs, "
        f"best val GEH<5 {max(h['val_geh5_%'] for h in ghist):.1f}% "
        f"(pinball {min(h['val_pinball'] for h in ghist):.4f})")

    out = np.zeros((len(tgt), 3))
    for sl in ["A", "B", "C"]:
        rows = tgt[tgt.slice == sl]
        if not len(rows):
            continue
        log(f"predicting slice {sl} ({len(rows):,} rows)")
        use_nbr = sl == "C"
        Xte = build_features(rows, bb, statics, dinfo, lvl if sl != "A" else None,
                             lab_res if use_nbr else None, G if use_nbr else None, warm=warm,
                             shape_override=shp if sl != "A" else None)
        if sl == "A":
            # Blend the GNN with the LightGBM residual path. On the deployment arm
            # (fit 2024 + Jan-Apr 2025, score May-Jun 2025) the plain mean beats either
            # alone: backbone 82.7% GEH<5 / MAE 93.4, +LightGBM 82.9% / 92.0, GNN 83.4% /
            # 90.9, mean 83.5% / 90.7.
            p_gnn = V.predict_rows(gm, P, rows.assign(base=Xte.base.to_numpy()),
                                   device_gnn, gid).astype(np.float64)
            p_lgb = fit_path(Xtr, ytr, Xte, f, seed=args.seed) + Xte.base.to_numpy()[:, None]
            p = 0.5 * p_gnn + 0.5 * p_lgb
        else:
            p = fit_path(Xtr_c if use_nbr else Xtr, ytr, Xte, fc if use_nbr else f, seed=args.seed)
            p = p + Xte.base.to_numpy()[:, None]
            # On cold slices the residual learner is fitted on warm rows and does not
            # transfer: measured 285.7 vs 277.3 for the donor baseline alone on B.
            # Blend back toward the baseline rather than trusting it outright.
            b = Xte.base.to_numpy()[:, None]
            p = COLD_BLEND * p + (1 - COLD_BLEND) * np.concatenate(
                [p[:, :1] - p[:, 1:2] + b, b, p[:, 2:] - p[:, 1:2] + b], axis=1)
        grp = np.char.add(sl + "_", T.volume_band(p[:, 1]))
        sc = {k: v for k, v in scales.items() if not k.endswith("__global__")}
        sc["__global__"] = scales[f"{sl}__global__"]
        p = T.apply_conformal_additive(p, grp, sc)
        out[tgt.slice.to_numpy() == sl] = p

    log("writing submission")
    rel = apply_reliability(rel_model, out, tgt.slice.to_numpy(), rows=tgt, hist_n=hist_n)
    lo, mid, hi = (np.expm1(out[:, i]).clip(0) for i in range(3))
    lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)

    sub = pd.read_csv(T.DATA_DIR / "submission_template.csv", low_memory=False)
    sub["forecast_volume"], sub["lower_90"], sub["upper_90"] = mid, lo, hi
    sub["reliability_score"] = rel
    sub.to_csv(args.out, index=False)
    log(f"wrote {args.out}  ({len(sub):,} rows)")

    print("\nper-slice summary:")
    print(pd.DataFrame({"slice": tgt.slice, "forecast": mid, "width": hi - lo, "rel": rel})
          .groupby("slice").agg(rows=("forecast", "size"), mean_forecast=("forecast", "mean"),
                                mean_width=("width", "mean"), mean_rel=("rel", "mean")).round(2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["validate", "submit"])
    p.add_argument("--out", default="submission_pipeline.csv")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
