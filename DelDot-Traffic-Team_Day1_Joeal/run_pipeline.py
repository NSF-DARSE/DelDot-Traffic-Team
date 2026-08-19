"""
Day-1 pipeline runner (full: LightGBM + GNN cold-start + combine + metrics).
Step 1  LightGBM ratio model    -> target rows whose series has history ("seen")
Step 2  GNN level interpolation -> target rows with no history (graph conv; level x peer-shape)
Step 3  Combine by routing      -> one submission (history->LightGBM, no history->GNN)
Metrics MAE, RMSE, R2, WAPE, MAPE, GEH(<5/<10), 90% coverage, Winkler, + cold-start holdout sim.

Every number written to outputs/ is computed here; nothing is typed in. Row counts
and scores therefore depend on the input files and are reported at run time.
Run: python run_pipeline.py    (expects the participant files in ./day1)
"""
from __future__ import annotations
import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))
import pipeline as P, metrics as M, gnn as G, scenarios as SC

# Paths resolve against this file, not the shell's cwd, so the run works from anywhere.
# DATA_DIR / OUT_DIR override them (OUT_DIR keeps test runs away from the real outputs).
_data_env = os.environ.get("DATA_DIR")
DATA = Path(_data_env) if _data_env else (HERE / "day1")
OUT = Path(os.environ.get("OUT_DIR", HERE / "outputs"))
(OUT / "figures").mkdir(parents=True, exist_ok=True)   # viz/*.py write in here
FLOOR = 20.0

def mape(actual, pred, floor=FLOOR):
    a = np.asarray(actual, float); p = np.asarray(pred, float); m = a >= floor
    return float(np.mean(np.abs(a[m] - p[m]) / a[m]) * 100) if m.any() else float("nan")

def build_cold_forecast(levels_by_station, shapes, net, rows):
    """level (GNN, blended with the peer prior) x hourly peer-group shape.

    Peer group is functional_class where present and the aadt_band rung where it is
    blank, so the 28% of stations with no functional_class do not all collapse onto
    one global curve.
    """
    peer = G.peer_groups(net)
    r = P.add_calendar(rows)
    glob = shapes["__global__"]
    vals = [v for v in levels_by_station.values() if np.isfinite(v)]
    fallback = float(np.mean(vals)) if vals else 1.0
    how = pd.to_numeric(r["how"], errors="coerce").fillna(0).to_numpy().astype(int) % 168
    stn = r["station_key"].astype(str).to_numpy()
    lvl = np.array([levels_by_station.get(s, np.nan) for s in stn], float)
    lvl = np.where(np.isfinite(lvl), lvl, fallback)
    shp = np.array([shapes.get(peer.get(s), glob)[h] for s, h in zip(stn, how)], float)
    return np.maximum(lvl * np.nan_to_num(shp, nan=1.0), 0.0)


def _wape(actual, pred):
    a = np.asarray(actual, float); d = float(np.sum(np.abs(a)))
    return float(np.sum(np.abs(a - np.asarray(pred, float))) / d) if d else float("nan")

def main():
    print("Loading (authentic day-one, offset left as-is)...")
    train, val, net, edges, template = P.load_all(DATA, apply_offset=False)
    net["station_key"] = net["station_key"].astype(str)
    stat = P.static_table(net); a = val[P.TARGET].to_numpy()

    print("Step 1 - LightGBM backtest (fit 2024 -> score H1 2025)...")
    prof, glob = P.fit_climatology(train); naive = P.apply_climatology(val, prof, glob)
    lgbm = P.train_model(train, prof, glob, stat); pred = P.predict(lgbm, val, prof, glob, stat)
    qmap = P.fit_intervals(a, pred); lo, hi = P.apply_intervals(pred, qmap)
    oprof, oglob = P.fit_climatology(val); oracle = P.apply_climatology(val, oprof, oglob)
    met = M.all_metrics(a, pred, lo, hi, naive=naive, oracle=oracle)
    met["MAPE"] = mape(a, pred); met["naive_MAE"] = float(np.mean(np.abs(a - naive)))
    met["n_seen_backtest_rows"] = int(len(a))

    print("Step 2 - GNN cold-start (graph conv over road network)...")
    print(f"  gradient check: {G._grad_check():.1e}")
    hist = pd.concat([train, val], ignore_index=True)
    stations, idx, A = G.build_graph(net, edges); Xb = G.node_features(net, stations, idx)
    lv = G.station_levels(hist); y = np.full(len(stations), np.nan)
    for st, i in idx.items():
        if st in lv.index: y[i] = np.log1p(lv[st])
    known = ~np.isnan(y)
    if not known.any():
        raise RuntimeError("no station in network.csv matched the traffic history; "
                           "check station_key formatting in the input files")
    rng = np.random.default_rng(7); kn = np.where(known)[0]
    n_hold = max(1, int(0.2 * len(kn))) if len(kn) > 1 else 0
    hold = rng.choice(kn, size=n_hold, replace=False)
    in_train = known.copy(); in_train[hold] = False
    held = [stations[i] for i in hold]

    # The holdout stations must be invisible to EVERY part of the cold-start
    # estimate, not just the GCN loss mask. The peer-group shapes and the level
    # prior are fitted on in-train stations only; using `hist` here would let a
    # held-out station help predict itself through its own peer median.
    hist_sim = hist[~hist["station_key"].isin(held)]
    shapes_sim = G.class_shapes(hist_sim, net)
    clv_sim = G.class_levels(hist_sim, net)

    Xa = G.augment_with_levels(Xb, np.nan_to_num(y), in_train)
    yfill = np.nan_to_num(y, nan=float(np.nanmean(y[in_train])) if in_train.any() else 0.0)
    g_sim = G.train_gcn(A, Xa, yfill, in_train, hid=16, epochs=500, lr=0.02, seed=0)
    sim_levels = np.expm1(g_sim.forward(A, Xa))
    lvl_map = G.blend_levels({stations[i]: max(sim_levels[i], 1.0) for i in hold},
                             clv_sim, net, weight=0.5)
    sr = val[val["station_key"].isin(held)].copy()
    sim_pred = build_cold_forecast(lvl_map, shapes_sim, net, sr)
    sim_act = sr[P.TARGET].to_numpy()
    met["cold_sim_stations"] = int(len(hold)); met["cold_sim_rows"] = int(len(sr))
    met["cold_sim_MAPE"] = mape(sim_act, sim_pred)
    met["cold_sim_WAPE"] = _wape(sim_act, sim_pred)
    met.update({f"cold_{k}": v for k, v in M.geh_metrics(sim_act, sim_pred).items()})
    gm = np.expm1(float(np.mean(y[in_train]))) if in_train.any() else 1.0
    naive_cold = build_cold_forecast({s: gm for s in held},
                                     {"__global__": shapes_sim["__global__"]}, net, sr)
    met["cold_naive_WAPE"] = _wape(sim_act, naive_cold)

    # Cold intervals are calibrated on this holdout: the empirical 5th/95th
    # percentiles of actual/predicted. 1.645 x WAPE would treat a mean absolute
    # error as a standard deviation, which it is not.
    ratio = sim_act / np.maximum(sim_pred, 1e-9)
    ratio = ratio[np.isfinite(ratio) & (sim_pred > 0)]
    if ratio.size >= 50:
        cold_lo_mult = float(np.clip(np.quantile(ratio, P.LOWER_Q), 0.0, 1.0))
        cold_hi_mult = float(max(np.quantile(ratio, P.UPPER_Q), 1.0))
    else:
        cold_lo_mult, cold_hi_mult = 0.25, 2.5
    met["cold_lo_mult"] = cold_lo_mult; met["cold_hi_mult"] = cold_hi_mult

    print("Final - combine LightGBM(seen) + GNN(cold)...")
    # the delivered file may use every station with history, so the priors here are
    # fitted on all of `hist` (nothing is being scored, so this is not leakage)
    shapes_all = G.class_shapes(hist, net); clv_all = G.class_levels(hist, net)
    prof_f, glob_f = P.fit_climatology(hist); lgbm_f = P.train_model(hist, prof_f, glob_f, stat)
    Xa_all = G.augment_with_levels(Xb, np.nan_to_num(y), known)
    g_full = G.train_gcn(A, Xa_all, yfill, known, hid=16, epochs=500, lr=0.02, seed=0)
    full_levels = np.expm1(g_full.forward(A, Xa_all))
    cold_level_map = G.blend_levels(
        {stations[i]: max(full_levels[i], 1.0) for i in range(len(stations)) if not known[i]},
        clv_all, net, weight=0.5)
    tmpl = P.add_calendar(template)          # normalises station_key/direction_code too
    seen_series = set(map(tuple, hist[P.SERIES].drop_duplicates().to_numpy()))
    key = list(zip(tmpl["station_key"], tmpl["direction_code"]))
    is_seen = np.array([k in seen_series for k in key])
    print(f"  seen rows: {is_seen.sum():,} (LightGBM) | cold rows: {(~is_seen).sum():,} (GNN)")
    if not is_seen.any():
        raise RuntimeError("no template row matched a series with history; check that "
                           "station_key/direction_code formats agree across the files")
    fv = np.zeros(len(tmpl)); lo_f = np.zeros(len(tmpl)); hi_f = np.zeros(len(tmpl))
    srw = tmpl[is_seen].copy(); sp = P.predict(lgbm_f, srw, prof_f, glob_f, stat)
    slo, shi = P.apply_intervals(sp, qmap); fv[is_seen] = sp; lo_f[is_seen] = slo; hi_f[is_seen] = shi
    if (~is_seen).any():
        crw = tmpl[~is_seen].copy(); cp = build_cold_forecast(cold_level_map, shapes_all, net, crw)
        fv[~is_seen] = cp
        lo_f[~is_seen] = np.maximum(cp * cold_lo_mult, 0.0)
        hi_f[~is_seen] = cp * cold_hi_mult

    # Reliability is DERIVED from what this run actually measured, not asserted:
    # accuracy-based skill (1 - WAPE) per path, then a night-hours haircut because
    # low-volume hours are proportionally the hardest.
    rel_seen = float(np.clip(1.0 - met["WAPE"], 0.10, 0.95))
    rel_cold = float(np.clip(1.0 - met["cold_sim_WAPE"], 0.10, 0.95))
    met["reliability_seen"] = rel_seen; met["reliability_cold"] = rel_cold
    rel = np.where(is_seen, rel_seen, rel_cold)
    hr = tmpl["hour"].fillna(12).to_numpy()
    night = (hr <= 4) | (hr >= 22)
    rel = np.clip(rel * np.where(night, 0.90, 1.0), 0.10, 0.95)
    sub = template.copy()
    sub["forecast_volume"] = np.round(fv, 2); sub["lower_90"] = np.round(np.minimum(lo_f, fv).clip(0), 2)
    sub["upper_90"] = np.round(np.maximum(hi_f, fv).clip(0), 2); sub["reliability_score"] = np.round(rel, 4)

    # Self-check the invariants the official validator looks for, so a broken file
    # is caught here rather than at submission time.
    _cols = ["forecast_volume", "lower_90", "upper_90", "reliability_score"]
    assert len(sub) == len(template), f"row count changed: {len(sub)} vs {len(template)}"
    assert sub[_cols].notna().all().all(), "submission contains NaN"
    assert np.isfinite(sub[_cols].to_numpy(float)).all(), "submission contains inf"
    assert (sub["forecast_volume"] >= 0).all(), "negative forecast_volume"
    assert (sub["lower_90"] <= sub["forecast_volume"]).all(), "lower_90 above forecast"
    assert (sub["upper_90"] >= sub["forecast_volume"]).all(), "upper_90 below forecast"
    assert sub["reliability_score"].between(0, 1).all(), "reliability_score out of [0,1]"
    met["submission_rows"] = int(len(sub))
    met["submission_selfcheck"] = "PASS"
    sub.to_csv(OUT / "submission.csv", index=False); print(f"  submission.csv written ({len(sub):,} rows, self-check PASS)")
    scn = SC.build_scenarios(sub, pd.read_csv(DATA / "scenario_template.csv").columns,
                             OUT / "scenario_results.csv",
                             eligible=set(tmpl.loc[is_seen, "station_key"]))
    print(f"  scenario_results.csv (computed, {len(scn)} scenarios)")
    with open(OUT / "metrics.json", "w") as f: json.dump(met, f, indent=2)
    for k, v in met.items(): print(f"    {k:22s}: {v:.4f}" if isinstance(v, float) else f"    {k:22s}: {v}")
    vv = P.add_calendar(val); vv["pred"] = pred
    pwd = vv[~vv["is_weekend"]].groupby("hour")[P.TARGET].mean().reindex(range(24)).to_numpy()
    pwe = vv[vv["is_weekend"]].groupby("hour")[P.TARGET].mean().reindex(range(24)).to_numpy()
    busy = vv.groupby(P.SERIES)[P.TARGET].mean().idxmax()
    one = vv[(vv["station_key"] == busy[0]) & (vv["direction_code"] == busy[1])].sort_values("timestamp").head(168)
    np.savez(OUT / "artifacts.npz", actual=a, pred=pred, lo=lo, hi=hi, naive=naive, oracle=oracle,
             prof_wd=np.nan_to_num(pwd), prof_we=np.nan_to_num(pwe),
             one_actual=one[P.TARGET].to_numpy(), one_pred=one["pred"].to_numpy(), seen_widths=(shi - slo),
             cold_sim_actual=sim_act, cold_sim_pred=sim_pred, cold_naive=naive_cold, metrics=json.dumps(met))
    print("  artifacts.npz written\nDone.")

if __name__ == "__main__":
    main()
