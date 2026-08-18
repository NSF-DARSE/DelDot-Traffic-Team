"""
Robustness evaluation -- 15% of the score, and previously unmeasured.

The README names the hidden slices without disclosing membership: "limited-history
locations, difficult operating periods, incomplete observations, uncertainty". This
re-scores the protocol's held-out predictions on stress subsets that stand in for those,
so a model is judged on where it breaks rather than on one blended number.

Nothing here refits a model: `validate_protocol.run_fold` hands back its held-out
predictions and every slice is a mask over those same rows.

    python robustness_slices.py --folds 4 --out robustness_results.json
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
PEAK = (7, 8, 9, 16, 17, 18)
NIGHT = (0, 1, 2, 3, 4, 5)


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def gap_flags(lab: pd.DataFrame) -> pd.Series:
    """Hours whose pair has a >=24h hole immediately before them.

    Missingness in this dataset is whole-day, never partial, so a pair's series jumps
    across outages. Recovery rows after an outage are exactly the "incomplete
    observations" case the README calls out.
    """
    d = lab[["pair", "timestamp"]].sort_values(["pair", "timestamp"])
    prev = d.groupby("pair", observed=True).timestamp.shift(1)
    hole = (d.timestamp - prev) >= pd.Timedelta("24h")
    return hole.fillna(False).reindex(lab.index).astype(bool)


def build_slices(rows: pd.DataFrame, ctx: dict) -> dict[str, np.ndarray]:
    """Boolean masks over `rows`. Each stands in for a named hidden slice."""
    y = np.expm1(rows.ly.to_numpy())
    st = rows.station_key.to_numpy()
    pair = rows.pair.to_numpy()
    hour = rows.hour.to_numpy()
    month = rows.month.to_numpy()

    hist_n = ctx["hist_rows_per_pair"]
    n_hist = pd.Series(pair).map(hist_n).fillna(0).to_numpy()
    thin = np.nanquantile(hist_n.to_numpy(), 0.25) if len(hist_n) else 0

    return {
        "ALL": np.ones(len(rows), bool),
        # difficult operating periods
        "winter (Dec-Feb)": np.isin(month, (12, 1, 2)),
        "summer (Jun-Aug)": np.isin(month, (6, 7, 8)),
        "peak hours": np.isin(hour, PEAK),
        "night hours": np.isin(hour, NIGHT),
        "weekend": rows.is_weekend.to_numpy().astype(bool),
        "holiday +/-1d": (rows.is_holiday.to_numpy() | rows.holiday_adj.to_numpy()).astype(bool),
        # incomplete observations
        "post-outage (>=24h gap)": ctx["gap"].reindex(rows.index).fillna(False).to_numpy().astype(bool),
        "limited history pair": n_hist <= thin,
        # volume regimes -- GEH is self-normalising, so both tails matter
        "low volume (<50 veh/h)": y < 50,
        "high volume (>1000 veh/h)": y > 1000,
        # graph / context stress
        "no corridor neighbour": np.isin(st, list(ctx["no_corridor"])),
        "unreliable AADT band": np.isin(st, list(ctx["aadt_bad"])),
        "functional_class missing": np.isin(st, list(ctx["fclass_missing"])),
    }


def slice_scores(rows: pd.DataFrame, pred: np.ndarray, ctx: dict, min_n=200) -> dict:
    out = {}
    y_log = rows.ly.to_numpy()
    for name, mask in build_slices(rows, ctx).items():
        if mask.sum() < min_n:
            continue
        s = V.score(y_log[mask], pred[mask])
        lo, hi = np.expm1(pred[mask, 0]), np.expm1(pred[mask, 2])
        yv = np.expm1(y_log[mask])
        s["cov90_raw_%"] = round(100 * float(((yv >= lo) & (yv <= hi)).mean()), 1)
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--gnn-hidden", type=int, default=192)
    ap.add_argument("--gnn-head", type=int, default=384)
    ap.add_argument("--gnn-layers", type=int, default=3)
    ap.add_argument("--out", default="robustness_results.json")
    a = ap.parse_args()

    log("loading")
    D = T.load_all()
    G = T.build_graph(D["edges"], D["network"])
    axis = T.station_axis(D["labeled"], D["targets"])
    statics = R.static_table(D["network"], G)
    lab, net = D["labeled"], D["network"]

    s = net.set_index("station_key")
    ctx = {
        "gap": gap_flags(lab),
        "hist_rows_per_pair": lab[lab.timestamp < V.HIST_END].groupby("pair", observed=True).size(),
        "no_corridor": {st for st in net.station_key
                        if st not in G or not any(G.edges[st, nb]["etype"] == "same_corridor"
                                                  for nb in G.neighbors(st))},
        "aadt_bad": {st for st in net.station_key if s.aadt_band.get(st) in T.AADT_BAD},
        "fclass_missing": {st for st in net.station_key if pd.isna(s.functional_class.get(st))},
    }
    log(f"context: {len(ctx['no_corridor'])} stations without a corridor neighbour, "
        f"{len(ctx['aadt_bad'])} with an unreliable AADT band, "
        f"{len(ctx['fclass_missing'])} missing functional_class")

    report, folds = V.make_splits(net, G, D["warm"], a.seed, V.N_REPORT, a.folds)
    pool = set().union(*folds)

    per_fold = []
    for i, unseen in enumerate(folds):
        seen = pool - unseen
        log(f"--- fold {i+1}/{a.folds} ---")
        preds: dict = {}
        V.run_fold(D, statics, G, axis, seen, unseen, a.seed, use_gnn=not a.no_gnn,
                   gnn_kw=dict(hidden=a.gnn_hidden, head=a.gnn_head, layers=a.gnn_layers,
                               select="geh"),
                   preds_out=preds)
        per_fold.append({m: slice_scores(rows, pred, ctx) for m, (rows, pred) in preds.items()})
        log(f"   scored {len(preds)} models on stress slices")

    models = sorted(per_fold[0])
    agg = {}
    for m in models:
        names = [n for n in per_fold[0][m]]
        agg[m] = {}
        for n in names:
            vals = [f[m][n] for f in per_fold if n in f[m]]
            agg[m][n] = {
                "GEH<5_%": round(float(np.mean([v["GEH<5_%"] for v in vals])), 1),
                "GEH<5_sd": round(float(np.std([v["GEH<5_%"] for v in vals])), 1),
                "MAE": round(float(np.mean([v["MAE"] for v in vals])), 1),
                "cov90_raw_%": round(float(np.mean([v["cov90_raw_%"] for v in vals])), 1),
                "n": int(np.mean([v["n"] for v in vals])),
            }

    for m in models:
        print("\n" + "=" * 88)
        print(f"{m}   (mean over {a.folds} folds)")
        print("-" * 88)
        print(f"{'slice':30s} {'GEH<5':>8s} {'±sd':>6s} {'MAE':>9s} {'cov90':>7s} {'n':>9s}")
        base = agg[m].get("ALL", {}).get("GEH<5_%")
        for n, v in sorted(agg[m].items(), key=lambda kv: kv[1]["GEH<5_%"]):
            d = "" if base is None or n == "ALL" else f"  ({v['GEH<5_%']-base:+.1f})"
            print(f"{n:30s} {v['GEH<5_%']:7.1f}% {v['GEH<5_sd']:6.1f} {v['MAE']:9.1f} "
                  f"{v['cov90_raw_%']:6.1f}% {v['n']:9,d}{d}")

    Path(a.out).write_text(json.dumps({"config": vars(a), "agg": agg, "folds": per_fold}, indent=2))
    log(f"wrote {a.out}")


def _selftest():
    """The two pieces with real logic: outage detection and slice masking."""
    ts = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00",   # contiguous
                         "2024-01-03 01:00",                        # 48h hole -> flagged
                         "2024-01-03 02:00"])
    df = pd.DataFrame({"pair": ["A|1"] * 4, "timestamp": ts})
    g = gap_flags(df).to_numpy()
    assert list(g) == [False, False, True, False], g
    rows = pd.DataFrame({
        "pair": ["A|1"] * 4, "station_key": ["A"] * 4, "ly": np.log1p([10, 2000, 10, 10]),
        "hour": [8, 3, 8, 3], "month": [1, 7, 1, 7], "is_weekend": [0, 0, 1, 1],
        "is_holiday": [0, 0, 0, 0], "holiday_adj": [0, 0, 0, 0],
    })
    ctx = {"gap": pd.Series(False, index=rows.index),
           "hist_rows_per_pair": pd.Series({"A|1": 100}),
           "no_corridor": set(), "aadt_bad": set(), "fclass_missing": set()}
    sl = build_slices(rows, ctx)
    assert sl["peak hours"].tolist() == [True, False, True, False]
    assert sl["low volume (<50 veh/h)"].tolist() == [True, False, True, True]
    assert sl["high volume (>1000 veh/h)"].tolist() == [False, True, False, False]
    assert sl["winter (Dec-Feb)"].tolist() == [True, False, True, False]
    print("selftest ok")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
