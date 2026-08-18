"""
A validation arm that mimics DEPLOYMENT, not just the protocol.

The protocol's warm-future arm fits on 2024 and scores on 2025 H1. Deployment is different:
we hold 2024 AND 2025 H1, and forecast 2025 H2. That matters for the profile estimate,
because the shape drifts year over year (0.157 log across years vs a 0.097 within-year
noise floor) -- so the most recent half-year is the best evidence about the *current*
shape, while the matched months a year earlier are the best evidence about the *seasonal*
shape. The protocol arm cannot see this trade-off at all, because it has only one year.

Closest available imitation, following brief section 9.2:

    fit on 2024 + Jan-Apr 2025   ->   score May-Jun 2025

Both signals are present (recent partial year, and matched months a year back) and the
scored window is genuinely out of sample. Compares ways of combining them.

    python deploy_arm.py
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import traffic_pipeline as T
import validate_protocol as V

T0 = time.time()
HOURS, DOWS = 24, 7
CELLS = pd.MultiIndex.from_product([range(HOURS), range(DOWS)])
CUT = pd.Timestamp("2025-05-01")


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def cyc(a, b):
    return np.minimum(np.abs(a - b), 12 - np.abs(a - b))


def prof_w(d, w):
    med = d.groupby("pair").ly.transform("median")
    dev = (d.ly - med).to_numpy()
    k = pd.MultiIndex.from_arrays([d.pair, d.hour, d.dow])
    num = pd.Series(dev * w, index=k).groupby(level=[0, 1, 2]).sum()
    den = pd.Series(w, index=k).groupby(level=[0, 1, 2]).sum()
    return (num / den).unstack(level=[1, 2]).reindex(columns=CELLS)


def truth_by_month(d):
    med = d.groupby("pair").ly.transform("median")
    return (d.ly - med).groupby([d.pair, d.month, d.hour, d.dow]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deploy_arm.json")
    a = ap.parse_args()

    D = T.load_all(); lab = D["labeled"]
    lab = lab[lab.station_key.isin(D["warm"])]
    hist = lab[lab.timestamp < CUT]
    test = lab[lab.timestamp >= CUT]
    Tm = truth_by_month(test)
    log(f"fit on {len(hist):,} rows (2024 + Jan-Apr 2025), score {len(test):,} rows (May-Jun 2025)")

    hm = hist.month.to_numpy()
    age = (CUT - hist.timestamp).dt.total_seconds().to_numpy() / 86400.0
    is25 = (hist.year.to_numpy() == 2025).astype(float)

    def build(name):
        if name == "pooled (deployed)":
            return np.ones(len(hist))
        if name == "month kernel w=2":
            return None
        if name == "2025 only":
            return is25
        if name == "2024 only":
            return 1.0 - is25
        if name == "recency 180d":
            return 0.5 ** (age / 180.0)
        if name == "recency 365d":
            return 0.5 ** (age / 365.0)
        raise KeyError(name)

    names = ["pooled (deployed)", "2024 only", "2025 only", "recency 365d", "recency 180d",
             "month kernel w=2", "month kernel w=2 x recency 365d",
             "month kernel w=2 x recency 180d", "month kernel w=2, 2x weight on 2025"]
    res = {}
    for name in names:
        errs = []
        for m in sorted(test.month.unique()):
            if "month kernel" in name:
                w = np.exp(-0.5 * (cyc(hm, m) / 2.0) ** 2)
                if "recency 365" in name:
                    w = w * 0.5 ** (age / 365.0)
                elif "recency 180" in name:
                    w = w * 0.5 ** (age / 180.0)
                elif "2x weight on 2025" in name:
                    w = w * (1.0 + is25)
            else:
                w = build(name)
            P = prof_w(hist, w)
            t = Tm.xs(m, level=1).unstack(level=[1, 2]).reindex(columns=CELLS)
            i = P.index.intersection(t.index)
            aa, bb = P.loc[i], t.loc[i]
            msk = (aa.notna() & bb.notna()).to_numpy()
            errs.append(float(np.abs(aa.to_numpy()[msk] - bb.to_numpy()[msk]).mean()))
        res[name] = float(np.mean(errs))
        log(f"{name:36s} {res[name]:.4f}")

    print("\n" + "=" * 66)
    print("mean |profile error| on May-Jun 2025, fit on 2024 + Jan-Apr 2025")
    print("-" * 66)
    base = res["pooled (deployed)"]
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        print(f"{k:38s} {v:.4f}  ({100*(base-v)/base:+5.1f}%)")
    print("-" * 66)
    Path(a.out).write_text(json.dumps(res, indent=2))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
