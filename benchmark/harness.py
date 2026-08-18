"""
Comparison harness for the three-slice traffic problem.

Every model scores on the same frozen station folds, the same fit windows and the same
metrics, so two models can be compared by reading two numbers. Nothing here is specific to
any particular model -- a model is a small object with `fit` and `predict`.

    ~/.virtualenvs/data_science/bin/python benchmark/harness.py --model benchmark/models/backbone.py
    ~/.virtualenvs/data_science/bin/python benchmark/harness.py --model benchmark/models/team_v1.py --arm deploy
    ~/.virtualenvs/data_science/bin/python benchmark/harness.py --leaderboard

The two arms answer different questions and their numbers are NOT comparable:

  protocol  4-fold nested *station* CV. Fit on 2024 for the 45 seen stations, then score
            three experiments -- warm-future (seen stations, 2025 H1), cold-interp (unseen
            stations, 2024) and cold-future (unseen stations, 2025 H1). The only arm that
            can measure cold start, because it hides whole stations. Systematically
            pessimistic for warm stations, because it throws away the 2025 H1 half-year
            that deployment actually holds.

  deploy    Fit on 2024 + Jan-Apr 2025, score May-Jun 2025, all labelled stations. Mirrors
            the real task for warm rows. Warm only -- it cannot say anything about cold
            start, because every station is seen.

The same backbone scores 71.7% GEH<5 on the protocol and 82.7% on the deployment arm.
Quoting one against the other is the single easiest way to produce a meaningless comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

import traffic_pipeline as T
import validate_protocol as VP
import run_pipeline as R

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LEADERBOARD = HERE / "LEADERBOARD.md"

EXPERIMENTS = ("warm-future", "cold-interp", "cold-future")
DEPLOY_FIT_END = pd.Timestamp("2025-04-30 23:00:00")
DEPLOY_SCORE_START = pd.Timestamp("2025-05-01 00:00:00")


# --------------------------------------------------------------------------- #
# the model contract
# --------------------------------------------------------------------------- #

@dataclass
class FitContext:
    """Everything a model is allowed to see for one fit.

    `labeled` has already had the held-out stations and the out-of-window rows removed.
    Reading anything else -- the parquet files, the submission template, a global cache --
    reintroduces the leakage the folds exist to prevent.
    """

    labeled: pd.DataFrame
    network: pd.DataFrame
    edges: pd.DataFrame
    statics: pd.DataFrame
    graph: object
    axis: dict
    seen_stations: set
    experiment: str
    fold: int
    seed: int = 0
    extra: dict = field(default_factory=dict)


class BenchModel:
    """What a model must implement.

    name       short identifier, used as the results filename and the leaderboard row.
    fit(ctx)   fit on ctx.labeled only.
    predict(rows) -> (n, 3) array of VOLUMES: lower_90, forecast_volume, upper_90.
                  `rows` carries station_key, direction_code, pair, timestamp and the
                  calendar columns from `traffic_pipeline.add_calendar`.
    """

    name = "unnamed"

    def fit(self, ctx: FitContext) -> "BenchModel":
        raise NotImplementedError

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# metrics -- all in volume space, which is what the challenge scores
# --------------------------------------------------------------------------- #

def geh(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    denom = np.clip(pred + obs, 1e-9, None)
    return np.abs(pred - obs) * np.sqrt(2.0 / denom)


def score(y: np.ndarray, p3: np.ndarray) -> dict:
    """y: observed volumes. p3: (n, 3) lower / point / upper, in volumes."""
    lo, p, hi = p3[:, 0], p3[:, 1], p3[:, 2]
    g = geh(p, y)
    err = p - y
    denom = np.clip((np.abs(p) + np.abs(y)) / 2.0, 1e-9, None)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    # Winkler interval score at 90%
    alpha = 0.10
    wink = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return {
        # WAPE is the organisers' primary automated forecast metric (confirmed by DelDOT,
        # 2026-08-18). sum|y - yhat| / sum|y|: the denominator does not depend on the
        # prediction, so on a fixed row set WAPE is a rescaling of MAE -- but the rescaling
        # differs per slice, so only WAPE is comparable across slices.
        "WAPE_%": round(float(np.abs(err).sum() / max(np.abs(y).sum(), 1e-9) * 100), 2),
        "GEH<5_%": round(float(np.mean(g < 5) * 100), 2),
        "GEH<10_%": round(float(np.mean(g < 10) * 100), 2),
        "within20_%": round(float(np.mean(np.abs(err) <= 0.20 * np.clip(y, 1e-9, None)) * 100), 2),
        "MAE": round(float(np.mean(np.abs(err))), 2),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "sMAPE": round(float(np.mean(np.abs(err) / denom) * 100), 2),
        "R2": round(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), 4),
        "cov90": round(float(np.mean(covered)), 4),
        "width": round(float(np.mean(width)), 1),
        "winkler": round(float(np.mean(wink)), 1),
        "n": int(len(y)),
    }


def aggregate(per_fold: list[dict]) -> dict:
    """Mean over folds, plus the spread that makes a cold-slice number readable.

    WAPE is pooled as a ratio of sums rather than a mean of per-fold ratios, because a
    mean of ratios silently reweights the folds by how little volume each one carries.
    """
    if not per_fold:
        return {}
    keys = [k for k in per_fold[0] if k != "n"]
    out = {k: round(float(np.mean([f[k] for f in per_fold])), 2) for k in keys}
    num = sum(f["MAE"] * f["n"] for f in per_fold)
    den = sum(f["MAE"] * f["n"] / (f["WAPE_%"] / 100) for f in per_fold if f["WAPE_%"] > 0)
    if den > 0:
        out["WAPE_%"] = round(100 * num / den, 2)
    out["WAPE_sd"] = round(float(np.std([f["WAPE_%"] for f in per_fold])), 2)
    out["GEH<5_sd"] = round(float(np.std([f["GEH<5_%"] for f in per_fold])), 2)
    out["MAE_sd"] = round(float(np.std([f["MAE"] for f in per_fold])), 2)
    out["n"] = int(np.sum([f["n"] for f in per_fold]))
    out["folds"] = len(per_fold)
    return out


# --------------------------------------------------------------------------- #
# the arms
# --------------------------------------------------------------------------- #

def _windows(lab: pd.DataFrame):
    hist = lab[lab.timestamp.dt.year == 2024]
    fut = lab[lab.timestamp.dt.year == 2025]
    return hist, fut


def run_protocol(build, D, aux, seed: int = 0, max_folds: int | None = None) -> dict:
    """4-fold nested station CV: fit on 2024, score the three experiments.

    The folds themselves always come from `splits_frozen.json`. `max_folds` runs only the
    first few of them, for a smoke test -- it never changes how a fold is constructed, so a
    partial run stays comparable to the corresponding folds of a full one.
    """
    lab, net, edges = D["labeled"], D["network"], D["edges"]
    G, axis, statics = aux["G"], aux["axis"], aux["statics"]
    report, fold_sets = VP.make_splits(net, G, D["warm"], seed, VP.N_REPORT, VP.N_FOLDS)
    if max_folds:
        fold_sets = fold_sets[:max_folds]
    hist, fut = _windows(lab)

    runs = []
    for i, unseen in enumerate(fold_sets):
        seen = D["warm"] - unseen - report
        fit_rows = hist[hist.station_key.isin(seen)]
        cases = {
            "warm-future": (fut[fut.station_key.isin(seen)]),
            "cold-interp": (hist[hist.station_key.isin(unseen)]),
            "cold-future": (fut[fut.station_key.isin(unseen)]),
        }
        row = {}
        for exp, test in cases.items():
            if not len(test):
                continue
            t0 = time.time()
            m = build()
            m.fit(FitContext(labeled=fit_rows, network=net, edges=edges, statics=statics,
                             graph=G, axis=axis, seen_stations=set(seen), experiment=exp,
                             fold=i, seed=seed))
            p3 = np.asarray(m.predict(test), dtype=np.float64)
            _check(p3, test)
            row[exp] = score(test.volume.to_numpy(np.float64), p3)
            row[exp]["fit_predict_s"] = round(time.time() - t0, 1)
            print(f"   fold {i+1} {exp:12s} GEH<5 {row[exp]['GEH<5_%']:5.1f}%  "
                  f"MAE {row[exp]['MAE']:7.1f}  cov90 {row[exp]['cov90']:.3f}")
        runs.append(row)

    summary = {exp: aggregate([r[exp] for r in runs if exp in r]) for exp in EXPERIMENTS}
    return {"folds": runs, "summary": {k: v for k, v in summary.items() if v}}


def run_deploy(build, D, aux, seed: int = 0) -> dict:
    """Fit 2024 + Jan-Apr 2025, score May-Jun 2025. Warm rows only, no station is hidden."""
    lab, net, edges = D["labeled"], D["network"], D["edges"]
    G, axis, statics = aux["G"], aux["axis"], aux["statics"]
    fit_rows = lab[lab.timestamp <= DEPLOY_FIT_END]
    test = lab[lab.timestamp >= DEPLOY_SCORE_START]

    t0 = time.time()
    m = build()
    m.fit(FitContext(labeled=fit_rows, network=net, edges=edges, statics=statics, graph=G,
                     axis=axis, seen_stations=set(D["warm"]), experiment="deploy",
                     fold=0, seed=seed))
    p3 = np.asarray(m.predict(test), dtype=np.float64)
    _check(p3, test)
    s = score(test.volume.to_numpy(np.float64), p3)
    s["fit_predict_s"] = round(time.time() - t0, 1)
    print(f"   deploy       GEH<5 {s['GEH<5_%']:5.1f}%  MAE {s['MAE']:7.1f}  cov90 {s['cov90']:.3f}")
    return {"folds": [{"deploy": s}], "summary": {"deploy": s}}


def _check(p3: np.ndarray, rows: pd.DataFrame) -> None:
    if p3.shape != (len(rows), 3):
        raise ValueError(f"predict must return ({len(rows)}, 3), got {p3.shape}")
    if not np.isfinite(p3).all():
        raise ValueError("predict returned non-finite values")
    if (p3 < 0).any():
        raise ValueError("predict returned negative volumes")
    if not ((p3[:, 0] <= p3[:, 1] + 1e-9) & (p3[:, 1] <= p3[:, 2] + 1e-9)).all():
        raise ValueError("predict must return lower <= point <= upper")


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #

def load_model(path: str):
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build"):
        raise SystemExit(f"{p} must define build() -> BenchModel")
    return mod.build


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def write_leaderboard() -> None:
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text())
        rows.append(d)
    lines = [
        "# Leaderboard",
        "",
        "Generated by `benchmark/harness.py --leaderboard`. Every row is produced by the",
        "same folds (`splits_frozen.json`), the same fit windows and the same metrics.",
        "",
        "**The two arms are not comparable to each other.** The protocol hides whole",
        "stations and fits on 2024 only; the deployment arm sees every station and fits on",
        "2024 + Jan-Apr 2025. The same backbone scores 71.7% on one and 82.7% on the other.",
        "",
        "### How to read this",
        "",
        "**WAPE is the primary metric.** Confirmed by the organisers on 2026-08-18: the",
        "primary automated forecast metric is WAPE, `sum|y - yhat| / sum|y|`, and scoring",
        "happens across organiser-defined conditions rather than one undifferentiated pool.",
        "Rows are sorted by warm-future WAPE. Lower is better.",
        "",
        "GEH<5 is kept because it is the traffic-engineering agreement statistic and it",
        "reads differently -- **the two disagree, sharply**. On the protocol arm",
        "`team_v1_reimpl` has the best warm-future WAPE and the worst warm-future GEH<5.",
        "GEH<5 is a hit rate that low-volume rows pass easily; WAPE weights every error by",
        "its absolute size. A model can win one and lose the other, and here one does.",
        "",
        "Diagnosis so far, on our backbone, warm-future fold 0: half the absolute error",
        "comes from the bottom five volume deciles, which carry 12.4% of the volume -- we",
        "over-predict low-volume hours. Two fixes were tried and rejected: a global",
        "multiplicative recentring (oracle multiplier is worth 0.08 WAPE points) and",
        "re-anchoring the pair level from `median(log1p(y))` to `log1p(mean(y))` (worse,",
        "21.40% -> 23.51%). The open candidate is an L1-in-volume-space objective.",
        "",
        "Interval quality is `cov90` and `winkler`; the organisers evaluate",
        "`lower_90`/`upper_90` calibration separately from `reliability_score`, which is a",
        "*relative* 0-1 confidence indicator and not the probability of any specific event.",
        "",
    ]
    for arm, title in (("protocol", "Protocol — 4-fold nested station CV"),
                       ("deploy", "Deployment arm — fit 2024 + Jan–Apr 2025, score May–Jun 2025")):
        sel = [d for d in rows if d["arm"] == arm]
        if not sel:
            continue
        lines += [f"## {title}", ""]
        if arm == "protocol":
            lines += ["| model | warm-future WAPE | cold-interp WAPE | cold-future WAPE | warm GEH<5 | ±sd | cold-interp GEH<5 | ±sd | cold-future GEH<5 | ±sd |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for d in sorted(sel, key=lambda x: x["summary"].get("warm-future", {}).get("WAPE_%", 1e9)):
                s = d["summary"]
                def g(e, k, dash="—"):
                    return s.get(e, {}).get(k, dash)
                lines.append(
                    f"| `{d['model']}` | **{g('warm-future','WAPE_%')}** | **{g('cold-interp','WAPE_%')}** "
                    f"| **{g('cold-future','WAPE_%')}** "
                    f"| {g('warm-future','GEH<5_%')} | {g('warm-future','GEH<5_sd')} "
                    f"| {g('cold-interp','GEH<5_%')} | {g('cold-interp','GEH<5_sd')} "
                    f"| {g('cold-future','GEH<5_%')} | {g('cold-future','GEH<5_sd')} |")
        else:
            lines += ["| model | WAPE | GEH<5 | MAE | sMAPE | cov90 | width | Winkler |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for d in sorted(sel, key=lambda x: x["summary"]["deploy"]["WAPE_%"]):
                s = d["summary"]["deploy"]
                lines.append(f"| `{d['model']}` | **{s['WAPE_%']}** | {s['GEH<5_%']} | {s['MAE']} "
                             f"| {s['sMAPE']} | {s['cov90']} | {s['width']} | {s['winkler']} |")
        lines.append("")
    lines += [
        "## Reference ceilings",
        "",
        "Oracle substitution on warm-future, protocol arm — the model is handed one perfect",
        "component. Not achievable, but it bounds what further work on that component can",
        "be worth. Produced by `warm_decomp.py` on the corrected data file.",
        "",
        "| perfect component | WAPE | GEH<5 |",
        "|---|---:|---:|",
        "| *nothing* (current backbone) | 13.11% | 79.5% |",
        "| + month | 12.79% | 80.3% |",
        "| + shape | 12.40% | 81.0% |",
        "| + **level** | **12.28%** | 81.4% |",
        "| + level + shape + month | 11.03% | 84.6% |",
        "| full oracle pair x hour x dow x month | **8.48%** | 88.6% |",
        "| + true network shock | 7.43% | 91.9% |",
        "",
        "**The lever depends on which metric you ask.** Under WAPE, level is worth slightly",
        "more than shape on warm-future (-0.83 against -0.71): WAPE weights an error by its",
        "absolute size, and level error dominates at high volume. Under GEH<5 the same",
        "decomposition says the opposite (+9.3 shape against +1.7 level), because GEH is a",
        "hit rate that low-volume rows pass easily.",
        "",
        "Cold-slice ceilings were measured in GEH on the pre-correction data (perfect level",
        "+18.5 points on cold-interp against +2.0 for shape) and have **not** been",
        "re-measured in WAPE. Cold-future has no measured ceiling in either metric.",
        "",
    ]
    LEADERBOARD.write_text("\n".join(lines))
    print(f"wrote {LEADERBOARD}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", help="path to a model file defining build()")
    ap.add_argument("--arm", default="protocol", choices=["protocol", "deploy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=None,
                    help="run only the first N of the 4 frozen folds (smoke test)")
    ap.add_argument("--tag", default=None, help="results filename override")
    ap.add_argument("--leaderboard", action="store_true", help="regenerate LEADERBOARD.md and exit")
    a = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if a.leaderboard and not a.model:
        write_leaderboard()
        return
    if not a.model:
        ap.error("--model is required (or use --leaderboard)")

    build = load_model(a.model)
    print(f"loading data")
    D = T.load_all()
    G = T.build_graph(D["edges"], D["network"])
    aux = {"G": G, "axis": T.station_axis(D["labeled"], D["targets"]),
           "statics": R.static_table(D["network"], G)}

    name = getattr(build(), "name", Path(a.model).stem)
    print(f"running {name} on the {a.arm} arm")
    res = run_protocol(build, D, aux, a.seed, a.max_folds) if a.arm == "protocol" \
        else run_deploy(build, D, aux, a.seed)

    out = {
        "model": name,
        "arm": a.arm,
        "seed": a.seed,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "code_sha": _git_sha(),
        "splits_sha": hashlib.sha256((HERE / "splits_frozen.json").read_bytes()).hexdigest()[:12],
        "source": str(Path(a.model).as_posix()),
        **res,
    }
    if a.max_folds:
        out["partial_folds"] = a.max_folds
    fp = RESULTS / f"{a.tag or name}__{a.arm}.json"
    fp.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {fp}")
    for exp, s in out["summary"].items():
        print(f"  {exp:12s} GEH<5 {s['GEH<5_%']:5.1f}%  MAE {s['MAE']:7.1f}  "
              f"cov90 {s.get('cov90', float('nan')):.3f}")
    write_leaderboard()


if __name__ == "__main__":
    main()
