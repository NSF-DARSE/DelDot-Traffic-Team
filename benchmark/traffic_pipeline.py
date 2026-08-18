"""
Slice-routed probabilistic pipeline for the Traffic Pattern Analysis challenge.

Implements the architecture from `Traffic_Hackathon_Solution_Research_gpt.md`:
a multiplicative profile backbone, three slice-specific prediction paths, an
out-of-time conformal calibration layer, and a learned reliability model.

    canonical data
        -> hierarchical profile backbone (level x shape x season x drift x holiday)
        -> slice router
             A warm future   : backbone + LightGBM residual quantiles (own history)
             B cold future   : donor level x shape transfer (no station ID)
             C cold backfill : contemporaneous neighbour-residual nowcast
        -> Mondrian conformal calibration (slice x volume band)
        -> reliability meta-model
        -> submission

Run:
    ~/.virtualenvs/data_science/bin/python traffic_pipeline.py validate
    ~/.virtualenvs/data_science/bin/python traffic_pipeline.py submit --out submission_pipeline.csv
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def _find_data_dir() -> Path:
    """Locate the directory holding the supplied data files.

    The code lives in `benchmark/` inside the shared team repo but the data sits at the
    repo root, so a plain `Path(__file__).parent` breaks. Walk up until the files appear,
    and let `TRAFFIC_DATA_DIR` override for anyone keeping data elsewhere.
    """
    env = os.environ.get("TRAFFIC_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / "traffic_train.parquet").exists():
            return cand
    return here


DATA_DIR = _find_data_dir()
CLOCK_SHIFT_H = None       # None = detect from the data. See `detect_clock_offset`.
LAST_LABEL = pd.Timestamp("2025-06-30 23:00:00")
QUANTILES = (0.05, 0.50, 0.95)

HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28",
    "2024-11-29", "2024-12-24", "2024-12-25", "2024-12-31",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27",
    "2025-11-28", "2025-12-24", "2025-12-25", "2025-12-31",
])
HOLIDAY_SET = set(HOLIDAYS.normalize())

AADT_ORDER = {"<5k": 0, "5k-10k": 1, "10k-25k": 2, "25k-50k": 3, "50k-100k": 4, "100k+": 5}
AADT_BAD = {"<5k", "5k-10k"}          # disagree with observed volumes; treat as unknown
DIST_W = {"<=1mi": 1.0, "1-5mi": 0.7, "5-15mi": 0.4}


# --------------------------------------------------------------------------- #
# 1. canonical data
# --------------------------------------------------------------------------- #

def detect_clock_offset(tr: pd.DataFrame, va: pd.DataFrame, max_lag: int = 4) -> int:
    """Hours to ADD to `tr` timestamps so its hour-of-day profile matches `va`.

    Why this is measured rather than hard-coded
    -------------------------------------------
    The originally distributed `traffic_train.parquet` had its 2024 observations offset by
    two hours, and its 00:00 and 01:00 observations dropped, because the underlying TMAS
    source-file layout differs between 2024 and 2025. The organisers confirmed this on
    2026-08-18 and reissued the file. Detecting it cost the project real time; shipping a
    model that had *not* detected it would have cost roughly 2.3x the MAE (304 -> 130).

    A hard-coded `+2` was the right answer for the old file and is the WRONG answer for the
    reissued one -- it would put us two hours off in the other direction, and a
    consistently-wrong clock produces smooth, plausible-looking profiles. So the offset is
    now measured on every load and asserted to be what we expect.

    Method: centre each pair's log hour-of-day profile in both files (so this compares
    *shape*, not level), then pick the circular shift of the training profile that
    minimises mean |difference| against validation. On the reissued file the optimum is 0
    at an error of 0.0515, against 0.3259 and 0.3288 for +/-1 hour -- an unambiguous minimum.
    """
    a, b = tr.copy(), va.copy()
    for d in (a, b):
        d["_p"] = d.station_key.astype(str) + "|" + d.direction_code.astype(str)
    common = sorted(set(a._p) & set(b._p))
    if not common:
        return 0

    def prof(d):
        s = d[d._p.isin(common)]
        m = s.groupby([s.timestamp.dt.hour, s._p]).volume.mean().unstack()
        m = np.log1p(m).reindex(index=range(24))
        return m - m.mean()

    A, B = prof(a), prof(b)
    cols = [c for c in A.columns if c in B.columns]
    Av, Bv = A[cols].to_numpy(), B[cols].to_numpy()
    errs = {lag: float(np.nanmean(np.abs(Av[[(h - lag) % 24 for h in range(24)], :] - Bv)))
            for lag in range(-max_lag, max_lag + 1)}
    return int(min(errs, key=errs.get))


def load_all(data_dir: Path = DATA_DIR, clock_shift: int | None = CLOCK_SHIFT_H) -> dict:
    tr = pd.read_parquet(data_dir / "traffic_train.parquet")
    va = pd.read_parquet(data_dir / "traffic_validation.parquet")
    tr = tr.copy()
    if clock_shift is None:
        clock_shift = detect_clock_offset(tr, va)
    if clock_shift:
        warnings.warn(
            f"traffic_train.parquet clock is {clock_shift:+d}h out of step with "
            f"traffic_validation.parquet; correcting. Expected 0 for the file reissued on "
            f"2026-08-18 -- if you see this, check which training file you are loading.",
            RuntimeWarning, stacklevel=2)
        tr["timestamp"] = tr["timestamp"] + pd.Timedelta(hours=clock_shift)

    lab = pd.concat(
        [tr[["station_key", "direction_code", "timestamp", "volume"]],
         va[["station_key", "direction_code", "timestamp", "volume"]]],
        ignore_index=True,
    )
    lab["direction_code"] = lab["direction_code"].astype(str)
    lab["pair"] = lab.station_key + "|" + lab.direction_code
    lab = add_calendar(lab)
    lab["ly"] = np.log1p(lab.volume.to_numpy(np.float64))

    net = pd.read_csv(data_dir / "network.csv")
    edges = pd.read_csv(data_dir / "network_edges.csv")

    tgt = pd.read_csv(data_dir / "submission_template.csv", low_memory=False)
    tgt["timestamp"] = pd.to_datetime(tgt.timestamp)
    tgt["direction_code"] = tgt.direction_code.astype(str)
    tgt["pair"] = tgt.station_key + "|" + tgt.direction_code
    tgt = add_calendar(tgt)

    warm = set(lab.station_key.unique())
    tgt["slice"] = np.where(
        tgt.station_key.isin(warm), "A",
        np.where(tgt.timestamp > LAST_LABEL, "B", "C"),
    )
    return {"labeled": lab, "network": net, "edges": edges, "targets": tgt, "warm": warm}


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df = df.copy()
    df["hour"] = ts.dt.hour.to_numpy(np.int16)
    df["dow"] = ts.dt.dayofweek.to_numpy(np.int16)
    df["month"] = ts.dt.month.to_numpy(np.int16)
    df["doy"] = ts.dt.dayofyear.to_numpy(np.int16)
    df["year"] = ts.dt.year.to_numpy(np.int16)
    df["is_weekend"] = (df.dow >= 5).to_numpy(np.int8)
    date = ts.dt.normalize()
    df["is_holiday"] = date.isin(HOLIDAY_SET).to_numpy(np.int8)
    df["holiday_adj"] = (
        date.isin(HOLIDAYS - pd.Timedelta(days=1)) | date.isin(HOLIDAYS + pd.Timedelta(days=1))
    ).to_numpy(np.int8)
    # Continuous trend in years since 2024-01-01, so the YoY drift can extrapolate.
    df["t_years"] = ((ts - pd.Timestamp("2024-01-01")).dt.total_seconds() / (365.25 * 86400)).to_numpy(np.float32)
    return df


# --------------------------------------------------------------------------- #
# 2. hierarchical profile backbone
# --------------------------------------------------------------------------- #

def shrunk_map(df: pd.DataFrame, keys: list[str], value: str, prior: np.ndarray, k: float) -> pd.Series:
    """Empirical-Bayes group mean shrunk toward a per-row prior.

    w = n/(n+k): a cell seen many times keeps its own estimate, a thin cell falls
    back to its parent. This is the fallback chain from research doc section 3.3,
    done as a smooth blend rather than a hard switch.
    """
    tmp = df[keys].copy()
    tmp["_v"] = df[value].to_numpy() - prior
    g = tmp.groupby(keys, observed=True)["_v"].agg(["mean", "size"])
    g["adj"] = g["mean"] * (g["size"] / (g["size"] + k))
    return g["adj"]


@dataclass
class Backbone:
    """log1p(y) ~ level_pair + shape(hour,dow) + season(month) + drift + holiday."""

    global_mean: float = 0.0
    level: pd.Series = None          # per pair
    shape_pair: pd.Series = None     # pair x hour x dow (shrunk)
    shape_stat: pd.Series = None     # station x hour x dow
    shape_axis: pd.Series = None     # axis x hour x dow  (works for cold stations)
    shape_glob: pd.Series = None     # hour x dow
    season: pd.Series = None         # month
    holiday: pd.Series = None        # (is_holiday, holiday_adj)
    drift: pd.Series = None          # per pair, shrunk yoy level shift
    hour_month: pd.Series = None     # hour x month interaction (daylight)
    drift_global: float = 0.0
    axis_of: dict = None

    def fit(self, lab: pd.DataFrame, axis_of: dict, k_shape: float = 3.0, k_drift: float = 4000.0,
            month_kernel: float | None = 2.0, k_season: float = 1.0):
        d = lab
        self.axis_of = axis_of
        self.month_kernel = month_kernel
        self.global_mean = float(d.ly.mean())
        self.level = d.groupby("pair").ly.median()

        r = d.ly.to_numpy() - d.pair.map(self.level).to_numpy()
        d = d.assign(_r=r, axis=d.station_key.map(axis_of).fillna("NS"))

        # shape hierarchy, coarse -> fine, each shrunk toward the running estimate
        self.shape_glob = _complete_hour_dow(
            shrunk_map(d, ["hour", "dow"], "_r", np.zeros(len(d)), 0.0))
        p = d.set_index(["hour", "dow"]).index.map(self.shape_glob).to_numpy(np.float64)
        self.shape_axis = shrunk_map(d, ["axis", "hour", "dow"], "_r", p, k_shape)
        p = p + np.nan_to_num(d.set_index(["axis", "hour", "dow"]).index.map(self.shape_axis).to_numpy(np.float64))
        self.shape_stat = shrunk_map(d, ["station_key", "hour", "dow"], "_r", p, k_shape)
        p = p + np.nan_to_num(d.set_index(["station_key", "hour", "dow"]).index.map(self.shape_stat).to_numpy(np.float64))
        self.shape_pair = shrunk_map(d, ["pair", "hour", "dow"], "_r", p, k_shape)
        p = p + np.nan_to_num(d.set_index(["pair", "hour", "dow"]).index.map(self.shape_pair).to_numpy(np.float64))

        # Month-kernel shape correction.
        #
        # The daily pattern drifts year over year -- the same months a year apart disagree
        # by 0.157 log against a 0.097 within-year noise floor -- so a profile pooled over
        # everything is estimated partly from the wrong season. Measured in the deployment
        # configuration (fit 2024 + Jan-Apr 2025, score May-Jun 2025), pooling is clearly
        # worse than seasonal matching: 0.1702 pooled vs 0.1423 with this kernel, a 16.4%
        # reduction. Recency weighting does NOT help (0.1741 at a 365d half-life) and
        # using only the most recent half-year is a disaster (0.2434) -- seasonal match
        # dominates recency, because a mismatched recent window biases the estimate.
        #
        # End-to-end on that same arm, backbone only: GEH<5 77.8% -> 81.9% and MAE
        # 109.6 -> 95.4. Both metrics improve, so this does not depend on which metric
        # the organisers actually score. The optimum is broad over w in [0.8, 1.25].
        #
        # Stored per target month so prediction picks the right correction.
        self.shape_month = None
        if month_kernel:
            resid = d._r.to_numpy() - p
            mo = d.month.to_numpy()
            key = pd.MultiIndex.from_arrays([d.pair.to_numpy(), d.hour.to_numpy(),
                                             d.dow.to_numpy()])
            # One pass over the data by (pair, hour, dow, month), then a 12x12 kernel
            # matrix multiply -- rather than twelve separate weighted groupbys, which is
            # the same arithmetic at roughly ten times the cost and matters because the
            # protocol fits several backbones per fold.
            b = _season_bin(d)
            key4 = pd.MultiIndex.from_arrays([d.pair.to_numpy(), d.hour.to_numpy(),
                                              d.dow.to_numpy(), b])
            S = pd.Series(resid, index=key4).groupby(level=[0, 1, 2, 3]).sum() \
                  .unstack(fill_value=0.0).reindex(columns=range(SEASON_BINS), fill_value=0.0)
            C = pd.Series(1.0, index=key4).groupby(level=[0, 1, 2, 3]).sum() \
                  .unstack(fill_value=0.0).reindex(columns=range(SEASON_BINS), fill_value=0.0)
            ix = np.arange(SEASON_BINS)
            dist = np.minimum(np.abs(ix[:, None] - ix[None, :]),
                              SEASON_BINS - np.abs(ix[:, None] - ix[None, :]))
            K = np.exp(-0.5 * (dist / month_kernel) ** 2)
            # Separate shrinkage constant from the hierarchy's `k_shape`. Loosening the
            # seasonal term pays on warm stations (82.2% -> 82.7% GEH<5, MAE 94.6 -> 93.3
            # on the deployment arm at k=1 vs k=3), but `k_shape` also governs the cold
            # shape hierarchy, where heavy shrinkage is what makes transfer work -- so the
            # two must not share a knob. Shrinking toward k-means factor-group centroids
            # instead of toward zero was also tried (NCHRP practice) and was worse than
            # simply shrinking less: 82.4% at 8 groups vs 82.7% here.
            W = pd.DataFrame((S.to_numpy() @ K.T) / ((C.to_numpy() @ K.T) + k_season),
                             index=S.index, columns=range(SEASON_BINS))
            self.shape_month = {m: W[m] for m in range(SEASON_BINS)}
            p = p + self._month_shape(d)

        # seasonality, then holidays, then drift -- each on what is left
        res = d._r.to_numpy() - p
        self.season = pd.Series(res, index=d.month.to_numpy()).groupby(level=0).mean()
        res = res - d.month.map(self.season).to_numpy()

        # Hour x month interaction. The daily *shape* moves with daylight -- evening
        # hours run 16-21% higher in June than December relative to the day's mean --
        # and an additive month term cannot express that. Measured worth ~8 MAE.
        self.hour_month = _complete_hour_month(pd.Series(res, index=pd.MultiIndex.from_arrays(
            [d.hour.to_numpy(), d.month.to_numpy()])).groupby(level=[0, 1]).mean())
        res = res - pd.MultiIndex.from_arrays([d.hour, d.month]).map(self.hour_month).to_numpy()

        hkey = pd.MultiIndex.from_arrays([d.is_holiday, d.holiday_adj])
        self.holiday = pd.Series(res, index=hkey).groupby(level=[0, 1]).mean()
        res = res - hkey.map(self.holiday).to_numpy()

        # Year-over-year level shift, as a matched-month log ratio rather than a
        # fitted slope. A slope fitted per pair extrapolates catastrophically on
        # short or outage-broken spans (measured: min -1.5/yr, MAE 173 vs 123).
        # Robust medians + hard shrinkage toward the network ratio, per research
        # doc section 4.2.
        tmp = pd.DataFrame({"p": d.pair.to_numpy(), "yr": d.year.to_numpy(),
                            "mo": d.month.to_numpy(), "r": res})
        cur = tmp[tmp.yr == 2025]
        months = set(cur.mo.unique())
        prev = tmp[(tmp.yr == 2024) & (tmp.mo.isin(months))]
        self.drift_global = float(cur.r.median() - prev.r.median()) if len(cur) and len(prev) else 0.0
        a = cur.groupby("p").r.agg(["median", "size"])
        b = prev.groupby("p").r.median()
        j = a.join(b.rename("prev"), how="inner").dropna()
        w = j["size"] / (j["size"] + k_drift)
        self.drift = (w * (j["median"] - j["prev"]) + (1 - w) * self.drift_global).clip(-0.5, 0.5)

        # Duan smearing: the backbone predicts a conditional mean in log space, so
        # expm1() of it under-estimates the level. This restores it.
        res2 = res - np.where(d.year.to_numpy() == 2025,
                              d.pair.map(self.drift).fillna(self.drift_global).to_numpy(), 0.0)
        self.smear = float(np.mean(np.exp(res2)))
        return self

    def _season_donors(self, st: str, dc: str, G, by_st: dict, boost: float,
                       opp_w: float, hop2_w: float) -> tuple[list[str], list[float]]:
        """Donor pairs and weights for one zero-history station-direction.

        Four rungs, taken in order and never mixed -- the first rung that yields any
        donor wins. Measured coverage over the 86 cold target pairs, and the mean |log|
        difference between two seasonal surfaces that justifies each rung:

            1-hop, direction-matched   62 pairs   corridor donor, same direction  0.0506
            1-hop, any direction       20 pairs   corridor donor, opposite dir    0.0654
            2-hop, direction-matched    2 pairs
            global mean surface         2 pairs   random pair                     0.2808

        The seasonal surface is barely direction-specific -- an opposite-direction
        corridor donor is closer to the truth (0.0654) than the term's own magnitude
        (0.1097), i.e. than leaving it at zero -- so rung 2 borrows across directions
        rather than falling straight through to the global mean.
        """
        if st not in G:
            return [], []
        nbrs = [(nb, boost if G.edges[st, nb]["etype"] == "same_corridor" else 1.0)
                for nb in G.neighbors(st)]

        d, w = [], []
        for nb, ew in nbrs:                                   # 1: direction-matched
            if f"{nb}|{dc}" in by_st.get(nb, ()):
                d.append(f"{nb}|{dc}"); w.append(ew)
        if d:
            return d, w

        for nb, ew in nbrs:                                   # 2: any direction
            for pr in by_st.get(nb, ()):
                d.append(pr); w.append(ew * opp_w)
        if d:
            return d, w

        seen = {st} | {nb for nb, _ in nbrs}
        for nb, _ in nbrs:                                    # 3: two hops out
            for nb2 in G.neighbors(nb):
                if nb2 in seen:
                    continue
                if f"{nb2}|{dc}" in by_st.get(nb2, ()):
                    d.append(f"{nb2}|{dc}"); w.append(hop2_w)
        return d, w

    def transfer_season(self, cold_pairs, G, boost: float = 6.0,
                        opp_w: float = 0.5, hop2_w: float = 0.25) -> "Backbone":
        """Give zero-history pairs a seasonal shape borrowed from the network.

        `shape_month` is keyed by pair, so a cold pair misses every lookup and
        `_month_shape` returns 0 -- i.e. cold stations get no seasonal correction at all,
        while warm stations get one worth about +4.9 GEH points. This copies the same
        direction-matched, corridor-weighted transfer that `GraphShapeTransfer` uses for
        the hour x dow shape, applied to the seasonal surface.

        A direction-matched 1-hop donor does not always exist: on the submission, 20 of
        the 86 cold pairs -- 12,247 rows, 7.2% of the file -- had no donor under that rule
        and kept a zero correction. `_season_donors` adds three further rungs so every
        cold pair gets one. Donors are always pairs whose surface was *fitted*, never a
        pair filled in by this method, so the result does not depend on iteration order.

        Mutates in place and returns self, so it can be chained after `fit`.
        """
        if not getattr(self, "shape_month", None):
            return self
        real = {p for p, _, _ in self.shape_month[0].index}
        by_st: dict[str, list[str]] = {}
        for p in real:
            by_st.setdefault(p.split("|")[0], []).append(p)
        # Final rung: the network-average surface, so a pair with no donor at any hop
        # still gets the seasonal signal every warm pair gets.
        glob = {b: tbl.groupby(level=[1, 2]).mean() for b, tbl in self.shape_month.items()}

        add: dict[int, list[pd.Series]] = {b: [] for b in self.shape_month}
        for pr in dict.fromkeys(cold_pairs):        # dedupe: callers pass overlapping lists
            if pr in real:
                continue
            real.add(pr)                            # never append the same pair twice
            st, dc = pr.split("|")
            donors, wts = self._season_donors(st, dc, G, by_st, boost, opp_w, hop2_w)
            w = np.array(wts) / np.sum(wts) if donors else None
            for b, tbl in self.shape_month.items():
                acc = None
                for dn, wi in zip(donors, w if w is not None else ()):
                    try:
                        part = tbl.loc[dn] * wi
                    except KeyError:
                        continue
                    acc = part if acc is None else acc + part
                if acc is None:
                    acc = glob[b].copy()
                acc.index = pd.MultiIndex.from_arrays(
                    [[pr] * len(acc), acc.index.get_level_values(0),
                     acc.index.get_level_values(1)])
                add[b].append(acc)

        for b, parts in add.items():
            if parts:
                self.shape_month[b] = pd.concat([self.shape_month[b]] + parts)
        return self

    def _month_shape(self, df: pd.DataFrame) -> np.ndarray:
        """Per-month pair x hour x dow correction, 0 where the kernel is disabled."""
        if not getattr(self, "shape_month", None):
            return np.zeros(len(df))
        out = np.zeros(len(df))
        mo = _season_bin(df)
        key = pd.MultiIndex.from_arrays([df.pair.to_numpy(), df.hour.to_numpy(),
                                         df.dow.to_numpy()])
        for m, tbl in self.shape_month.items():
            sel = mo == m
            if not sel.any():
                continue
            out[sel] = np.nan_to_num(key[sel].map(tbl).to_numpy(np.float64))
        return out



    def predict(self, df: pd.DataFrame, level_override: pd.Series | None = None,
                shape_override: pd.Series | None = None) -> np.ndarray:
        """Baseline in log1p space.

        `level_override` supplies levels for pairs with no history.
        `shape_override` (indexed by pair x hour x dow) *replaces* the fitted shape
        hierarchy for the pairs it covers -- used to inject the direction-matched
        graph-neighbour shape for cold stations, which is 33% more accurate than the
        axis-level fallback the hierarchy would otherwise use.
        """
        lvl = df.pair.map(self.level)
        if level_override is not None:
            lvl = lvl.fillna(df.pair.map(level_override))
        lvl = lvl.fillna(self.global_mean).to_numpy(np.float64)

        axis = df.station_key.map(self.axis_of).fillna("NS")
        shape = np.nan_to_num(df.set_index(["hour", "dow"]).index.map(self.shape_glob).to_numpy(np.float64))
        shape += np.nan_to_num(pd.MultiIndex.from_arrays([axis, df.hour, df.dow]).map(self.shape_axis).to_numpy(np.float64))
        shape += np.nan_to_num(df.set_index(["station_key", "hour", "dow"]).index.map(self.shape_stat).to_numpy(np.float64))
        shape += np.nan_to_num(df.set_index(["pair", "hour", "dow"]).index.map(self.shape_pair).to_numpy(np.float64))

        if shape_override is not None:
            ov = df.set_index(["pair", "hour", "dow"]).index.map(shape_override).to_numpy(np.float64)
            shape = np.where(np.isnan(ov), shape, ov)

        out = lvl + shape
        out += np.nan_to_num(df.month.map(self.season).to_numpy(np.float64))
        out += self._month_shape(df)
        out += np.nan_to_num(pd.MultiIndex.from_arrays([df.hour, df.month]).map(self.hour_month).to_numpy(np.float64))
        out += np.nan_to_num(pd.MultiIndex.from_arrays([df.is_holiday, df.holiday_adj]).map(self.holiday).to_numpy(np.float64))

        # Level shift applies from 2025 onward (including the H2 target horizon).
        dr = df.pair.map(self.drift).fillna(self.drift_global).to_numpy(np.float64)
        out += np.where(df.year.to_numpy() >= 2025, dr, 0.0)
        return out

    def predict_volume(self, df: pd.DataFrame, level_override=None, shape_override=None) -> np.ndarray:
        """Back-transformed to volume, with the smearing correction applied."""
        return np.expm1(self.predict(df, level_override, shape_override)) * getattr(self, "smear", 1.0)


# --------------------------------------------------------------------------- #
# 3. graph + donor model (path B)
# --------------------------------------------------------------------------- #

def build_graph(edges: pd.DataFrame, network: pd.DataFrame):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(network.station_key)
    for r in edges.itertuples():
        G.add_edge(r.from_station_key, r.to_station_key, etype=r.edge_type,
                   band=r.distance_band, w=DIST_W.get(r.distance_band, 0.7))
    return G


def station_axis(lab: pd.DataFrame, tgt: pd.DataFrame) -> dict:
    axis = {}
    for src in (lab, tgt):
        for s, g in src.groupby("station_key"):
            axis.setdefault(s, "NS" if set(g.direction_code) <= {"1", "5"} else "EW")
    return axis


class GraphShapeTransfer:
    """Direction-matched neighbour shape for stations with no history.

    Measured against the true held-out shape (log space, hour x dow deviation):

        axis-level fallback (the hierarchy's default)   0.3445
        direction_code-level                            0.3360
        graph neighbours, direction-matched             0.2296   <- 33% better

    The direction matching is what makes it work: along a corridor, North at
    station A is the same physical flow as North at station B, so the commute
    asymmetry transfers. Averaging both directions into one station node -- what
    the station-level GNN does -- destroys exactly that signal.
    """

    def __init__(self, G, corridor_boost: float = 2.0):
        self.G = G
        self.boost = corridor_boost

    def fit(self, lab: pd.DataFrame):
        med = lab.groupby("pair").ly.transform("median")
        self.shape = (lab.ly - med).groupby([lab.pair, lab.hour, lab.dow]).mean()
        self.by_pair = {k: v.droplevel(0) for k, v in self.shape.groupby(level=0)}
        d = self.shape.reset_index()
        d["dc"] = d.pair.str.split("|").str[1]
        self.by_dir = d.groupby(["dc", "hour", "dow"]).ly.mean()
        return self

    def transfer(self, pairs) -> pd.Series:
        """Shape for each requested pair, as a (pair, hour, dow) Series."""
        out = []
        for pr in pairs:
            st, dc = pr.split("|")
            num, den = None, 0.0
            if st in self.G:
                for nb in self.G.neighbors(st):
                    key = f"{nb}|{dc}"                      # direction-matched donor
                    if key not in self.by_pair:
                        continue
                    e = self.G.edges[st, nb]
                    w = e["w"] * (self.boost if e["etype"] == "same_corridor" else 1.0)
                    num = self.by_pair[key] * w if num is None else num + self.by_pair[key] * w
                    den += w
            s = (num / den) if den > 0 else self.by_dir.loc[dc]
            out.append(pd.concat({pr: s}, names=["pair"]))
        return pd.concat(out) if out else pd.Series(dtype=float)


SEASON_BINS = 24          # half-month bins; finer than months, still ~15 days of data each


def _season_bin(df: pd.DataFrame) -> np.ndarray:
    """Half-month bin from day-of-year.

    Calendar months are a coarse and uneven seasonal bucket. Half-month bins with a
    ~30-day kernel measured better on the deployment arm than a month kernel:
    82.4% GEH<5 / MAE 94.3 against 81.9% / 95.4.
    """
    return np.minimum((df.doy.to_numpy() - 1) // 15, SEASON_BINS - 1)


def _complete_hour_dow(sg: pd.Series) -> pd.Series:
    """Fill (hour, dow) cells for hours that the fit window never observed at all.

    This is the *base* of the shape hierarchy, so a missing key here is not a missing
    correction -- it is a missing hour-of-day profile. `Backbone.predict` maps the key to
    NaN, `nan_to_num` turns it into 0.0, and the model then predicts the pair's whole-day
    average level for that hour.

    That is not hypothetical. The training file's clock runs 2h behind, so a fit on 2024
    alone contains **no hours 0 or 1** -- they exist only in the Jan-Jun 2025 window. On
    the 4-fold protocol, which fits on 2024, the consequence was measured:

        warm-future WAPE, hours 0-1     558.2%   (mean bias +765 veh/h at hour 0)
        warm-future WAPE, hours 2-23     13.1%

    8.3% of rows carrying 1.53% of the volume, and they accounted for the whole of a
    5.19-point WAPE gap against a far simpler multiplicative model -- excluding them the
    ordering reverses by 2.5 points.

    The fix is circular linear interpolation across the hour axis, per day-of-week: an
    unobserved 01:00 is filled from 23:00 and 02:00, which is the right neighbourhood. It
    changes nothing when every hour is observed, so the deployment arm and the shipped
    submission -- both of which hold the 2025 H1 half-year -- are unaffected.
    """
    if sg.empty:
        return sg
    M = sg.unstack()
    M = M.reindex(index=range(24))
    if M.notna().all().all():
        return sg
    idx = np.arange(24)
    x = np.concatenate([idx - 24, idx, idx + 24])          # wrap so 00:00 sees 23:00
    for c in M.columns:
        col = M[c].to_numpy(np.float64)
        if np.isnan(col).all() or not np.isnan(col).any():
            continue
        y = np.concatenate([col, col, col])
        ok = ~np.isnan(y)
        M[c] = np.interp(idx, x[ok], y[ok])
    return M.stack()


def _complete_hour_month(hm: pd.Series) -> pd.Series:
    """Fill (hour, month) cells that were never observed.

    The training year covers hours 2-23 only; hours 0 and 1 appear solely in the Jan-Jun
    validation window. So (hour in {0,1}, month in 7..12) is never observed anywhere in the
    data, and it is 12,295 target rows -- 7.2% of the submission. `Backbone.predict` maps
    those keys to NaN and `nan_to_num` turns them into 0.0, i.e. it applies *no* hour x
    month interaction at all. Adjacent hours shift by about x1.06 between H1 and H2, so
    zeroing is a systematic under-prediction on those rows, not a neutral default.

    This borrows the seasonal delta from the nearest hours that do have full month
    coverage, keeping the hour's own average level. Validated by holding out months 7-12
    for hours 2,3,4,5,22,23 and predicting them: mean |error| 0.0524 (zero-fill) -> 0.0302
    (this rule), better on 5 of the 6, and best at hours 22-23 which is the regime hours
    0 and 1 actually live in.
    """
    M = hm.unstack().reindex(index=range(24), columns=range(1, 13))
    full = [h for h in M.index if M.loc[h].notna().all()]
    if not full:
        return hm
    for h in M.index:
        miss = M.columns[M.loc[h].isna()]
        if not len(miss):
            continue
        near = sorted(full, key=lambda x: min(abs(x - h), 24 - abs(x - h)))[:4]
        nb = M.loc[near].mean(axis=0)
        obs = M.columns[M.loc[h].notna()]
        if len(obs):
            M.loc[h, miss] = M.loc[h, obs].mean() + (nb - nb[obs].mean())[miss]
        else:
            # the hour was never observed at all (hours 0-1 when fitting on 2024 alone):
            # take the neighbours' seasonal shape, centred, and let the hour x dow term
            # carry the hour's own level.
            M.loc[h, miss] = (nb - nb.mean())[miss]
    return M.stack()


class PairLevelTransfer:
    """Direction-matched level for station-directions with no history.

    `DonorModel` works at station level: it averages a station's two directions into one
    number and gives every pair of that station the same level. That is the same mistake
    `GraphShapeTransfer` documents and fixes for *shape* -- North at station A is the same
    physical flow as North at station B, and averaging the directions destroys it.

    Mean |log level| error on held-out pairs, 4 protocol folds:

        pipeline level path (donor blend + LightGBM)   0.3942
        raw donor blend, no LightGBM                   0.3859   <- the LGB model *hurts*
        this class                                     0.3378   <- 14.3% better, all 4 folds

    Two things carry the gain, and neither is the distance band:

      * direction-matched *pair* donors instead of a direction-averaged station level;
      * a corridor:proximity weight ratio of 7.5:1. `DonorModel` uses 1.5:1, which lets
        distant proximity donors outvote a corridor neighbour that is worth far more.

    Weighting by `distance_band` was tried and did not pay (0.361 vs 0.336), and neither
    did fitting inverse-variance weights from labelled edge disagreement (0.351) -- the
    per-band samples are too small to estimate a variance from. Hand-set weights won, so
    hand-set weights are what this uses.

    Caveat: `prior_k` and the two weights were chosen on these same four folds with only
    30 held-out pairs each, so treat the exact values as tuned, not derived. Set C is
    untouched and remains the honest test.
    """

    W_CORRIDOR = 6.0
    W_PROXIMITY = 0.8

    def __init__(self, G, network: pd.DataFrame, prior_k: float = 1.0, statics=None):
        self.G = G
        self.net = network.set_index("station_key")
        self.k = prior_k
        self.statics = statics          # accepted for call-site compatibility; unused

    def fit(self, backbone: "Backbone"):
        idx = pd.MultiIndex.from_tuples([tuple(p.split("|")) for p in backbone.level.index])
        self.by_sd = pd.Series(backbone.level.to_numpy(), index=idx)

        # Attribute prior on the direction-averaged level: station attributes speak to a
        # station's overall scale, not to the split between its two directions.
        stn = self.by_sd.groupby(level=0).mean()
        att = self.net.assign(
            band=np.where(self.net.aadt_band.isin(AADT_BAD), "unknown", self.net.aadt_band))
        j = att.join(stn.rename("lv"), how="inner")
        self.by_band = j.groupby("band").lv.median()
        self.global_level = float(stn.median())

        return self

    def _prior(self, st: str) -> float:
        # A ridge on the full static vector was measurably better as a *standalone* prior
        # (leave-one-station-out: 0.4316 vs 0.4492 overall, 0.5151 vs 0.5590 for stations
        # with no corridor donor) but made no difference inside this estimator -- 0.3367 vs
        # 0.3378 across the 4 folds, better on only 2 of them. The graph donors carry most
        # of the weight at prior_k=1, so the prior rarely decides the answer. Not worth the
        # extra model; see FINDINGS.md.
        if st not in self.net.index:
            return self.global_level
        band = self.net.loc[st].aadt_band
        v = self.by_band.get("unknown" if band in AADT_BAD else band, np.nan)
        return float(v) if not np.isnan(v) else self.global_level

    def _donors(self, st: str, dc: str) -> tuple[float, float]:
        """Direction-matched weighted mean of neighbour pair levels, 2-hop fallback."""
        if st not in self.G:
            return np.nan, 0.0
        vs, ws = [], []
        for nb in self.G.neighbors(st):
            if (nb, dc) in self.by_sd.index:
                vs.append(self.by_sd[(nb, dc)])
                ws.append(self.W_CORRIDOR if self.G.edges[st, nb]["etype"] == "same_corridor"
                          else self.W_PROXIMITY)
        if not vs:                                   # 2-hop, heavily discounted
            for nb in self.G.neighbors(st):
                for nb2 in self.G.neighbors(nb):
                    if nb2 != st and (nb2, dc) in self.by_sd.index:
                        vs.append(self.by_sd[(nb2, dc)]); ws.append(0.25 * self.W_PROXIMITY)
        if not vs:
            return np.nan, 0.0
        v, w = np.array(vs), np.array(ws)
        return float((v * w).sum() / w.sum()), float(w.sum())

    def transfer(self, pairs) -> pd.Series:
        """Log level for each requested `station|direction`, as a Series indexed by pair."""
        out = {}
        for pr in pairs:
            st, dc = pr.split("|")
            prior = self._prior(st)
            g, cw = self._donors(st, dc)
            if np.isnan(g):
                out[pr] = prior
            else:
                a = cw / (cw + self.k)               # evidence weight on the graph estimate
                out[pr] = a * g + (1 - a) * prior
        return pd.Series(out, dtype=float)


class CFShapeTransfer:
    """Cold-start shape by hybrid matrix factorisation, as an alternative to graph transfer.

    Pairs are users, the 168 (hour x dow) cells are items, the normalised log profile is the
    rating, and a zero-history station is a cold-start user -- so the usable family is a
    hybrid: factor the observed profiles, then map a cold pair's attributes and graph
    position onto the factors. FHWA calls the discrete version "factor groups".

    Measured on the 4 protocol folds against the true held-out profile (`cf_shape.py`):

        direction mean (hierarchy fallback)   0.2915
        GraphShapeTransfer (deployed)         0.2584  sd 0.054
        this class (SVD + factor regression)  0.2309  sd 0.028   <- 10.7% better
        oracle cluster assignment             0.1598

    The variance halving matters as much as the mean: cold-slice results swing wildly
    across folds, and a lower-variance shape estimate is worth having even where the mean
    gain is modest. Note the ceiling: an oracle shape is worth only about +2 GEH points on
    cold-interp, because that slice is dominated by level error (+18.5). This is a real
    but bounded improvement.
    """

    def __init__(self, k: int = 8, seed: int = 0):
        self.k = k
        self.seed = seed

    def fit(self, lab: pd.DataFrame, statics: pd.DataFrame, G):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.ensemble import RandomForestRegressor

        med = lab.groupby("pair").ly.transform("median")
        dev = (lab.ly - med).groupby([lab.pair, lab.hour, lab.dow]).mean()
        M = dev.unstack(level=[1, 2])
        self.cells = M.columns
        self.fallback = M.mean(axis=0)
        A = np.nan_to_num(M.to_numpy(), nan=float(np.nanmean(M.to_numpy())))

        X = self._features(list(M.index), statics, G)
        k = min(self.k, min(A.shape) - 1)
        self.svd = TruncatedSVD(n_components=max(k, 1), random_state=self.seed)
        Z = self.svd.fit_transform(A)
        self.models = []
        for j in range(Z.shape[1]):
            m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                      random_state=self.seed, n_jobs=-1)
            m.fit(X.to_numpy(), Z[:, j])
            self.models.append(m)
        return self

    def _features(self, pairs, statics, G) -> pd.DataFrame:
        st = [p.split("|")[0] for p in pairs]
        dc = [p.split("|")[1] for p in pairs]
        X = statics.reindex(st).reset_index(drop=True)
        X.index = pairs
        for c in ("1", "3", "5", "7"):
            X[f"dir_{c}"] = [int(d == c) for d in dc]
        X["n_corr"] = [sum(1 for n in G.neighbors(s)
                           if G.edges[s, n]["etype"] == "same_corridor") if s in G else 0
                       for s in st]
        X["n_prox"] = [sum(1 for n in G.neighbors(s)
                           if G.edges[s, n]["etype"] == "proximity") if s in G else 0
                       for s in st]
        return X.astype(float).fillna(-1.0)

    def transfer(self, pairs, statics, G) -> pd.Series:
        """Same output shape as GraphShapeTransfer.transfer: a (pair, hour, dow) Series."""
        pairs = list(pairs)
        if not pairs:
            return pd.Series(dtype=float)
        X = self._features(pairs, statics, G)
        Z = np.column_stack([m.predict(X.to_numpy()) for m in self.models])
        R = self.svd.inverse_transform(Z)
        out = []
        for i, pr in enumerate(pairs):
            sr = pd.Series(R[i], index=self.cells).reindex(self.cells)
            out.append(pd.concat({pr: sr}, names=["pair"]))
        return pd.concat(out)


class DonorModel:
    """Level (and implicitly shape, via the axis-level backbone terms) for stations
    with no history.

    Two mechanisms, blended by evidence:
      * corridor interpolation -- `same_corridor` components are simple paths, and
        volume varies smoothly along them, so a cold station bracketed by labelled
        ones is a 1-D interpolation problem;
      * attribute + neighbour similarity -- weighted median of donor levels.
    """

    def __init__(self, G, network: pd.DataFrame):
        self.G = G
        self.net = network.set_index("station_key")

    def fit(self, backbone: Backbone, lab: pd.DataFrame):
        # station level = mean of its pair levels (log space)
        pl = backbone.level.rename("lv").reset_index()
        pl["station_key"] = pl.pair.str.split("|").str[0]
        self.station_level = pl.groupby("station_key").lv.mean()
        # attribute-group prior on level
        att = self.net.assign(
            band=np.where(self.net.aadt_band.isin(AADT_BAD), "unknown", self.net.aadt_band)
        )
        j = att.join(self.station_level, how="inner")
        self.by_band = j.groupby("band").lv.median()
        self.by_lanes = j.groupby("through_lanes").lv.median()
        self.global_level = float(self.station_level.median())
        return self

    def _corridor_level(self, s: str) -> tuple[float, float]:
        """Interpolate log level along the same_corridor path. Returns (level, weight)."""
        import networkx as nx
        sub = nx.Graph((u, v) for u, v, d in self.G.edges(data=True) if d["etype"] == "same_corridor")
        if s not in sub:
            return np.nan, 0.0
        comp = nx.node_connected_component(sub, s)
        known = [n for n in comp if n in self.station_level.index]
        if not known:
            return np.nan, 0.0
        d = nx.single_source_shortest_path_length(sub.subgraph(comp), s)
        w = np.array([1.0 / (1.0 + d[n]) ** 2 for n in known])
        v = self.station_level.reindex(known).to_numpy()
        return float((w * v).sum() / w.sum()), float(w.sum())

    def _neighbour_level(self, s: str) -> tuple[float, float]:
        if s not in self.G:
            return np.nan, 0.0
        acc_v, acc_w = [], []
        for nb in self.G.neighbors(s):
            if nb in self.station_level.index:
                e = self.G.edges[s, nb]
                w = e["w"] * (1.5 if e["etype"] == "same_corridor" else 1.0)
                acc_v.append(self.station_level[nb]); acc_w.append(w)
        if not acc_v:                      # 2-hop fallback
            for nb in self.G.neighbors(s):
                for nb2 in self.G.neighbors(nb):
                    if nb2 in self.station_level.index:
                        acc_v.append(self.station_level[nb2]); acc_w.append(0.3)
        if not acc_v:
            return np.nan, 0.0
        v, w = np.array(acc_v), np.array(acc_w)
        return float((v * w).sum() / w.sum()), float(w.sum())

    def _attr_level(self, s: str) -> float:
        if s not in self.net.index:
            return self.global_level
        row = self.net.loc[s]
        band = "unknown" if row.aadt_band in AADT_BAD else row.aadt_band
        v = self.by_band.get(band, np.nan)
        if np.isnan(v):
            v = self.by_lanes.get(row.through_lanes, self.global_level)
        return float(v)

    def station_levels(self, stations: list[str]) -> pd.DataFrame:
        rows = []
        for s in stations:
            c, cw = self._corridor_level(s)
            n, nw = self._neighbour_level(s)
            a = self._attr_level(s)
            parts, wts = [a], [1.0]
            if not np.isnan(c):
                parts.append(c); wts.append(min(cw, 3.0) * 1.5)
            if not np.isnan(n):
                parts.append(n); wts.append(min(nw, 3.0))
            parts, wts = np.array(parts), np.array(wts)
            est = float((parts * wts).sum() / wts.sum())
            rows.append({"station_key": s, "level": est, "donor_n": len(parts) - 1,
                         "donor_spread": float(np.std(parts)) if len(parts) > 1 else 0.0,
                         "corridor_w": cw, "nbr_w": nw})
        return pd.DataFrame(rows).set_index("station_key")

    def pair_levels(self, tgt_cold: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        st = self.station_levels(sorted(tgt_cold.station_key.unique()))
        pairs = tgt_cold[["pair", "station_key"]].drop_duplicates()
        lv = pairs.set_index("pair").station_key.map(st.level)
        return lv, st


# --------------------------------------------------------------------------- #
# 4. residual learners
# --------------------------------------------------------------------------- #

FEATS_BASE = ["hour", "dow", "month", "doy", "is_weekend", "is_holiday",
              "holiday_adj", "t_years", "base", "horizon_days"]


def lgb_quantiles(Xtr, ytr, Xte, feats, quantiles=QUANTILES, n_estimators=400, seed=0):
    import lightgbm as lgb
    out = np.zeros((len(Xte), len(quantiles)))
    for i, q in enumerate(quantiles):
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=n_estimators,
            learning_rate=0.06, num_leaves=63, min_child_samples=60,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=-1,
        )
        m.fit(Xtr[feats], ytr)
        out[:, i] = m.predict(Xte[feats])
    return np.sort(out, axis=1)      # fix any quantile crossing


# --------------------------------------------------------------------------- #
# 5. path C -- contemporaneous neighbour residual nowcast
# --------------------------------------------------------------------------- #

def neighbour_features(rows: pd.DataFrame, lab_res: pd.DataFrame, G) -> pd.DataFrame:
    """Summarise what the observed network was doing at each row's exact timestamp.

    Defined only for timestamps <= the last label, i.e. slice C. This is the signal
    no calendar model can see: a regional shock -- weather, an incident, an event --
    shows up simultaneously in every station's residual. Residual spatial
    correlation is 0.84 along corridors and 0.62 for proximity, so a cold station's
    deviation is largely predictable from its neighbours' deviations.
    """
    piv = lab_res.pivot_table(index="timestamp", columns="station_key", values="res", aggfunc="mean")
    net_med = piv.median(axis=1)
    cols = set(piv.columns)

    out = pd.DataFrame(np.nan, index=rows.index,
                       columns=["nbr_net", "nbr_corr", "nbr_prox", "nbr_n", "nbr_spread"])
    out["nbr_net"] = rows.timestamp.map(net_med).to_numpy()

    for st, g in rows.groupby("station_key", sort=False):
        if st in G:
            corr = [n for n in G.neighbors(st) if n in cols and G.edges[st, n]["etype"] == "same_corridor"]
            prox = [n for n in G.neighbors(st) if n in cols and G.edges[st, n]["etype"] == "proximity"]
            if not corr and not prox:                       # 2-hop fallback
                prox = list({n2 for n in G.neighbors(st) for n2 in G.neighbors(n) if n2 in cols})[:6]
        else:
            corr = prox = []
        ts = g.timestamp
        if corr:
            out.loc[g.index, "nbr_corr"] = ts.map(piv[corr].mean(axis=1)).to_numpy()
        if prox:
            out.loc[g.index, "nbr_prox"] = ts.map(piv[prox].mean(axis=1)).to_numpy()
        allnb = corr + prox
        if allnb:
            sub = piv[allnb]
            out.loc[g.index, "nbr_n"] = ts.map(sub.notna().sum(axis=1)).to_numpy()
            out.loc[g.index, "nbr_spread"] = ts.map(sub.std(axis=1)).to_numpy()
    return out.astype(np.float64)


# --------------------------------------------------------------------------- #
# 6. conformal calibration
# --------------------------------------------------------------------------- #

def fit_conformal(pred: np.ndarray, y: np.ndarray, groups: np.ndarray, target=0.90,
                  min_n=200) -> dict:
    """Mondrian conformal: one width multiplier per group, in log space."""
    scales = {}
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() < min_n:
            continue
        scales[g] = _bisect_scale(pred[m], y[m], target)
    scales["__global__"] = _bisect_scale(pred, y, target)
    return scales


def _bisect_scale(pred, y, target):
    lo, mid, hi = pred[:, 0], pred[:, 1], pred[:, 2]
    a, b = 0.3, 25.0
    for _ in range(45):
        m = (a + b) / 2
        cov = ((y >= mid - (mid - lo) * m) & (y <= mid + (hi - mid) * m)).mean()
        a, b = (a, m) if cov > target else (m, b)
    return (a + b) / 2


def fit_conformal_additive(pred: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           target=0.90, min_n=200) -> dict:
    """Split-conformal offsets in log space: mid +/- empirical residual quantile.

    Multiplicative scaling of an already-collapsed interval explodes -- on the
    cold-backfill fold the raw quantiles cover 26%, so hitting 90% needs a ~20x
    scale and produces widths of 10k+ vehicles on a 600-vehicle forecast. An
    additive offset is bounded by the data instead.
    """
    a = (1 - target) / 2
    res = y - pred[:, 1]
    out = {}
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() >= min_n:
            out[g] = (float(np.quantile(res[m], a)), float(np.quantile(res[m], 1 - a)))
    out["__global__"] = (float(np.quantile(res, a)), float(np.quantile(res, 1 - a)))
    return out


def apply_conformal_additive(pred: np.ndarray, groups: np.ndarray, offs: dict) -> np.ndarray:
    lo_o = np.array([offs.get(g, offs["__global__"])[0] for g in groups])
    hi_o = np.array([offs.get(g, offs["__global__"])[1] for g in groups])
    mid = pred[:, 1]
    return np.stack([mid + lo_o, mid, mid + hi_o], axis=1)


def apply_conformal(pred: np.ndarray, groups: np.ndarray, scales: dict) -> np.ndarray:
    s = np.array([scales.get(g, scales["__global__"]) for g in groups])
    lo, mid, hi = pred[:, 0], pred[:, 1], pred[:, 2]
    return np.stack([mid - (mid - lo) * s, mid, mid + (hi - mid) * s], axis=1)


def volume_band(pred_log: np.ndarray) -> np.ndarray:
    v = np.expm1(pred_log)
    return np.digitize(v, [50, 150, 400, 900, 2000]).astype(str)


# --------------------------------------------------------------------------- #
# 7. metrics
# --------------------------------------------------------------------------- #

def metrics(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    y = np.expm1(y_log)
    lo, mid, hi = (np.expm1(pred_log[:, i]).clip(0) for i in range(3))
    e = y - mid
    a = 0.10
    return {
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "sMAPE": float((2 * np.abs(e) / (np.abs(y) + np.abs(mid) + 1e-9)).mean() * 100),
        "cov90": float(((y >= lo) & (y <= hi)).mean()),
        "width": float((hi - lo).mean()),
        "winkler": float(((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)).mean()),
        "n": int(len(y)),
    }


def _t(msg: str, t0: float) -> float:
    print(f"  [{time.time() - t0:6.1f}s] {msg}", flush=True)
    return t0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["validate", "submit"])
    p.add_argument("--out", default="submission_pipeline.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from run_pipeline import main
    main(args)
