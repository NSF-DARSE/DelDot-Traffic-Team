"""
Day-1 traffic forecasting pipeline (honest, leakage-safe).
==========================================================
Method, in one line: a per-series seasonal profile (climatology) carries the
shape, a LightGBM model learns a correction on top using calendar + roadway
features, split-conformal residuals give 90% intervals, and series with no
history are filled from peer roads.

Everything a forecaster may use is restricted to the fitting window, so the
backtest (fit 2024, score H1 2025) is an honest rehearsal of the real task
(fit 2024 + H1 2025, predict H2 2025). No target values leak into features.

Built for the ORIGINAL 2024 file, whose hourly labels were later confirmed to be
offset by +2 hours with midnight/1 AM missing. The authentic day-one run left
that offset in place, so `load_all(apply_offset=False)` is the default; pass
apply_offset=True to realign by +2h and see what the correction is worth.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb

SERIES = ["station_key", "direction_code"]
TARGET = "volume"
LOWER_Q, UPPER_Q = 0.05, 0.95          # for the claimed 90% interval


# ------------------------------------------------------------------ load
def load_all(data_dir="day1", apply_offset=False):
    d = Path(data_dir)
    train = pd.read_parquet(d / "traffic_train.parquet")
    val = pd.read_parquet(d / "traffic_validation.parquet")
    net = pd.read_csv(d / "network.csv")
    edges = pd.read_csv(d / "network_edges.csv")
    template = pd.read_csv(d / "submission_template.csv")
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    val["timestamp"] = pd.to_datetime(val["timestamp"])
    # apply_offset=True realigns the (later-confirmed) +2h 2024 offset; the
    # authentic day-one run left it as-is, so the default is False.
    if apply_offset:
        train["timestamp"] = train["timestamp"] + pd.Timedelta(hours=2)
    for df in (train, val):
        df["station_key"] = _norm_key(df["station_key"])
        df["direction_code"] = _norm_dir(df["direction_code"])
        # recompute the calendar columns from the timestamp rather than trusting
        # the ones shipped in the parquet: they are stale if the offset is applied
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = df["day_of_week"] >= 5
    net["station_key"] = _norm_key(net["station_key"])
    # one row per station: node_features/level priors index net by station_key and
    # a duplicate would turn those scalar lookups into Series
    net = net.drop_duplicates(subset="station_key", keep="first").reset_index(drop=True)
    for c in ("from_station_key", "to_station_key"):
        edges[c] = _norm_key(edges[c])
    template["station_key"] = _norm_key(template["station_key"])
    return train, val, net, edges, template


def _norm_key(s):
    """station_key as a clean string. Files mix str/int/float reads; unify or merges
    silently produce no matches and every series looks like a cold start."""
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _norm_dir(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def add_calendar(df):
    df = df.copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        df["is_weekend"] = ts.dt.dayofweek >= 5
    if "station_key" in df.columns:
        df["station_key"] = _norm_key(df["station_key"])
    df["direction_code"] = _norm_dir(df["direction_code"])
    df["how"] = df["day_of_week"] * 24 + df["hour"]          # hour-of-week 0..167
    return df


# ------------------------------------------------------- climatology (shape)
def fit_climatology(history):
    """Median volume per (series, hour-of-week). This is the seasonal-naive core."""
    h = add_calendar(history)
    prof = (h.groupby(SERIES + ["how"])[TARGET].median()
              .rename("clim").reset_index())
    glob = h.groupby("how")[TARGET].median().rename("clim_glob").reset_index()
    return prof, glob


def apply_climatology(rows, prof, glob):
    r = add_calendar(rows).merge(prof, on=SERIES + ["how"], how="left")
    r = r.merge(glob, on="how", how="left")
    r["clim"] = r["clim"].fillna(r["clim_glob"]).fillna(0.0)
    return r["clim"].to_numpy()


# --------------------------------------------------------- roadway features
_AADT = {"<5k": 0, "5k-10k": 1, "10k-25k": 2, "25k-50k": 3, "50k-100k": 4, "100k+": 5}
_FC = {"Local": 0, "Minor Collector": 1, "Major Collector": 2, "Minor Arterial": 3,
       "Other Principal Arterials": 4, "Other Expressways & Freeway": 5, "Interstate": 6}

def static_table(net):
    """Roadway features for the ratio model.

    functional_class is blank for a large minority of stations and is filled with
    the Minor Arterial rung. That looks crude, but aadt_band is also a feature and
    is never missing, so the tree recovers the same information: adding lane_width,
    roadway_width_band, surface_type and an aadt-based class imputation was measured
    on the backtest and did not improve MAE or WAPE. Kept minimal on purpose.
    """
    s = net.copy()
    s["aadt_e"] = s["aadt_band"].map(_AADT).fillna(2)
    s["fc_e"] = s["functional_class"].map(_FC).fillna(3)
    s["lanes"] = pd.to_numeric(s["through_lanes"], errors="coerce").fillna(2)
    s["speed"] = pd.to_numeric(s["speed_limit"], errors="coerce").fillna(45)
    return s[["station_key", "aadt_e", "fc_e", "lanes", "speed"]]


def build_features(rows, clim, stat):
    r = add_calendar(rows)
    f = pd.DataFrame(index=r.index)
    f["hour"] = r["hour"]; f["dow"] = r["day_of_week"]; f["month"] = r["month"]
    f["is_weekend"] = r["is_weekend"].astype(int); f["how"] = r["how"]
    f["hour_sin"] = np.sin(2*np.pi*r["hour"]/24); f["hour_cos"] = np.cos(2*np.pi*r["hour"]/24)
    f["dir"] = pd.to_numeric(r["direction_code"], errors="coerce").fillna(0)
    f = f.join(r[["station_key"]]).merge(stat, on="station_key", how="left").drop(columns="station_key")
    f["clim"] = np.log1p(np.maximum(clim, 0))
    return f


# ------------------------------------------------------------------- model
FEATS = ["hour", "dow", "month", "is_weekend", "how", "hour_sin", "hour_cos",
         "dir", "aadt_e", "fc_e", "lanes", "speed", "clim"]


def train_model(history, prof, glob, stat, sample=450_000, seed=20260817):
    """LightGBM learns log-ratio target = log(volume+1) - log(clim+1)."""
    h = history
    if len(h) > sample:
        h = h.sample(sample, random_state=seed)
    clim = apply_climatology(h, prof, glob)
    X = build_features(h, clim, stat)[FEATS]
    y = np.log1p(h[TARGET].to_numpy()) - np.log1p(np.maximum(clim, 0))
    params = dict(objective="regression", metric="mae", learning_rate=0.05,
                  num_leaves=63, max_depth=7, min_child_samples=80,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                  reg_alpha=0.1, reg_lambda=1.0, verbose=-1, seed=seed)
    model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=400)
    return model


def predict(model, rows, prof, glob, stat):
    clim = apply_climatology(rows, prof, glob)
    X = build_features(rows, clim, stat)[FEATS]
    ratio = model.predict(X)
    pred = np.expm1(np.log1p(np.maximum(clim, 0)) + ratio)
    return np.maximum(pred, 0.0)


# ------------------------------------------------- split-conformal intervals
_BINS = [-1, 50, 200, 500, 1000, 2000, 1e9]          # shared by fit + apply: editing
_LABELS = ["b0", "b1", "b2", "b3", "b4", "b5"]       # one alone would misalign the map


def _buckets(pred):
    return pd.cut(np.asarray(pred, float), _BINS, labels=_LABELS)


def fit_intervals(actual, pred):
    """Residual quantiles per volume bucket -> heteroscedastic 90% bands."""
    df = pd.DataFrame({"a": np.asarray(actual, float), "p": np.asarray(pred, float)})
    df["res"] = df["a"] - df["p"]
    df["bucket"] = _buckets(df["p"])
    q = {}
    for b, g in df.groupby("bucket", observed=True):
        if len(g) >= 50:
            q[str(b)] = (np.quantile(g["res"], LOWER_Q), np.quantile(g["res"], UPPER_Q))
    q["__fallback__"] = (np.quantile(df["res"], LOWER_Q), np.quantile(df["res"], UPPER_Q))
    return q


def apply_intervals(pred, qmap, widen=1.0):
    """Vectorised: look the bucket quantiles up as a table instead of row by row."""
    pred = np.asarray(pred, float)
    fb = qmap["__fallback__"]
    b = pd.Categorical(_buckets(pred), categories=_LABELS)
    ql_by = np.array([qmap.get(l, fb)[0] for l in _LABELS], float)
    qh_by = np.array([qmap.get(l, fb)[1] for l in _LABELS], float)
    code = b.codes                                   # -1 for NaN/out-of-range predictions
    ql = np.where(code >= 0, ql_by[code], fb[0])
    qh = np.where(code >= 0, qh_by[code], fb[1])
    lo = pred + ql * widen
    hi = pred + qh * widen
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, np.maximum(lo, pred))
    lo = np.minimum(lo, pred)
    return lo, hi


# --------------------------------------------------------------- cold-start
_LADDER = [
    ["fc_e", "aadt_e", "dir", "how"],
    ["fc_e", "aadt_e", "how"],
    ["aadt_e", "how"],
    ["how"],
]


def cold_start(history, stat, rows):
    """Peer-group hourly profile for series with no history of their own."""
    h = add_calendar(history).merge(stat, on="station_key", how="left")
    h["dir"] = pd.to_numeric(h["direction_code"], errors="coerce").fillna(0)
    t = add_calendar(rows).merge(stat, on="station_key", how="left")
    t["dir"] = pd.to_numeric(t["direction_code"], errors="coerce").fillna(0)
    pred = np.full(len(t), np.nan); lo = np.full(len(t), np.nan); hi = np.full(len(t), np.nan)
    for keys in _LADDER:
        pending = np.isnan(pred)
        if not pending.any():
            break
        prof = (h.groupby(keys)[TARGET]
                  .agg(p="median", lo=lambda s: s.quantile(LOWER_Q),
                       hi=lambda s: s.quantile(UPPER_Q), n="count").reset_index())
        prof = prof[prof["n"] >= 20]
        if prof.empty:
            continue
        j = t.loc[pending, keys].merge(prof, on=keys, how="left")
        j.index = t.loc[pending].index
        ok = j["p"].notna()
        idx = j.index[ok]
        pred[t.index.get_indexer(idx)] = j.loc[ok, "p"].to_numpy()
        lo[t.index.get_indexer(idx)] = j.loc[ok, "lo"].to_numpy()
        hi[t.index.get_indexer(idx)] = j.loc[ok, "hi"].to_numpy()
    pred = np.nan_to_num(pred, nan=float(np.nanmedian(history[TARGET])))
    lo = np.nan_to_num(lo, nan=0.0); hi = np.nan_to_num(hi, nan=pred*1.5)
    lo = np.minimum(np.maximum(lo, 0), pred); hi = np.maximum(hi, pred)
    return pred, lo, hi
