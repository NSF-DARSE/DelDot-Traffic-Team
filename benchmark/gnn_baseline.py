"""
Relational-GCN baseline for the Traffic Pattern Analysis challenge.

Design goals
------------
* Fit comfortably on a GTX 1050 Ti (4 GB, Pascal) and stay usable on CPU.
* Handle the three target slices with one model (see PROBLEM_BRIEF.md §5.3):
  warm future, cold future, cold backfill.
* Emit the three numbers the submission actually needs -- point forecast plus a
  calibrated 90% interval -- directly from the network, via quantile regression.

Architecture
------------
    station attributes ──┐
                         ├─> R-GCN over the station graph ──> z_station[127, d]
    graph adjacency   ───┘                                        │
                                                                  ├─> MLP head ──> 3 quantiles
    learned station id embedding (warm only, randomly dropped) ───┤    (log1p space)
    direction embedding + calendar features ──────────────────────┘

The station graph is tiny (127 nodes, 186 edges), so message passing runs as two
dense matmuls per relation over the *whole* graph once per forward pass. Cost is
negligible; the per-row MLP dominates.

The learned station-id embedding is dropped at random during training
(``--station-dropout``). That forces the attribute+graph pathway to carry the
prediction on its own, which is exactly the situation for the 43 target stations
that have no history at all.

Usage
-----
    python gnn_baseline.py train  --split temporal --epochs 30
    python gnn_baseline.py train  --split cold     --epochs 30
    python gnn_baseline.py submit --epochs 30 --out my_submission.csv

Interpreter
-----------
Use the ``Data-Science-Kernel`` venv for GPU support::

    ~/.virtualenvs/data_science/bin/python gnn_baseline.py train --epochs 30

That env has torch 2.6.0+cu124 (arch list includes sm_60), which runs on the
GTX 1050 Ti: ~1.0 s/epoch vs ~8.0 s on CPU. The base conda env has
torch 2.13.0+cu130, built for sm_75 and up -- there ``cuda.is_available()``
returns True and every kernel then fails, so ``pick_device`` falls back to CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from traffic_pipeline import DATA_DIR          # resolves to the repo root

# The training file's clock runs 2h behind validation and the targets.
# See PROBLEM_BRIEF.md §5.1 -- this single line is worth ~57% of baseline MAE.
TRAIN_CLOCK_SHIFT_HOURS = 2

QUANTILES = (0.05, 0.50, 0.95)

# US federal holidays plus the observed shifts, 2024-2025, and the few extra
# dates that behave like holidays on this network (Christmas Eve, day after
# Thanksgiving). Traffic on these drops 17-50%.
HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28",
    "2024-11-29", "2024-12-24", "2024-12-25", "2024-12-31",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27",
    "2025-11-28", "2025-12-24", "2025-12-25", "2025-12-31",
])

# aadt_band is a strong level prior for four of its six levels. `<5k` and
# `5k-10k` disagree badly with the observed volumes (PROBLEM_BRIEF.md §6.4), so
# they get flagged as unreliable instead of trusted as ordinal values.
AADT_ORDER = {"<5k": 0, "5k-10k": 1, "10k-25k": 2, "25k-50k": 3, "50k-100k": 4, "100k+": 5}
AADT_UNRELIABLE = {"<5k", "5k-10k"}
WIDTH_ORDER = {"<24": 0, "24-35": 1, "36-47": 2, "48-59": 3, "60+": 4}


# --------------------------------------------------------------------------- #
# device
# --------------------------------------------------------------------------- #

def pick_device(requested: str = "auto") -> torch.device:
    """Pick a device, and actually verify it can run a kernel.

    ``torch.cuda.is_available()`` returns True on a GPU whose compute capability
    the installed wheel was not built for (e.g. torch+cu130 on a Pascal card);
    the failure only shows up later as "no kernel image is available". Probe with
    a real matmul so the fallback happens here rather than mid-training.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cpu")
    try:
        torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as exc:  # noqa: BLE001 - any CUDA failure means fall back
        cap = torch.cuda.get_device_capability(0)
        msg = (
            f"CUDA reported available but is unusable (sm_{cap[0]}{cap[1]}): {exc}\n"
            f"  installed torch {torch.__version__} builds for {torch.cuda.get_arch_list()}\n"
            "  fix: run with the Data-Science-Kernel interpreter instead --\n"
            "       ~/.virtualenvs/data_science/bin/python (torch 2.6.0+cu124, has sm_60)"
        )
        if requested == "cuda":
            raise RuntimeError(msg) from exc
        print(f"[device] {msg}\n[device] falling back to CPU")
        return torch.device("cpu")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    """Everything the model needs, as plain arrays."""

    # per-row
    X: np.ndarray           # calendar features, float32 [n, n_time_feat]
    station_idx: np.ndarray # int64 [n]
    dir_idx: np.ndarray     # int64 [n]
    y: np.ndarray           # log1p(volume), float32 [n]
    ts: np.ndarray          # datetime64 [n], for splitting only

    # per-station (graph side)
    station_feat: np.ndarray  # float32 [n_stations, n_static_feat]
    adj: list[np.ndarray]     # normalized dense adjacency per relation
    stations: list[str]       # index -> station_key
    warm_mask: np.ndarray     # bool [n_stations], True if the station has labels

    feature_names: list[str] = field(default_factory=list)


def _cyc(values: np.ndarray, period: float, harmonics: int = 1) -> list[np.ndarray]:
    """Sine/cosine encoding. Keeps hour continuous, which matters because train
    has no data at all for hours 0 and 1 after the clock fix -- a lookup table
    cannot fill those, a cyclic feature interpolates across them."""
    out = []
    for k in range(1, harmonics + 1):
        angle = 2 * np.pi * k * values / period
        out += [np.sin(angle), np.cos(angle)]
    return out


def build_time_features(ts: pd.Series) -> tuple[np.ndarray, list[str]]:
    ts = pd.to_datetime(ts)
    hour = ts.dt.hour.to_numpy(np.float32)
    dow = ts.dt.dayofweek.to_numpy(np.float32)
    doy = ts.dt.dayofyear.to_numpy(np.float32)

    date = ts.dt.normalize()
    hol = date.isin(HOLIDAYS).to_numpy(np.float32)
    hol_adj = (
        date.isin(HOLIDAYS - pd.Timedelta(days=1))
        | date.isin(HOLIDAYS + pd.Timedelta(days=1))
    ).to_numpy(np.float32)

    # Linear trend, so the model can pick up the ~-10% year-over-year drift and
    # extrapolate it into H2 2025 instead of freezing 2024 levels.
    trend = ((ts - pd.Timestamp("2024-01-01")).dt.total_seconds() / (365.25 * 86400)).to_numpy(np.float32)

    cols = (
        _cyc(hour, 24, harmonics=3)      # 3 harmonics: the daily curve is not a sine
        + _cyc(dow, 7, harmonics=2)
        + _cyc(doy, 365.25, harmonics=2)
        + [(dow >= 5).astype(np.float32), hol, hol_adj, trend]
    )
    names = (
        [f"hour_{f}{k}" for k in (1, 2, 3) for f in ("sin", "cos")]
        + [f"dow_{f}{k}" for k in (1, 2) for f in ("sin", "cos")]
        + [f"doy_{f}{k}" for k in (1, 2) for f in ("sin", "cos")]
        + ["is_weekend", "holiday", "holiday_adjacent", "trend"]
    )
    return np.stack(cols, axis=1).astype(np.float32), names


def build_station_features(
    network: pd.DataFrame, edges: pd.DataFrame, stations: list[str], axis_map: dict[str, str]
) -> tuple[np.ndarray, list[str]]:
    net = network.set_index("station_key").reindex(stations)

    parts, names = [], []

    def num(col: str) -> None:
        v = net[col].to_numpy(np.float32)
        missing = np.isnan(v).astype(np.float32)
        v = np.nan_to_num(v, nan=float(np.nanmedian(v)))
        parts.append(((v - v.mean()) / (v.std() + 1e-6))[:, None])
        parts.append(missing[:, None])
        names.extend([col, f"{col}_missing"])

    for c in ("through_lanes", "speed_limit", "lane_width"):
        num(c)

    def onehot(col: str) -> None:
        d = pd.get_dummies(net[col].fillna("__missing__"), prefix=col)
        parts.append(d.to_numpy(np.float32))
        names.extend(d.columns.tolist())

    onehot("functional_class")
    onehot("surface_type")

    aadt = net["aadt_band"]
    ordinal = aadt.map(AADT_ORDER).fillna(2.5).to_numpy(np.float32) / 5.0
    unreliable = aadt.isin(AADT_UNRELIABLE).to_numpy(np.float32)
    # Zero out the ordinal where the band disagrees with observed volume, so the
    # model leans on the graph for those stations instead of a wrong prior.
    parts.append((ordinal * (1 - unreliable))[:, None])
    parts.append(unreliable[:, None])
    names.extend(["aadt_ordinal", "aadt_unreliable"])

    width = net["roadway_width_band"].map(WIDTH_ORDER).fillna(2).to_numpy(np.float32) / 4.0
    parts.append(width[:, None])
    names.append("roadway_width")

    # Graph position. Cheap, and it separates corridor stations from isolated ones.
    deg = {s: 0 for s in stations}
    corridor_deg = {s: 0 for s in stations}
    for r in edges.itertuples():
        for s in (r.from_station_key, r.to_station_key):
            if s in deg:
                deg[s] += 1
                corridor_deg[s] += int(r.edge_type == "same_corridor")
    d = np.array([deg[s] for s in stations], np.float32)
    cd = np.array([corridor_deg[s] for s in stations], np.float32)
    parts.append((d / 6.0)[:, None])
    parts.append((cd / 3.0)[:, None])
    names.extend(["degree", "corridor_degree"])

    # Travel axis (N/S vs E/W), recoverable for cold stations too because the
    # target file lists both of their direction codes.
    ns = np.array([axis_map.get(s) == "NS" for s in stations], np.float32)
    ew = np.array([axis_map.get(s) == "EW" for s in stations], np.float32)
    parts.extend([ns[:, None], ew[:, None]])
    names.extend(["axis_ns", "axis_ew"])

    return np.concatenate(parts, axis=1).astype(np.float32), names


def build_adjacency(edges: pd.DataFrame, stations: list[str]) -> list[np.ndarray]:
    """One symmetric, degree-normalized dense adjacency per edge type.

    Kept separate rather than merged: same_corridor edges carry residual
    correlation 0.84 vs 0.62 for proximity, so they deserve their own weights.
    """
    idx = {s: i for i, s in enumerate(stations)}
    n = len(stations)
    mats = []
    for etype in ("same_corridor", "proximity"):
        a = np.zeros((n, n), np.float32)
        sub = edges[edges.edge_type == etype]
        for r in sub.itertuples():
            i, j = idx.get(r.from_station_key), idx.get(r.to_station_key)
            if i is None or j is None:
                continue
            # Nearer neighbours weigh more; correlation decays with distance.
            w = {"<=1mi": 1.0, "1-5mi": 0.7, "5-15mi": 0.4}.get(r.distance_band, 0.7)
            a[i, j] = a[j, i] = w
        deg = a.sum(1, keepdims=True)
        mats.append(a / np.clip(deg, 1e-6, None))  # row-normalized mean aggregation
    return mats


def load_dataset(data_dir: Path = DATA_DIR, include_targets: bool = False) -> tuple[Dataset, pd.DataFrame]:
    train = pd.read_parquet(data_dir / "traffic_train.parquet")
    val = pd.read_parquet(data_dir / "traffic_validation.parquet")
    network = pd.read_csv(data_dir / "network.csv")
    edges = pd.read_csv(data_dir / "network_edges.csv")
    targets = pd.read_csv(data_dir / "submission_template.csv")
    targets["timestamp"] = pd.to_datetime(targets["timestamp"])

    # THE clock fix. Everything downstream assumes the validation convention.
    train = train.copy()
    train["timestamp"] = train["timestamp"] + pd.Timedelta(hours=TRAIN_CLOCK_SHIFT_HOURS)

    labeled = pd.concat(
        [train[["station_key", "direction_code", "timestamp", "volume"]],
         val[["station_key", "direction_code", "timestamp", "volume"]]],
        ignore_index=True,
    )
    # direction_code is str in the parquets and int64 in the CSVs. Normalize once.
    labeled["direction_code"] = labeled["direction_code"].astype(str)
    targets["direction_code"] = targets["direction_code"].astype(str)

    stations = sorted(network["station_key"].unique())
    st_idx = {s: i for i, s in enumerate(stations)}
    dirs = ["1", "3", "5", "7"]
    dir_idx = {d: i for i, d in enumerate(dirs)}

    axis_map: dict[str, str] = {}
    for src in (labeled, targets):
        for s, g in src.groupby("station_key"):
            codes = set(g["direction_code"])
            axis_map.setdefault(s, "NS" if codes <= {"1", "5"} else "EW")

    station_feat, static_names = build_station_features(network, edges, stations, axis_map)
    adj = build_adjacency(edges, stations)

    X, time_names = build_time_features(labeled["timestamp"])
    # .copy() keeps the arrays writable; torch.from_numpy warns on read-only views.
    ds = Dataset(
        X=X,
        station_idx=labeled["station_key"].map(st_idx).to_numpy(np.int64).copy(),
        dir_idx=labeled["direction_code"].map(dir_idx).to_numpy(np.int64).copy(),
        y=np.log1p(labeled["volume"].to_numpy(np.float32)).copy(),
        ts=labeled["timestamp"].to_numpy("datetime64[ns]"),
        station_feat=station_feat,
        adj=adj,
        stations=stations,
        warm_mask=np.array([s in set(labeled["station_key"]) for s in stations]),
        feature_names=time_names + static_names,
    )

    if include_targets:
        tX, _ = build_time_features(targets["timestamp"])
        targets = targets.assign(
            _X=list(tX),
            _station=targets["station_key"].map(st_idx).to_numpy(np.int64),
            _dir=targets["direction_code"].map(dir_idx).to_numpy(np.int64),
        )
    return ds, targets


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class RGCN(nn.Module):
    """Relational GCN over the station graph. Dense on purpose -- 127 nodes."""

    def __init__(self, in_dim: int, hidden: int, n_relations: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict({
                    "self": nn.Linear(d, hidden),
                    "rel": nn.ModuleList([nn.Linear(d, hidden, bias=False) for _ in range(n_relations)]),
                    "norm": nn.LayerNorm(hidden),
                })
            )
            d = hidden
        self.dropout = dropout

    def forward(self, h: torch.Tensor, adj: list[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            out = layer["self"](h)
            for a, lin in zip(adj, layer["rel"]):
                out = out + a @ lin(h)
            h = F.dropout(F.gelu(layer["norm"](out)), self.dropout, self.training)
        return h


class TrafficGNN(nn.Module):
    def __init__(
        self,
        n_static: int,
        n_time: int,
        n_stations: int,
        n_relations: int,
        hidden: int = 64,
        head_hidden: int = 128,
        n_quantiles: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gnn = RGCN(n_static, hidden, n_relations, layers=2, dropout=dropout)
        # Identity embedding for stations we have history for. Randomly dropped
        # during training so the graph path learns to stand alone (cold start).
        self.station_id = nn.Embedding(n_stations, hidden)
        nn.init.normal_(self.station_id.weight, std=0.02)
        self.dir_emb = nn.Embedding(4, 8)

        # +1 for the "do I have an identity embedding for this station" flag.
        # Without it the quantile head cannot tell a warm row from a cold one and
        # emits the same interval width for both -- which collapses coverage on
        # cold stations (measured 0.46 against a nominal 0.90 before this flag).
        in_dim = hidden * 2 + 8 + n_time + 1
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        # Predict the median directly, and the two tails as positive offsets so
        # lower <= median <= upper holds by construction (the validator checks it).
        self.out_median = nn.Linear(head_hidden, 1)
        self.out_spread = nn.Linear(head_hidden, n_quantiles - 1)

    def station_repr(
        self, station_feat: torch.Tensor, adj: list[torch.Tensor], id_keep: torch.Tensor
    ) -> torch.Tensor:
        z = self.gnn(station_feat, adj)                       # attributes + graph
        ident = self.station_id.weight * id_keep[:, None]     # zeroed where dropped/cold
        return torch.cat([z, ident], dim=1)

    def forward(
        self,
        station_feat: torch.Tensor,
        adj: list[torch.Tensor],
        id_keep: torch.Tensor,
        station_idx: torch.Tensor,
        dir_idx: torch.Tensor,
        x_time: torch.Tensor,
    ) -> torch.Tensor:
        zs = self.station_repr(station_feat, adj, id_keep)[station_idx]
        has_id = id_keep[station_idx][:, None]
        h = self.head(torch.cat([zs, self.dir_emb(dir_idx), x_time, has_id], dim=1))
        median = self.out_median(h)
        spread = F.softplus(self.out_spread(h)) + 1e-3
        lower = median - spread[:, :1]
        upper = median + spread[:, 1:]
        return torch.cat([lower, median, upper], dim=1)       # log1p space


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles=QUANTILES) -> torch.Tensor:
    """Quantile (pinball) loss. Optimising this gives calibrated intervals for
    free -- no separate conformal step, and the heteroscedasticity across volume
    levels is absorbed by the network."""
    t = target[:, None]
    losses = []
    for i, q in enumerate(quantiles):
        e = t - pred[:, i : i + 1]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return torch.cat(losses, dim=1).mean()


# --------------------------------------------------------------------------- #
# splits and metrics
# --------------------------------------------------------------------------- #

def make_split(ds: Dataset, kind: str, cutoff: str = "2025-05-01", n_cold: int = 15, seed: int = 0):
    """Return (train_rows, val_rows, cold_station_ids).

    temporal -- train on everything before `cutoff`, score the rest. Mimics the
                warm-future slice (extrapolating past the end of the labels).
    cold     -- same time cut, but also remove a set of stations from training
                entirely and mask their id embedding. Mimics the 43 target
                stations that have no history at all.
    full     -- everything is training data (used for the real submission).
    """
    ts = ds.ts
    if kind == "full":
        return np.arange(len(ts)), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    cut = np.datetime64(pd.Timestamp(cutoff))
    tr = np.flatnonzero(ts < cut)
    va = np.flatnonzero(ts >= cut)
    cold: np.ndarray = np.array([], dtype=np.int64)

    if kind == "cold":
        rng = np.random.default_rng(seed)
        warm = np.flatnonzero(ds.warm_mask)
        cold = rng.choice(warm, size=min(n_cold, len(warm)), replace=False)
        tr = tr[~np.isin(ds.station_idx[tr], cold)]
        va = va[np.isin(ds.station_idx[va], cold)]  # score only the held-out stations
    return tr, va, cold


def metrics(y_true_log: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    """Report in the original volume space -- that is what gets scored."""
    y = np.expm1(y_true_log)
    lo, mid, hi = (np.expm1(pred_log[:, i]).clip(0) for i in range(3))
    err = y - mid
    cover = ((y >= lo) & (y <= hi)).mean()
    a = 0.10
    winkler = ((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)).mean()
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "sMAPE": float((2 * np.abs(err) / (np.abs(y) + np.abs(mid) + 1e-9)).mean() * 100),
        "PI_coverage": float(cover),
        "PI_width": float((hi - lo).mean()),
        "Winkler90": float(winkler),
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def train_model(
    ds: Dataset,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    cold_stations: np.ndarray,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 8192,
    lr: float = 3e-3,
    hidden: int = 64,
    head_hidden: int = 128,
    station_dropout: float = 0.5,
    dropout: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[TrafficGNN, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # The whole dataset is ~120 MB as float32; keep it resident on the device and
    # index it directly. No DataLoader, no per-batch host->device copies.
    X = torch.from_numpy(ds.X).to(device)
    y = torch.from_numpy(ds.y).to(device)
    s_idx = torch.from_numpy(ds.station_idx).to(device)
    d_idx = torch.from_numpy(ds.dir_idx).to(device)
    sf = torch.from_numpy(ds.station_feat).to(device)
    adj = [torch.from_numpy(a).to(device) for a in ds.adj]

    # A station's id embedding is usable only if it has training rows.
    trainable_ids = np.zeros(len(ds.stations), np.float32)
    trainable_ids[np.unique(ds.station_idx[train_rows])] = 1.0
    trainable_ids[cold_stations] = 0.0
    id_avail = torch.from_numpy(trainable_ids).to(device)

    model = TrafficGNN(
        n_static=ds.station_feat.shape[1],
        n_time=ds.X.shape[1],
        n_stations=len(ds.stations),
        n_relations=len(ds.adj),
        hidden=hidden,
        head_hidden=head_hidden,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr = torch.from_numpy(train_rows).to(device)
    steps = max(1, len(train_rows) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)

    history: list[dict] = []
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[model] {n_params:,} params | {len(train_rows):,} train rows | {steps} steps/epoch | {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        total, t0 = 0.0, time.time()
        for k in range(steps):
            b = perm[k * batch_size : (k + 1) * batch_size]
            # Station-level dropout: hide the identity embedding for a random
            # subset each step, so the graph path is trained to work alone.
            keep = id_avail * (torch.rand(len(ds.stations), device=device) > station_dropout).float()
            pred = model(sf, adj, keep, s_idx[b], d_idx[b], X[b])
            loss = pinball_loss(pred, y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()

        row = {"epoch": epoch, "train_loss": total / steps, "sec": time.time() - t0}
        if len(val_rows):
            pred = predict(model, ds, val_rows, device, id_avail=id_avail)
            row.update(metrics(ds.y[val_rows], pred))
        history.append(row)
        if verbose:
            extra = "".join(f" | {k} {row[k]:.3f}" for k in ("MAE", "PI_coverage") if k in row)
            print(f"  epoch {epoch:3d} | loss {row['train_loss']:.4f} | {row['sec']:.1f}s{extra}")

    return model, {"history": history, "n_params": n_params}


@torch.no_grad()
def predict(
    model: TrafficGNN,
    ds: Dataset,
    rows: np.ndarray,
    device: torch.device,
    id_avail: torch.Tensor | None = None,
    batch_size: int = 65536,
) -> np.ndarray:
    model.eval()
    if id_avail is None:
        id_avail = torch.from_numpy(ds.warm_mask.astype(np.float32)).to(device)
    sf = torch.from_numpy(ds.station_feat).to(device)
    adj = [torch.from_numpy(a).to(device) for a in ds.adj]
    out = []
    for k in range(0, len(rows), batch_size):
        b = rows[k : k + batch_size]
        out.append(
            model(
                sf, adj, id_avail,
                torch.from_numpy(ds.station_idx[b]).to(device),
                torch.from_numpy(ds.dir_idx[b]).to(device),
                torch.from_numpy(ds.X[b]).to(device),
            ).cpu().numpy()
        )
    return np.concatenate(out)


@torch.no_grad()
def predict_targets(
    model: TrafficGNN, ds: Dataset, targets: pd.DataFrame, device: torch.device, batch_size: int = 65536
) -> np.ndarray:
    model.eval()
    sf = torch.from_numpy(ds.station_feat).to(device)
    adj = [torch.from_numpy(a).to(device) for a in ds.adj]
    id_avail = torch.from_numpy(ds.warm_mask.astype(np.float32)).to(device)
    X = np.stack(targets["_X"].to_numpy()).astype(np.float32)
    s = targets["_station"].to_numpy(np.int64)
    d = targets["_dir"].to_numpy(np.int64)
    out = []
    for k in range(0, len(targets), batch_size):
        sl = slice(k, k + batch_size)
        out.append(
            model(
                sf, adj, id_avail,
                torch.from_numpy(s[sl]).to(device),
                torch.from_numpy(d[sl]).to(device),
                torch.from_numpy(X[sl]).to(device),
            ).cpu().numpy()
        )
    return np.concatenate(out)


def fit_interval_scale(pred_log: np.ndarray, y_log: np.ndarray, target: float = 0.90) -> float:
    """Multiplier on the predicted spread that hits `target` empirical coverage.

    The pinball head is calibrated for rows it trained on, but not for stations it
    has never seen: station dropout hides the identity embedding, yet the model has
    still fitted that station's attribute vector, so a dropped-id training row is
    easier than a genuinely unseen station. Measured cold coverage lands near 0.50
    against a nominal 0.90. This is the standard conformal correction -- fit the
    multiplier on held-out data, apply it at submission time.
    """
    lo, mid, hi = pred_log[:, 0], pred_log[:, 1], pred_log[:, 2]
    a, b = 0.5, 20.0
    for _ in range(40):  # bisection on coverage, which is monotone in the scale
        m = (a + b) / 2
        cov = ((y_log >= mid - (mid - lo) * m) & (y_log <= mid + (hi - mid) * m)).mean()
        a, b = (a, m) if cov > target else (m, b)
    return (a + b) / 2


def apply_interval_scale(pred_log: np.ndarray, scale: float) -> np.ndarray:
    lo, mid, hi = pred_log[:, 0], pred_log[:, 1], pred_log[:, 2]
    return np.stack([mid - (mid - lo) * scale, mid, mid + (hi - mid) * scale], axis=1)


def make_submission(
    pred_log: np.ndarray,
    targets: pd.DataFrame,
    out_path: Path,
    scale_warm: float = 1.0,
    scale_cold: float = 1.0,
) -> pd.DataFrame:
    warm = targets["_warm"].to_numpy()
    pred_log = np.where(
        warm[:, None],
        apply_interval_scale(pred_log, scale_warm),
        apply_interval_scale(pred_log, scale_cold),
    )
    lo, mid, hi = (np.expm1(pred_log[:, i]) for i in range(3))
    lo, mid, hi = lo.clip(0), mid.clip(0), hi.clip(0)
    lo = np.minimum(lo, mid)
    hi = np.maximum(hi, mid)

    # Reliability: narrow relative interval -> confident. Cold stations get a
    # penalty because a leave-station-out check puts them ~2.7x worse than warm.
    rel_width = (hi - lo) / (mid + 10.0)
    score = 1.0 / (1.0 + rel_width)
    score = np.where(warm, score, score * 0.6)
    score = np.clip(score, 0.0, 1.0)

    sub = pd.read_csv(DATA_DIR / "submission_template.csv")
    sub["forecast_volume"] = mid
    sub["lower_90"] = lo
    sub["upper_90"] = hi
    sub["reliability_score"] = score
    sub.to_csv(out_path, index=False)
    return sub


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["train", "submit"])
    p.add_argument("--split", default="temporal", choices=["temporal", "cold", "full"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--head-hidden", type=int, default=128)
    p.add_argument("--station-dropout", type=float, default=0.5)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip the held-out interval calibration (submit mode)")
    p.add_argument("--out", default="submission_gnn.csv")
    args = p.parse_args()

    device = pick_device(args.device)
    ds, targets = load_dataset(include_targets=(args.mode == "submit"))

    common = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        hidden=args.hidden, head_hidden=args.head_hidden,
        station_dropout=args.station_dropout, seed=args.seed,
    )

    if args.mode == "train":
        tr, va, cold = make_split(ds, args.split, seed=args.seed)
        _, info = train_model(ds, tr, va, cold, device, **common)
        if len(va):
            print("\nfinal:", json.dumps({k: round(v, 3) for k, v in info["history"][-1].items()}))
        return

    # Submission: fit the interval scales on held-out data first, then retrain on
    # everything. Skipping calibration leaves cold-station coverage near 0.50.
    scale_warm = scale_cold = 1.0
    if not args.no_calibrate:
        for name, split in (("warm", "temporal"), ("cold", "cold")):
            print(f"\n[calibration] {name} split")
            tr, va, cold = make_split(ds, split, seed=args.seed)
            m, _ = train_model(ds, tr, va, cold, device, verbose=False, **common)
            p = predict(m, ds, va, device)
            s = fit_interval_scale(p, ds.y[va])
            before = metrics(ds.y[va], p)["PI_coverage"]
            after = metrics(ds.y[va], apply_interval_scale(p, s))["PI_coverage"]
            print(f"  coverage {before:.3f} -> {after:.3f} with scale x{s:.2f}")
            if name == "warm":
                scale_warm = s
            else:
                scale_cold = s

    print("\n[final] training on all labeled data")
    tr, va, cold = make_split(ds, "full")
    model, _ = train_model(ds, tr, va, cold, device, **common)

    targets["_warm"] = targets["station_key"].isin(set(np.array(ds.stations)[ds.warm_mask]))
    pred = predict_targets(model, ds, targets, device)
    sub = make_submission(pred, targets, Path(args.out), scale_warm, scale_cold)
    print(f"\nwrote {args.out} ({len(sub):,} rows) | scales warm x{scale_warm:.2f} cold x{scale_cold:.2f}")
    print("now run:  python validate_submission.py", args.out)


if __name__ == "__main__":
    main()
