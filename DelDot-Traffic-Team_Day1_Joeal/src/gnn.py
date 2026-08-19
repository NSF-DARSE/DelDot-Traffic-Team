"""
Graph Convolutional Network for cold-start stations (numpy).
============================================================
Cold-start stations have no history, so we cannot learn their hourly pattern
directly. But they sit in a road network: `network_edges.csv` connects them to
corridor and proximity neighbours. A 2-layer GCN learns to predict a station's
LEVEL (log mean volume) from its roadway features propagated across that graph,
trained on the stations that DO have history. The hourly SHAPE is then taken
from the station's functional-class profile. Level (GNN) x shape (class) = the
cold-start hourly forecast.

The GCN is message passing: H' = A_norm @ H @ W, with the symmetric
degree-normalised adjacency A_norm = D^-1/2 (A+I) D^-1/2. Backprop is derived by
hand and checked numerically in `_grad_check()`.

A note on `edge_type` / `distance_band`, which this file deliberately does NOT use.
Weighting edges by them (same_corridor > proximity, near > far) is the obvious
next idea and it was measured on the 15-station holdout: it is much worse, WAPE
0.55 -> 9.5, because the predicted levels diverge (mean level 874k against a true
~700). Down-weighting neighbours raises the self-loop's share of the normalised
adjacency, and since `augment_with_levels` feeds each node its own level, the GCN
starts reading its own answer instead of interpolating from neighbours -- fine on
the training nodes, useless on a cold node whose level input is zero. Making that
information pay off needs the level channel separated from the self-loop, not just
a weight on the edge. Left unweighted until that is done.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_AADT = {"<5k": 0, "5k-10k": 1, "10k-25k": 2, "25k-50k": 3, "50k-100k": 4, "100k+": 5}
_FC = {"Local": 0, "Minor Collector": 1, "Major Collector": 2, "Minor Arterial": 3,
       "Other Principal Arterials": 4, "Other Expressways & Freeway": 5, "Interstate": 6}
_W = {"<24": 0, "24-35": 1, "36-47": 2, "48-59": 3, "60+": 4}

# ----------------------------------------------------------------- graph build
def build_graph(net, edges):
    """Unweighted symmetric adjacency with self-loops, degree-normalised.
    See the module docstring for why edge_type/distance_band are not used as weights."""
    stations = net["station_key"].astype(str).tolist()
    idx = {s: i for i, s in enumerate(stations)}
    n = len(stations)
    A = np.zeros((n, n), float)
    for f, t in zip(edges["from_station_key"].astype(str), edges["to_station_key"].astype(str)):
        if f in idx and t in idx and f != t:
            A[idx[f], idx[t]] = 1.0
            A[idx[t], idx[f]] = 1.0
    A += np.eye(n)                                   # self-loops
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    A_norm = A * dinv[:, None] * dinv[None, :]       # D^-1/2 (A+I) D^-1/2
    return stations, idx, A_norm


def node_features(net, stations, idx):
    """Roadway attributes per node, standardised.

    Deliberately the six original columns. Adding surface_type, a blank-class flag
    and an aadt-based class imputation was measured on the holdout and made the
    cold-start worse (WAPE 0.55 -> 0.65): 60-odd training nodes cannot support the
    extra width. The aadt fallback still earns its keep in `peer_groups`, where it
    picks the hourly shape rather than feeding the network.
    """
    s = (net.assign(station_key=net["station_key"].astype(str))
            .drop_duplicates(subset="station_key", keep="first")
            .set_index("station_key")
            .reindex(stations))

    def num(col):
        return pd.to_numeric(s[col], errors="coerce").to_numpy(float)

    F = np.column_stack([
        num("through_lanes"),
        num("speed_limit"),
        s["functional_class"].map(_FC).fillna(3).astype(float).to_numpy(),
        s["aadt_band"].map(_AADT).fillna(2).astype(float).to_numpy(),
        s["roadway_width_band"].map(_W).fillna(2).astype(float).to_numpy(),
        num("lane_width"),
    ])
    # standardise columns (nan-safe)
    mu = np.nanmean(F, 0); sd = np.nanstd(F, 0) + 1e-9
    F = np.nan_to_num((F - mu) / sd, nan=0.0)
    return F


# --------------------------------------------------------------------- the GCN
class GCN:
    def __init__(self, in_dim, hid=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim), (in_dim, hid))
        self.W2 = rng.normal(0, np.sqrt(2 / hid), (hid, 1))

    def forward(self, A, X):
        self.AX = A @ X
        self.Z1 = self.AX @ self.W1
        self.A1 = np.maximum(self.Z1, 0.0)
        self.AA1 = A @ self.A1
        self.Z2 = self.AA1 @ self.W2
        return self.Z2[:, 0]

    def backward(self, A, X, y, mask):
        n = mask.sum()
        pred = self.Z2[:, 0]
        g = np.zeros_like(pred)
        g[mask] = (2.0 / n) * (pred[mask] - y[mask])
        dZ2 = g[:, None]                              # (N,1)
        dW2 = self.AA1.T @ dZ2
        dAA1 = dZ2 @ self.W2.T
        dA1 = A @ dAA1                                # A symmetric
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = self.AX.T @ dZ1
        return dW1, dW2

    def params(self):
        return [self.W1, self.W2]

    def loss(self, A, X, y, mask):
        pred = self.forward(A, X)
        return float(np.mean((pred[mask] - y[mask]) ** 2))


def _grad_check():
    rng = np.random.default_rng(1)
    n, f = 8, 4
    A = rng.normal(size=(n, n)); A = (A + A.T) / 2
    X = rng.normal(size=(n, f)); y = rng.normal(size=n)
    mask = np.array([True, True, True, False, True, False, True, True])
    g = GCN(f, 5, seed=2)
    g.forward(A, X)
    dW1, dW2 = g.backward(A, X, y, mask)
    eps = 1e-5
    num = np.zeros_like(g.W1)
    for i in range(g.W1.shape[0]):
        for j in range(g.W1.shape[1]):
            g.W1[i, j] += eps; lp = g.loss(A, X, y, mask)
            g.W1[i, j] -= 2 * eps; lm = g.loss(A, X, y, mask)
            g.W1[i, j] += eps; num[i, j] = (lp - lm) / (2 * eps)
    rel = np.abs(num - dW1).max() / (np.abs(num).max() + 1e-9)
    return rel


def train_gcn(A, X, y, mask, hid=16, epochs=400, lr=0.03, seed=0):
    g = GCN(X.shape[1], hid, seed)
    mW1 = np.zeros_like(g.W1); vW1 = np.zeros_like(g.W1)
    mW2 = np.zeros_like(g.W2); vW2 = np.zeros_like(g.W2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        g.forward(A, X)
        dW1, dW2 = g.backward(A, X, y, mask)
        for P, dP, m, v in [(g.W1, dW1, mW1, vW1), (g.W2, dW2, mW2, vW2)]:
            m[:] = b1 * m + (1 - b1) * dP
            v[:] = b2 * v + (1 - b2) * dP ** 2
            mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
            P -= lr * mhat / (np.sqrt(vhat) + eps)
    return g


# ------------------------------------------------------ class shapes + assemble
def augment_with_levels(X_base, y_log, in_training):
    """
    Append two columns so the graph can PROPAGATE known levels to cold nodes:
    a masked log-level (0 where unknown/held-out) and a 0/1 'is-known' flag.
    Message passing then carries neighbours' known levels into the prediction.
    """
    lvl = np.where(in_training, np.nan_to_num(y_log, nan=0.0), 0.0)
    flag = in_training.astype(float)
    return np.concatenate([X_base, lvl[:, None], flag[:, None]], axis=1)


def peer_groups(net):
    """station_key -> peer-group label used for both hourly shape and level prior.

    functional_class is the natural grouping but it is blank for a large minority
    of stations. Those used to fall straight through to the single global profile,
    which is badly wrong for e.g. a blank-class 100k+ freeway. aadt_band is always
    populated, so it is the fallback rung.
    """
    s = net.assign(station_key=net["station_key"].astype(str)) \
           .drop_duplicates(subset="station_key", keep="first")
    fc = s["functional_class"] if "functional_class" in s.columns else pd.Series(index=s.index, dtype=object)
    aadt = s["aadt_band"] if "aadt_band" in s.columns else pd.Series(index=s.index, dtype=object)
    label = fc.where(fc.notna() & (fc.astype(str).str.strip() != ""),
                     "AADT:" + aadt.fillna("unknown").astype(str))
    return dict(zip(s["station_key"], label))


def _how(df):
    if "how" in df.columns:
        return df["how"]
    if {"day_of_week", "hour"} <= set(df.columns):
        return df["day_of_week"] * 24 + df["hour"]
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    return ts.dt.dayofweek * 24 + ts.dt.hour


def class_shapes(history, net):
    """Normalised 168-hour profile per peer group (mean 1.0), plus a global fallback."""
    groups = peer_groups(net)
    h = history.copy()
    h["station_key"] = h["station_key"].astype(str)
    h["peer"] = h["station_key"].map(groups)
    h["how"] = _how(h)
    shapes = {}
    glob = h.groupby("how")["volume"].mean().reindex(range(168))
    glob = (glob / glob.mean()).fillna(1.0).to_numpy() if glob.notna().any() else np.ones(168)
    for peer, grp in h.groupby("peer"):
        prof = grp.groupby("how")["volume"].mean().reindex(range(168))
        if not prof.notna().any():
            continue
        prof = prof.fillna(prof.mean())
        m = prof.mean()
        shapes[peer] = (prof / m).to_numpy() if m else glob
    shapes["__global__"] = glob
    return shapes


def station_levels(history):
    h = history.copy()
    h["station_key"] = h["station_key"].astype(str)
    return h.groupby("station_key")["volume"].mean()


def class_levels(history, net):
    """Median mean-volume per peer group: a robust level prior for cold nodes."""
    groups = peer_groups(net)
    lv = station_levels(history).rename("lvl").reset_index()
    lv["peer"] = lv["station_key"].map(groups)
    by_peer = lv.groupby("peer")["lvl"].median().to_dict()
    by_peer["__global__"] = float(lv["lvl"].median()) if len(lv) else 1.0
    return by_peer


def blend_levels(gnn_level_map, class_level, net, weight=0.5):
    """Geometric blend of the GNN level with the peer-median prior (shrinkage)."""
    groups = peer_groups(net)
    fallback = class_level.get("__global__", 1.0)
    out = {}
    for stn, g in gnn_level_map.items():
        prior = class_level.get(groups.get(str(stn)), fallback)
        if not np.isfinite(prior):
            prior = fallback
        out[stn] = float(np.exp(weight * np.log(max(g, 1)) + (1 - weight) * np.log(max(prior, 1))))
    return out
