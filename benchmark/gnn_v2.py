"""
GNN v2 -- relational GCN over the *station-direction* graph.

Why v2 exists
-------------
v1 put one node per station and treated direction as an embedding. That averages
the two directions of a station into a single representation, which destroys the
signal that turns out to matter most: along a corridor, North at station A is the
same physical flow as North at station B, and the commute asymmetry transfers
along that flow. Measured on held-out stations (hour x dow shape, log space):

    axis-level fallback                 0.3445
    direction_code-level                0.3360
    graph neighbours, direction-matched 0.2296     <- 33% better

So v2 uses 236 station-direction nodes and five typed relations that preserve
the direction match:

    R1 corridor,  same direction      physical flow continuation (strongest)
    R2 corridor,  opposite direction  same road, other carriageway
    R3 proximity, same direction
    R4 proximity, opposite direction
    R5 same station, opposite direction  (raw correlation 0.916)

Node features add Laplacian positional encoding, giving the network continuous
coordinates on a graph that has no supplied geometry.

Training runs to convergence with early stopping on a temporal holdout, rather
than a fixed epoch budget with an LR schedule that manufactures a flat tail.

    ~/.virtualenvs/data_science/bin/python gnn_v2.py --split temporal
    ~/.virtualenvs/data_science/bin/python gnn_v2.py --split cold
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import traffic_pipeline as T
from gnn_baseline import pick_device

RELATIONS = ["corr_same", "corr_opp", "prox_same", "prox_opp", "station_opp"]
OPPOSITE = {"1": "5", "5": "1", "3": "7", "7": "3"}


# --------------------------------------------------------------------------- #
# graph construction
# --------------------------------------------------------------------------- #

def build_pair_graph(net: pd.DataFrame, edges: pd.DataFrame, pairs: list[str]):
    """Five normalized adjacency matrices over station-direction nodes."""
    idx = {p: i for i, p in enumerate(pairs)}
    n = len(pairs)
    A = {r: np.zeros((n, n), np.float32) for r in RELATIONS}

    by_station: dict[str, list[str]] = {}
    for p in pairs:
        by_station.setdefault(p.split("|")[0], []).append(p)

    for r in edges.itertuples():
        w = T.DIST_W.get(r.distance_band, 0.7)
        corr = r.edge_type == "same_corridor"
        for pa in by_station.get(r.from_station_key, []):
            da = pa.split("|")[1]
            for pb in by_station.get(r.to_station_key, []):
                db = pb.split("|")[1]
                same = da == db
                rel = ("corr_" if corr else "prox_") + ("same" if same else "opp")
                i, j = idx[pa], idx[pb]
                A[rel][i, j] = A[rel][j, i] = w

    for st, ps in by_station.items():          # within-station opposite direction
        for pa in ps:
            for pb in ps:
                if pa != pb and OPPOSITE.get(pa.split("|")[1]) == pb.split("|")[1]:
                    A["station_opp"][idx[pa], idx[pb]] = 1.0

    out = []
    for r in RELATIONS:
        a = A[r]
        deg = a.sum(1, keepdims=True)
        out.append(a / np.clip(deg, 1e-6, None))
    return out, idx


def laplacian_pe(adjs: list[np.ndarray], k: int = 8) -> np.ndarray:
    """Laplacian eigenvector positional encoding.

    The graph ships with no coordinates. The low eigenvectors of the combined
    normalized Laplacian give a continuous embedding that respects connectivity,
    which is the closest thing to a coordinate system the rules allow us to build.
    """
    A = np.clip(sum(adjs), 0, None)
    A = (A + A.T) / 2
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.clip(d, 1e-6, None))
    L = np.eye(len(A), dtype=np.float32) - (A * dinv[:, None]) * dinv[None, :]
    w, v = np.linalg.eigh(L)
    pe = v[:, 1 : k + 1]                        # skip the trivial constant vector
    pe = pe / (np.abs(pe).max(0, keepdims=True) + 1e-9)
    return pe.astype(np.float32)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class RGCN(nn.Module):
    def __init__(self, in_dim, hidden, n_rel, layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                "self": nn.Linear(d, hidden),
                "rel": nn.ModuleList([nn.Linear(d, hidden, bias=False) for _ in range(n_rel)]),
                "norm": nn.LayerNorm(hidden),
            }))
            d = hidden
        self.dropout = dropout

    def forward(self, h, adjs):
        for i, layer in enumerate(self.layers):
            out = layer["self"](h)
            for a, lin in zip(adjs, layer["rel"]):
                out = out + a @ lin(h)
            new = F.gelu(layer["norm"](out))
            h = new if i == 0 else h + new            # residual once dims match
            h = F.dropout(h, self.dropout, self.training)
        return h


class TrafficGNNv2(nn.Module):
    def __init__(self, n_node_feat, n_time, n_nodes, n_rel, hidden=96, head=192, dropout=0.1, layers=3):
        super().__init__()
        self.gnn = RGCN(n_node_feat, hidden, n_rel, layers=layers, dropout=dropout)
        self.node_id = nn.Embedding(n_nodes, hidden)
        nn.init.normal_(self.node_id.weight, std=0.02)
        in_dim = hidden * 2 + n_time + 1
        self.head = nn.Sequential(
            nn.Linear(in_dim, head), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head, head), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head, head // 2), nn.GELU(),
        )
        self.out_median = nn.Linear(head // 2, 1)
        self.out_spread = nn.Linear(head // 2, 2)

    def forward(self, nf, adjs, id_keep, node_idx, x_time):
        z = self.gnn(nf, adjs)
        ident = self.node_id.weight * id_keep[:, None]
        emb = torch.cat([z, ident], 1)[node_idx]
        h = self.head(torch.cat([emb, x_time, id_keep[node_idx][:, None]], 1))
        med = self.out_median(h)
        sp = F.softplus(self.out_spread(h)) + 1e-3
        return torch.cat([med - sp[:, :1], med, med + sp[:, 1:]], 1)


def pinball(pred, target, qs=T.QUANTILES, weight=None):
    t = target[:, None]
    ls = []
    for i, q in enumerate(qs):
        e = t - pred[:, i : i + 1]
        ls.append(torch.maximum(q * e, (q - 1) * e))
    L = torch.cat(ls, 1)
    if weight is not None:
        L = L * weight[:, None]
    return L.mean()


def geh_weight(y_abs_log):
    """Turn a log-space loss into GEH geometry, to first order.

    GEH ~ 2|sqrt(M) - sqrt(C)|, and d(sqrt M)/d(log M) = sqrt(M)/2, so an error of delta
    in log space costs about delta * sqrt(M) in GEH. Weighting each row by sqrt(volume)
    therefore makes an L1 log-space loss proportional to GEH, without re-plumbing the
    target space or breaking the residual decomposition against `base`.

    Normalised to mean 1 so the learning rate keeps its meaning.
    """
    w = torch.sqrt(torch.expm1(y_abs_log).clamp(min=0.0) + 1.0)
    return w / w.mean()


def geh_loss(pred, target):
    """1 - fraction below GEH 5, as a *selection* score (lower is better, like a loss).

    Pinball is a poor proxy for GEH: across a capacity sweep validation pinball moved
    0.0670 -> 0.0692 (i.e. looked worse) while GEH<5 rose 68.2% -> 79.0%. Early stopping
    on pinball therefore pointed away from a 10-point improvement. This is not
    differentiable and is only used for model selection, never for the gradient.
    """
    y = torch.expm1(target)
    m = torch.expm1(pred[:, 1]).clamp(min=0)
    geh = torch.sqrt(2 * (m - y) ** 2 / torch.clamp(m + y, min=1e-9))
    return 1.0 - (geh < 5).float().mean()


# --------------------------------------------------------------------------- #
# data prep
# --------------------------------------------------------------------------- #

def time_feats(df):
    h, d, doy = df.hour.to_numpy(np.float32), df.dow.to_numpy(np.float32), df.doy.to_numpy(np.float32)
    cols = []
    for v, per, K in ((h, 24, 3), (d, 7, 2), (doy, 365.25, 2)):
        for k in range(1, K + 1):
            cols += [np.sin(2 * np.pi * k * v / per), np.cos(2 * np.pi * k * v / per)]
    cols += [df.is_weekend.to_numpy(np.float32), df.is_holiday.to_numpy(np.float32),
             df.holiday_adj.to_numpy(np.float32), df.t_years.to_numpy(np.float32)]
    # The profile backbone as an input. LightGBM has always been given `base` as a feature
    # AND fitted on the residual `ly - base`; the GNN was not, so it had to re-derive every
    # pair's level and its hour x dow x month profile inside the node-ID embedding. That is
    # the most economical explanation for the measured capacity anomaly (width helps a lot,
    # depth does not) -- the bottleneck was memorisation bandwidth, not message passing.
    if "base" in df.columns:
        cols.append(df.base.to_numpy(np.float32))
    return np.stack(cols, 1).astype(np.float32)


@torch.no_grad()
def predict_rows(model, P, rows_df, device, id_avail, batch=65536):
    """Predict arbitrary rows (e.g. submission targets) with a fitted v2 model."""
    model.eval()
    nf = torch.from_numpy(P["node_feat"]).to(device)
    adjs = [torch.from_numpy(a).to(device) for a in P["adjs"]]
    X = time_feats(rows_df)
    node = rows_df.pair.map(P["idx"]).to_numpy(np.int64)
    out = []
    for k in range(0, len(rows_df), batch):
        sl = slice(k, k + batch)
        out.append(model(nf, adjs, id_avail,
                         torch.from_numpy(node[sl]).to(device),
                         torch.from_numpy(X[sl]).to(device)).cpu().numpy())
    pred = np.concatenate(out)
    if "base" in rows_df.columns:
        pred = pred + rows_df.base.to_numpy(np.float64)[:, None]
    return pred


def prepare(D, statics_fn):
    lab, tgt, net, edges = D["labeled"], D["targets"], D["network"], D["edges"]
    pairs = sorted(set(lab.pair) | set(tgt.pair))
    adjs, idx = build_pair_graph(net, edges, pairs)
    pe = laplacian_pe(adjs, k=8)

    st = statics_fn.reindex([p.split("|")[0] for p in pairs]).to_numpy(np.float32)
    st = np.nan_to_num(st, nan=0.0)
    st = (st - st.mean(0)) / (st.std(0) + 1e-6)
    dc = np.array([p.split("|")[1] for p in pairs])
    onehot = np.stack([(dc == c).astype(np.float32) for c in ["1", "3", "5", "7"]], 1)
    node_feat = np.concatenate([st, onehot, pe], 1).astype(np.float32)

    # When `base` is supplied the GNN predicts the residual, exactly like LightGBM does.
    base = (lab.base.to_numpy(np.float32) if "base" in lab.columns
            else np.zeros(len(lab), np.float32))
    return {
        "pairs": pairs, "idx": idx, "adjs": adjs, "node_feat": node_feat,
        "X": time_feats(lab), "node": lab.pair.map(idx).to_numpy(np.int64),
        "y": (lab.ly.to_numpy(np.float32) - base), "base": base,
        "ts": lab.timestamp.to_numpy("datetime64[ns]"),
        "station": lab.station_key.to_numpy(),
    }


# --------------------------------------------------------------------------- #
# training with early stopping
# --------------------------------------------------------------------------- #

def train(P, tr_rows, va_rows, dead_nodes, device, hidden=96, head=192, lr=2e-3,
          batch=8192, max_epochs=200, patience=12, node_dropout=0.4, seed=0, verbose=True,
          layers=3, dropout=0.1, select="geh", geh_loss_weight=True):
    torch.manual_seed(seed); np.random.seed(seed)
    nf = torch.from_numpy(P["node_feat"]).to(device)
    adjs = [torch.from_numpy(a).to(device) for a in P["adjs"]]
    X = torch.from_numpy(P["X"]).to(device)
    y = torch.from_numpy(P["y"]).to(device)
    node = torch.from_numpy(P["node"]).to(device)

    avail = np.zeros(len(P["pairs"]), np.float32)
    avail[np.unique(P["node"][tr_rows])] = 1.0
    avail[list(dead_nodes)] = 0.0
    id_avail = torch.from_numpy(avail).to(device)

    model = TrafficGNNv2(P["node_feat"].shape[1], P["X"].shape[1], len(P["pairs"]),
                         len(P["adjs"]), hidden, head, dropout=dropout, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    base_t = torch.from_numpy(P["base"]).to(device)
    tr = torch.from_numpy(tr_rows).to(device)
    va = torch.from_numpy(va_rows).to(device) if len(va_rows) else None
    steps = max(1, len(tr_rows) // batch)
    best, best_state, bad, hist = np.inf, None, 0, []
    if verbose:
        print(f"[v2] {sum(p.numel() for p in model.parameters()):,} params | "
              f"{len(P['pairs'])} nodes | {len(P['adjs'])} relations | {steps} steps/ep | {device}")

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        tot, t0 = 0.0, time.time()
        for k in range(steps):
            b = perm[k * batch : (k + 1) * batch]
            keep = id_avail * (torch.rand(len(P["pairs"]), device=device) > node_dropout).float()
            w = geh_weight(y[b] + base_t[b]) if geh_loss_weight else None
            loss = pinball(model(nf, adjs, keep, node[b], X[b]), y[b], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()

        model.eval()
        with torch.no_grad():
            if va is not None:
                vp = model(nf, adjs, id_avail, node[va], X[va])
                v_pin = float(pinball(vp, y[va]).item())
                b = base_t[va][:, None]
                v_geh = float(geh_loss(vp + b, y[va] + b[:, 0]).item())
            else:
                v_pin = v_geh = tot / steps
        vl = v_geh if select == "geh" else v_pin
        sched.step(vl)
        hist.append({"epoch": ep, "train": tot / steps, "val": vl,
                     "val_pinball": v_pin, "val_geh5_%": round(100 * (1 - v_geh), 2),
                     "sec": time.time() - t0})
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"  ep {ep:3d} | train {tot/steps:.4f} | pinball {v_pin:.4f} | "
                  f"GEH<5 {100*(1-v_geh):5.1f}% | best[{select}] {best:.4f} | bad {bad} | {time.time()-t0:.1f}s")
        if bad >= patience:
            if verbose:
                print(f"  early stop at epoch {ep} (no val improvement for {patience})")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, hist, id_avail


@torch.no_grad()
def predict(model, P, rows, device, id_avail, batch=65536):
    model.eval()
    nf = torch.from_numpy(P["node_feat"]).to(device)
    adjs = [torch.from_numpy(a).to(device) for a in P["adjs"]]
    out = []
    for k in range(0, len(rows), batch):
        b = rows[k : k + batch]
        out.append(model(nf, adjs, id_avail,
                         torch.from_numpy(P["node"][b]).to(device),
                         torch.from_numpy(P["X"][b]).to(device)).cpu().numpy())
    return np.concatenate(out) + P["base"][rows].astype(np.float64)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="temporal", choices=["temporal", "cold"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--node-dropout", type=float, default=0.4)
    ap.add_argument("--head", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--select", default="geh", choices=["geh", "pinball"])
    ap.add_argument("--no-geh-weight", action="store_true",
                    help="train on unweighted log-space pinball (the old geometry)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    import run_pipeline as R
    device = pick_device(a.device)
    D = T.load_all()
    G = T.build_graph(D["edges"], D["network"])
    P = prepare(D, R.static_table(D["network"], G))

    ts = P["ts"]
    cut = np.datetime64(pd.Timestamp("2025-05-01"))
    tr = np.flatnonzero(ts < cut)
    te = np.flatnonzero(ts >= cut)
    dead: set[int] = set()

    if a.split == "cold":
        rng = np.random.default_rng(a.seed)
        warm_st = sorted(set(P["station"]))
        hold = set(rng.choice(warm_st, 18, replace=False))
        mask = np.isin(P["station"], list(hold))
        dead = set(np.unique(P["node"][mask]).tolist())
        tr, te = tr[~mask[tr]], te[mask[te]]

    # inner temporal split for early stopping -- never the test block
    inner = int(len(tr) * 0.9)
    order = np.argsort(ts[tr])
    tr_fit, tr_val = tr[order[:inner]], tr[order[inner:]]

    model, hist, id_avail = train(P, tr_fit, tr_val, dead, device, hidden=a.hidden,
                                  max_epochs=a.epochs, patience=a.patience,
                                  node_dropout=a.node_dropout, seed=a.seed,
                                  head=a.head, layers=a.layers, select=a.select,
                                  geh_loss_weight=not a.no_geh_weight)
    pred = predict(model, P, te, device, id_avail)
    m = T.metrics((P["y"][te] + P["base"][te]).astype(np.float64), pred.astype(np.float64))
    print(f"\n[{a.split}] " + json.dumps({k: round(v, 3) for k, v in m.items()}))
    print(f"epochs run: {len(hist)} | selected on '{a.select}' | "
          f"best val pinball {min(h['val_pinball'] for h in hist):.4f} | "
          f"best val GEH<5 {max(h['val_geh5_%'] for h in hist):.1f}%")


if __name__ == "__main__":
    main()
