"""
Inductive spatio-temporal kriging for the cold-interp slice (IGNNK-style).

The gap this fills
------------------
`gnn_v2` learns one hidden vector per node from *static* features, so its output
for a cold station at 14:00 on 2024-03-12 is identical whether the network is
having an ordinary Tuesday or a snowstorm. It has no channel through which a
neighbour's *observed value at that hour* can enter.

Cold-interp is exactly the task where that channel is the whole point: at those
timestamps ~68 of the 75 labelled stations are recording, and residual
correlation is 0.844 along corridors, 0.617 for proximity.

So here the node features are time-varying:

    node feature at time t = [ static attributes,
                               observed residual at t   (0 where unobserved),
                               observed mask            (1 observed / 0 not) ]

and the batch dimension is *timestamps*, not rows. Message passing then carries
live observations from recording stations into the unobserved ones.

Training uses whole-station masking, per IGNNK: each step hides a random subset
of stations entirely and reconstructs them from the rest. Masking whole stations
rather than random cells matters -- random cells would let the model see the
held-out station's own behaviour at adjacent hours, which is not the cold-start
situation and would flatter the result.

The target is the *residual* after the profile backbone, so the network only has
to learn the spatially-correlated shock, not the level and shape as well.

    ~/.virtualenvs/data_science/bin/python gnn_krige.py --epochs 120
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import traffic_pipeline as T
import run_pipeline as R
from gnn_baseline import pick_device
from gnn_v2 import RELATIONS, build_pair_graph, laplacian_pe


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class KrigeRGCN(nn.Module):
    """Relational GCN over [B, N, F] -- B timestamps, N station-direction nodes."""

    def __init__(self, in_dim, hidden, n_rel, layers=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "self": nn.Linear(d, hidden),
                "rel": nn.ModuleList([nn.Linear(d, hidden, bias=False) for _ in range(n_rel)]),
                "norm": nn.LayerNorm(hidden),
            }))
            d = hidden
        self.dropout = dropout

    def forward(self, h, adjs):
        # h: [B, N, F]; adjs: list of [N, N]
        for i, blk in enumerate(self.blocks):
            out = blk["self"](h)
            for a, lin in zip(adjs, blk["rel"]):
                out = out + torch.einsum("nm,bmd->bnd", a, lin(h))
            new = F.gelu(blk["norm"](out))
            h = new if i == 0 else h + new
            h = F.dropout(h, self.dropout, self.training)
        return h


class KrigeNet(nn.Module):
    def __init__(self, n_static, n_rel, hidden=128, dropout=0.1, layers=3, n_q=3):
        super().__init__()
        # +2 channels: the observed residual, and the observed/missing mask
        self.gnn = KrigeRGCN(n_static + 2, hidden, n_rel, layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
        )
        self.out_median = nn.Linear(hidden // 2, 1)
        self.out_spread = nn.Linear(hidden // 2, n_q - 1)

    def forward(self, static, resid, mask, adjs):
        """static [N,S]; resid,mask [B,N] -> [B,N,3] quantiles of the residual."""
        B = resid.shape[0]
        s = static.unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([s, (resid * mask).unsqueeze(-1), mask.unsqueeze(-1)], dim=-1)
        z = self.head(self.gnn(h, adjs))
        med = self.out_median(z)
        sp = F.softplus(self.out_spread(z)) + 1e-3
        return torch.cat([med - sp[..., :1], med, med + sp[..., 1:]], dim=-1)


def pinball(pred, target, qs=T.QUANTILES):
    t = target.unsqueeze(-1)
    ls = []
    for i, q in enumerate(qs):
        e = t - pred[..., i : i + 1]
        ls.append(torch.maximum(q * e, (q - 1) * e))
    return torch.cat(ls, dim=-1)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def build_panel(lab: pd.DataFrame, backbone, pairs: list[str], idx: dict):
    """[T, N] residual matrix + observed mask, indexed by hour."""
    res = lab.ly.to_numpy() - backbone.predict(lab)
    d = pd.DataFrame({"ts": lab.timestamp.to_numpy(),
                      "node": lab.pair.map(idx).to_numpy(), "res": res})
    times = np.sort(d.ts.unique())
    tpos = {t: i for i, t in enumerate(times)}
    Rm = np.zeros((len(times), len(pairs)), np.float32)
    Mm = np.zeros((len(times), len(pairs)), np.float32)
    ti = d.ts.map(tpos).to_numpy()
    Rm[ti, d.node.to_numpy()] = d.res.astype(np.float32)
    Mm[ti, d.node.to_numpy()] = 1.0
    return Rm, Mm, times, tpos


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def train_krige(Rm, Mm, static, adjs, train_nodes, device, hidden=128, layers=3,
                epochs=120, patience=15, batch=128, lr=2e-3, mask_frac=0.3,
                seed=0, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    Rt = torch.from_numpy(Rm).to(device)
    Mt = torch.from_numpy(Mm).to(device)
    St = torch.from_numpy(static).to(device)
    At = [torch.from_numpy(a).to(device) for a in adjs]
    tn = torch.tensor(sorted(train_nodes), device=device)

    n_t = Rm.shape[0]
    cut = int(n_t * 0.9)                       # temporal inner split for early stop
    tr_t = np.arange(cut)
    va_t = np.arange(cut, n_t)

    model = KrigeNet(static.shape[1], len(adjs), hidden=hidden, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    if verbose:
        print(f"[krige] {sum(p.numel() for p in model.parameters()):,} params | "
              f"{Rm.shape[0]} timestamps x {Rm.shape[1]} nodes | {len(train_nodes)} trainable nodes")

    best, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(tr_t)
        tot, nb, t0 = 0.0, 0, time.time()
        for k in range(0, len(perm) - batch + 1, batch):
            bt = torch.from_numpy(perm[k : k + batch]).to(device)
            r, m = Rt[bt], Mt[bt]
            # hide whole stations: the cold-start situation, not random cells
            hide = tn[torch.rand(len(tn), device=device) < mask_frac]
            if len(hide) == 0:
                continue
            m_in = m.clone(); m_in[:, hide] = 0.0
            pred = model(St, r, m_in, At)
            # score only where the station was hidden AND actually observed
            tgt_mask = torch.zeros_like(m); tgt_mask[:, hide] = 1.0
            tgt_mask = tgt_mask * m
            if tgt_mask.sum() == 0:
                continue
            loss = (pinball(pred, r).mean(-1) * tgt_mask).sum() / tgt_mask.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            vt = torch.from_numpy(va_t).to(device)
            r, m = Rt[vt], Mt[vt]
            g = torch.Generator(device="cpu").manual_seed(ep)
            hide = tn[torch.rand(len(tn), generator=g).to(device) < mask_frac]
            m_in = m.clone(); m_in[:, hide] = 0.0
            pred = model(St, r, m_in, At)
            tm = torch.zeros_like(m); tm[:, hide] = 1.0; tm = tm * m
            vl = float(((pinball(pred, r).mean(-1) * tm).sum() / tm.clamp(min=1e-6).sum()).item())
        sched.step(vl)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  ep {ep:3d} | train {tot/max(nb,1):.4f} | val {vl:.4f} | best {best:.4f} | {time.time()-t0:.1f}s")
        if bad >= patience:
            if verbose:
                print(f"  early stop at epoch {ep}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def krige_predict(model, Rm, Mm, static, adjs, hide_nodes, device, batch=256):
    """Predict residuals for `hide_nodes`, which are masked out of the input."""
    model.eval()
    St = torch.from_numpy(static).to(device)
    At = [torch.from_numpy(a).to(device) for a in adjs]
    hide = torch.tensor(sorted(hide_nodes), device=device)
    out = np.zeros((Rm.shape[0], Rm.shape[1], 3), np.float32)
    for k in range(0, Rm.shape[0], batch):
        r = torch.from_numpy(Rm[k : k + batch]).to(device)
        m = torch.from_numpy(Mm[k : k + batch]).to(device).clone()
        m[:, hide] = 0.0                      # the cold stations are never observed
        out[k : k + batch] = model(St, r, m, At).cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# evaluation on the protocol's cold-interp task
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--mask-frac", type=float, default=0.3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import validate_protocol as V
    D = T.load_all()
    lab, net = D["labeled"], D["network"]
    G = T.build_graph(D["edges"], net)
    axis = T.station_axis(lab, D["targets"])
    statics = R.static_table(net, G)

    rep, folds = V.make_splits(net, G, D["warm"], a.seed, 15, 4)
    pool = set().union(*folds)
    unseen = folds[a.fold]
    seen = pool - unseen

    HE = V.HIST_END
    hist_seen = lab[(lab.timestamp < HE) & (lab.station_key.isin(seen))]
    hist_unseen = lab[(lab.timestamp < HE) & (lab.station_key.isin(unseen))]

    # backbone + cold-start level/shape, fitted on the seen stations only
    bb = T.Backbone().fit(hist_seen, axis)
    dm = T.DonorModel(G, net).fit(bb, hist_seen)
    dinfo = dm.station_levels(sorted(set(net.station_key)))
    lm, _ = R.fit_level_model(hist_seen, net, G, axis, statics, seed=a.seed)
    slev = R.predict_levels(lm, dinfo, statics, sorted(set(net.station_key)))
    gst = T.GraphShapeTransfer(G).fit(hist_seen)
    lvl = pd.Series({q: slev.get(q.split("|")[0], np.nan) for q in hist_unseen.pair.unique()})
    shp = gst.transfer(hist_unseen.pair.unique())

    pairs = sorted(set(lab.pair) | set(D["targets"].pair))
    adjs, idx = build_pair_graph(net, D["edges"], pairs)
    pe = laplacian_pe(adjs, k=8)
    st = statics.reindex([p.split("|")[0] for p in pairs]).to_numpy(np.float32)
    st = np.nan_to_num(st, nan=0.0)
    st = (st - st.mean(0)) / (st.std(0) + 1e-6)
    dc = np.array([p.split("|")[1] for p in pairs])
    oh = np.stack([(dc == c).astype(np.float32) for c in ["1", "3", "5", "7"]], 1)
    static = np.concatenate([st, oh, pe], 1).astype(np.float32)

    # panel over 2024, seen stations only -- the unseen ones are never in the input
    Rm, Mm, times, tpos = build_panel(hist_seen, bb, pairs, idx)
    train_nodes = {idx[p] for p in hist_seen.pair.unique()}
    hide_nodes = {idx[p] for p in hist_unseen.pair.unique() if p in idx}

    dev = pick_device("auto")
    model, best = train_krige(Rm, Mm, static, adjs, train_nodes, dev,
                              hidden=a.hidden, layers=a.layers, epochs=a.epochs,
                              mask_frac=a.mask_frac, seed=a.seed)

    pred = krige_predict(model, Rm, Mm, static, adjs, hide_nodes, dev)

    # map the held-out rows onto the [T, N] grid and add the backbone back
    base = bb.predict(hist_unseen, level_override=lvl, shape_override=shp)
    ti = pd.Series(hist_unseen.timestamp.to_numpy()).map(tpos).to_numpy()
    ni = hist_unseen.pair.map(idx).to_numpy()
    ok = ~pd.isna(ti)
    ti = ti[ok].astype(int); ni = ni[ok]
    resid_q = pred[ti, ni]                       # [n, 3]
    full = base[ok][:, None] + resid_q
    y = hist_unseen.volume.to_numpy()[ok]
    yl = hist_unseen.ly.to_numpy()[ok]

    def geh5(mid):
        g = np.sqrt(2 * (mid - y) ** 2 / np.maximum(mid + y, 1e-9))
        return 100 * float((g < 5).mean())

    base_v = np.expm1(base[ok]).clip(0)
    krig_v = np.expm1(full[:, 1]).clip(0)
    print("\nCOLD-INTERP  (held-out stations, 2024, neighbours observed at time t)")
    print(f"  backbone only (donor level x graph shape)  GEH<5 {geh5(base_v):5.1f}%  MAE {np.abs(y-base_v).mean():7.1f}")
    print(f"  + kriging GNN (this model)                 GEH<5 {geh5(krig_v):5.1f}%  MAE {np.abs(y-krig_v).mean():7.1f}")
    print(f"  n = {len(y):,}   |   best val pinball {best:.4f}")
    print(json.dumps({k: round(v, 3) for k, v in T.metrics(yl, full.astype(np.float64)).items()}))


if __name__ == "__main__":
    main()
