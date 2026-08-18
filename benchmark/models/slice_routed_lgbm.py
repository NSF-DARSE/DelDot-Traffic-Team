"""
`slice_routed_lgbm` -- the seasonal backbone plus a LightGBM quantile residual, routed
by slice.

Warm rows (the pair has history) get the boosted residual on top of the backbone. Cold
rows do not: the residual learner is fitted on warm rows and does not transfer -- measured
at MAE 285.7 against 277.3 for the donor baseline alone -- so its output is blended 50%
back toward the backbone rather than trusted.

This is the deployed pipeline minus the R-GCN branch. The full deployed model averages
this with `gnn_v2` (see FINDINGS.md); the GNN is left out here so the row runs in minutes
on a CPU and stays reproducible for anyone without a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import traffic_pipeline as T
from _common import apply_offsets, band_offsets

COLD_BLEND = 0.5      # weight on the learned residual for zero-history pairs
N_TREES = 120         # retuned down from 350 once the backbone became seasonal


def _features(rows: pd.DataFrame, base: np.ndarray, statics: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=rows.index)
    X["base"] = base
    X["hour"] = rows.hour.to_numpy()
    X["dow"] = rows.dow.to_numpy()
    X["month"] = rows.month.to_numpy()
    X["is_weekend"] = rows.is_weekend.to_numpy()
    X["is_holiday"] = rows.is_holiday.to_numpy()
    X["holiday_adj"] = rows.holiday_adj.to_numpy()
    X["t_years"] = rows.t_years.to_numpy()
    doy = rows.doy.to_numpy(np.float64)
    for k in (1, 2):
        X[f"doy_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"doy_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    st = statics.reindex(rows.station_key.to_numpy())
    for c in statics.columns:
        X[c] = st[c].to_numpy()
    return X.astype(np.float64).fillna(-1.0)


class SliceRoutedLGBM:
    name = "slice_routed_lgbm"

    def fit(self, ctx):
        self.bb = T.Backbone().fit(ctx.labeled, ctx.axis)
        self.known = set(self.bb.level.index)
        self.G, self.net, self.statics = ctx.graph, ctx.network, ctx.statics
        self.lab = ctx.labeled
        self._cold_ready = set()
        self._lvl = pd.Series(dtype=float)
        self._shp = pd.Series(dtype=float)

        base = self.bb.predict(ctx.labeled)
        y = ctx.labeled.ly.to_numpy(np.float64) - base
        self.Xtr = _features(ctx.labeled, base, ctx.statics)
        self.ytr = y
        self.feats = list(self.Xtr.columns)
        self.offs = band_offsets(y, base)
        return self

    def _prepare_cold(self, pairs: np.ndarray) -> None:
        new = [p for p in pd.unique(pairs) if p not in self._cold_ready]
        if not new:
            return
        self._cold_ready.update(new)
        lvl = T.PairLevelTransfer(self.G, self.net).fit(self.bb).transfer(new)
        shp = T.CFShapeTransfer().fit(self.lab, self.statics, self.G).transfer(
            new, self.statics, self.G)
        self.bb.transfer_season(new, self.G)
        self._lvl = pd.concat([self._lvl, lvl])
        self._shp = pd.concat([self._shp, shp])

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        cold_mask = ~rows.pair.isin(self.known).to_numpy()
        if cold_mask.any():
            self._prepare_cold(rows.pair.to_numpy()[cold_mask])
            base = self.bb.predict(rows, level_override=self._lvl, shape_override=self._shp)
        else:
            base = self.bb.predict(rows)

        Xte = _features(rows, base, self.statics)
        q = T.lgb_quantiles(self.Xtr, self.ytr, Xte, self.feats,
                            n_estimators=N_TREES, seed=0)
        pred = np.asarray(q, dtype=np.float64) + base[:, None]

        if cold_mask.any():
            # Blend the learned residual back toward the backbone where it does not transfer.
            b = base[:, None]
            blended = COLD_BLEND * pred + (1 - COLD_BLEND) * np.concatenate(
                [pred[:, :1] - pred[:, 1:2] + b, b, pred[:, 2:] - pred[:, 1:2] + b], axis=1)
            pred = np.where(cold_mask[:, None], blended, pred)

        # The quantile heads are overconfident out of sample; widen with the fitted
        # residual bands and keep the model's own ordering.
        wide = apply_offsets(pred[:, 1], self.offs)
        out = np.column_stack([
            np.minimum(np.expm1(pred[:, 0]), wide[:, 0]),
            np.expm1(pred[:, 1]),
            np.maximum(np.expm1(pred[:, 2]), wide[:, 2]),
        ])
        return np.clip(out, 0.0, None)


def build():
    return SliceRoutedLGBM()
