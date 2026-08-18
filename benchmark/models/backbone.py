"""
`backbone_seasonal` -- the shared profile backbone, on its own.

    log1p(y) = level_pair + shape(hour,dow) + seasonal_shape(half-month) + drift + holiday

with hierarchical shrinkage (global -> axis -> station -> pair) and a half-month seasonal
kernel. Zero-history pairs get their level from direction-matched graph donors
(`PairLevelTransfer`), their hour x dow shape from a factorised transfer
(`CFShapeTransfer`) and their seasonal surface from corridor donors (`transfer_season`).

This is the reference row on the leaderboard: no learned residual, no gradient boosting,
no neural network. A new model that cannot beat this is not yet earning its complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import traffic_pipeline as T
from _common import apply_offsets, band_offsets


class SeasonalBackbone:
    name = "backbone_seasonal"

    def fit(self, ctx):
        self.bb = T.Backbone().fit(ctx.labeled, ctx.axis)
        self.known = set(self.bb.level.index)
        self.G, self.net, self.statics = ctx.graph, ctx.network, ctx.statics
        self.lab = ctx.labeled
        self._cold_ready = set()

        pred = self.bb.predict(ctx.labeled)
        self.offs = band_offsets(ctx.labeled.ly.to_numpy(np.float64) - pred, pred)
        return self

    def _prepare_cold(self, pairs: np.ndarray) -> None:
        """Transfer level, shape and season onto pairs the fit never saw."""
        new = [p for p in pd.unique(pairs) if p not in self._cold_ready]
        if not new:
            return
        self._cold_ready.update(new)
        lvl = T.PairLevelTransfer(self.G, self.net).fit(self.bb).transfer(new)
        shp = T.CFShapeTransfer().fit(self.lab, self.statics, self.G).transfer(
            new, self.statics, self.G)
        self.bb.transfer_season(new, self.G)
        self._lvl = pd.concat([getattr(self, "_lvl", pd.Series(dtype=float)), lvl])
        self._shp = pd.concat([getattr(self, "_shp", pd.Series(dtype=float)), shp])

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        cold = ~rows.pair.isin(self.known).to_numpy()
        if cold.any():
            self._prepare_cold(rows.pair.to_numpy()[cold])
            pred = self.bb.predict(rows, level_override=self._lvl, shape_override=self._shp)
        else:
            pred = self.bb.predict(rows)
        return apply_offsets(pred, self.offs)


def build():
    return SeasonalBackbone()
