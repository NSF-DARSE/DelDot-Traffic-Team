"""
`team_v1_reimpl` -- the multiplicative model from `src/forecast_model.py`, re-expressed
against the harness contract so it can be scored on the same folds as everything else.

    forecast = station_baseline
             x hour_factor(hour, is_weekend)
             x dow_factor
             x month_factor
             x station_hour_adj  x  station_month_adj      (shrunk, n/500 capped at 1)
             x trend_factor                                (2025 rows only)

Cold-start stations: Ridge on the static road attributes, blended 40/60 with a
weighted neighbour average over `network_edges.csv` (corridor x2, <=1mi x3, 1-5mi x2),
then split across directions with a +/-2.5% asymmetry.

WHY THIS IS A RE-IMPLEMENTATION AND NOT AN IMPORT
-------------------------------------------------
`src/forecast_model.py` is a script: it reads the parquet files itself, fits the temporal
profiles on `train` *and* `val` together, and derives the year-over-year trend by comparing
train to val. The harness has to control exactly which rows a model may see -- that is the
whole point of hiding stations -- so the method is restated here as a `fit`/`predict` pair
that only ever touches `ctx.labeled`.

Two consequences worth stating plainly, because they are the honest caveats on this row:

  * The original learns hours 0-1 from the validation half-year (they are absent from
    2024). A fold that fits on 2024 alone cannot do that, so hours 0-1 fall back to the
    hour-2 factor. On the protocol arm this costs the row something real; on the
    deployment arm, which holds Jan-Apr 2025, it does not apply.
  * The original's published numbers (MAE 121 / ~108, R2 0.938, coverage 89.2%) come from
    fitting on train and scoring on validation with every station visible. That is neither
    of this harness's arms, so those numbers and these are not comparable. This row is not
    a restatement of their result; it is the same *method*, measured on the shared folds.

If this misrepresents the method, replace it -- the intent is a fair row, not a verdict.
The interval and reliability layers from `src/uncertainty.py` are not reproduced here;
intervals come from the shared residual-band helper so the comparison stays on the point
forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from _common import apply_offsets, band_offsets

SHRINKAGE_THRESHOLD = 500
FEATURE_COLS = ["through_lanes", "speed_limit", "lane_width"]
CAT_COLS = ["aadt_band", "functional_class", "roadway_width_band", "surface_type"]


def _neighbour_map(edges: pd.DataFrame) -> dict:
    nb: dict[str, list[tuple[str, float]]] = {}
    for r in edges.itertuples():
        w = 1.0
        if r.edge_type == "same_corridor":
            w *= 2.0
        if r.distance_band == "<=1mi":
            w *= 3.0
        elif r.distance_band == "1-5mi":
            w *= 2.0
        nb.setdefault(r.from_station_key, []).append((r.to_station_key, w))
        nb.setdefault(r.to_station_key, []).append((r.from_station_key, w))
    return nb


class TeamV1:
    name = "team_v1_reimpl"

    def fit(self, ctx):
        lab = ctx.labeled
        net = ctx.network
        vol = lab.volume.to_numpy(np.float64)
        self.global_mean = float(vol.mean())

        # --- Layer 1: baselines per station x direction --------------------------------
        self.baselines = lab.groupby(["station_key", "direction_code"]).volume.mean().to_dict()
        seen_stations = set(lab.station_key.unique())

        stn_mean = lab.groupby("station_key").volume.mean()
        known = net[net.station_key.isin(seen_stations)].copy()
        known = known.merge(stn_mean.rename("mean_volume"), on="station_key", how="inner")
        for c in CAT_COLS:
            known[c] = known[c].fillna("Unknown")
        Xk = pd.get_dummies(known[FEATURE_COLS + CAT_COLS], columns=CAT_COLS)
        Xk = Xk.fillna(Xk.median(numeric_only=True)).fillna(0.0)
        self.ridge = Ridge(alpha=10.0).fit(Xk.to_numpy(np.float64), known.mean_volume.to_numpy())
        self.ridge_cols = list(Xk.columns)

        cold = net[~net.station_key.isin(seen_stations)].copy()
        self.cold_baseline = {}
        if len(cold):
            for c in CAT_COLS:
                cold[c] = cold[c].fillna("Unknown")
            Xc = pd.get_dummies(cold[FEATURE_COLS + CAT_COLS], columns=CAT_COLS)
            for c in self.ridge_cols:
                if c not in Xc.columns:
                    Xc[c] = 0
            Xc = Xc[self.ridge_cols].fillna(0.0)
            reg = np.maximum(self.ridge.predict(Xc.to_numpy(np.float64)), 10.0)
            nbmap = _neighbour_map(ctx.edges)
            for i, stn in enumerate(cold.station_key.to_numpy()):
                vols, wts = [], []
                for nb, w in nbmap.get(stn, []):
                    if nb in stn_mean.index:
                        vols.append(float(stn_mean[nb])); wts.append(w)
                blended = 0.4 * reg[i] + 0.6 * np.average(vols, weights=wts) if vols else reg[i]
                self.cold_baseline[stn] = float(blended)

        # --- Layer 2: multiplicative temporal factors ----------------------------------
        wknd = lab.is_weekend.to_numpy().astype(bool)
        hw = pd.Series(vol).groupby([lab.hour.to_numpy(), wknd]).mean() / self.global_mean
        self.hour_factors = {(int(h), bool(w)): float(v) for (h, w), v in hw.items()}
        self.dow_factors = {int(d): float(v) for d, v in
                            (lab.groupby("dow").volume.mean() / self.global_mean).items()}
        self.month_factors = {int(m): float(v) for m, v in
                              (lab.groupby("month").volume.mean() / self.global_mean).items()}

        # --- Layer 2b: shrunk per-pair adjustments -------------------------------------
        self.stn_hour_adj, self.stn_month_adj = {}, {}
        for (stn, dc), g in lab.groupby(["station_key", "direction_code"]):
            n = len(g)
            w = min(n / SHRINKAGE_THRESHOLD, 1.0)
            m = float(g.volume.mean())
            if m < 1:
                continue
            gw = g.is_weekend.to_numpy().astype(bool)
            for (h, ww), v in pd.Series(g.volume.to_numpy(np.float64)).groupby(
                    [g.hour.to_numpy(), gw]).mean().items():
                exp = m * self.hour_factors.get((int(h), bool(ww)), 1.0)
                if exp > 0:
                    self.stn_hour_adj[(stn, dc, int(h), bool(ww))] = w * (v / exp) + (1 - w)
            for mo, v in g.groupby("month").volume.mean().items():
                exp = m * self.month_factors.get(int(mo), 1.0)
                if exp > 0:
                    self.stn_month_adj[(stn, dc, int(mo))] = w * (v / exp) + (1 - w)

        # --- Layer 3: year-over-year trend ---------------------------------------------
        # Matched months only, so the ratio is not a seasonal artefact.
        cur = lab[lab.year == 2025]
        prev = lab[(lab.year == 2024) & (lab.month.isin(set(cur.month.unique())))]
        if len(cur) and len(prev):
            self.global_trend = float(cur.volume.mean() / prev.volume.mean())
            a = cur.groupby("station_key").volume.mean()
            b = prev.groupby("station_key").volume.mean()
            j = pd.concat([a.rename("c"), b.rename("p")], axis=1).dropna()
            self.station_trend = (j.c / j.p).to_dict()
        else:
            # A fold that fits on 2024 alone has no second year to compare against.
            self.global_trend, self.station_trend = 1.0, {}

        resid_log = np.log1p(vol) - np.log1p(np.clip(self._point(lab), 0, None))
        self.offs = band_offsets(resid_log, np.log1p(np.clip(self._point(lab), 0, None)))
        return self

    def _point(self, rows: pd.DataFrame) -> np.ndarray:
        stn = rows.station_key.to_numpy()
        dc = rows.direction_code.to_numpy()
        hour = rows.hour.to_numpy()
        dow = rows.dow.to_numpy()
        month = rows.month.to_numpy()
        wknd = rows.is_weekend.to_numpy().astype(bool)
        year = rows.year.to_numpy()

        out = np.empty(len(rows), np.float64)
        for i in range(len(rows)):
            s, d, h, w, m = stn[i], dc[i], int(hour[i]), bool(wknd[i]), int(month[i])
            base = self.baselines.get((s, d))
            if base is None:
                b = self.cold_baseline.get(s)
                if b is None:
                    base = self.global_mean
                else:
                    base = b * (1.025 if d in ("1", "3") else 0.975)
            hf = self.hour_factors.get((h, w))
            if hf is None:                      # hours 0-1 are absent from a 2024-only fit
                hf = self.hour_factors.get((2, w), 1.0)
            f = (base * hf
                 * self.dow_factors.get(int(dow[i]), 1.0)
                 * self.month_factors.get(m, 1.0)
                 * self.stn_hour_adj.get((s, d, h, w), 1.0)
                 * self.stn_month_adj.get((s, d, m), 1.0))
            if year[i] >= 2025:
                f *= self.station_trend.get(s, self.global_trend)
            out[i] = max(f, 0.0)
        return out

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return apply_offsets(np.log1p(np.clip(self._point(rows), 0, None)), self.offs)


def build():
    return TeamV1()
