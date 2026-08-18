"""
Cold-Start Enhancement — Synthetic History from Neighbouring Stations
======================================================================
For stations with no observed history, reconstruct an estimated hourly history
from connected known neighbours, then fit station-specific temporal profiles on
that synthetic history.

Permitted by the evaluation guidance:
    "For stations with limited or no observed history, you may develop
     model-generated estimates or representations of their historical behaviour
     using the supplied neighbouring stations, network relationships, and other
     permitted data. Model-generated history should be treated as an estimate
     rather than observed truth."

Method
------
1. Weighted neighbour profile
   For each cold-start station, take a weighted average of hourly observations
   from its connected known neighbours over the full observed window
   (training + validation). Weights:
       edge_type: same_corridor x2, proximity x1
       distance:  <=1mi x3, 1-5mi x2, 5-15mi x1

2. Volume-level scaling
   Scale the neighbour profile to the station's own expected volume level:
       blended_mean = 0.4 * ridge_feature_regression + 0.6 * neighbour_average

3. Station-specific profiles
   From the synthetic history derive hour x weekend, day-of-week and month
   factors relative to that station's own mean.

4. Trend
   Weighted average of the neighbours' observed year-over-year trends.

5. Fallback
   Stations with no connected known neighbours get the feature-regression
   baseline and the global temporal profiles (flagged as such downstream via
   `prediction_path = cold_start_fallback`).

Output: output/cold_start_enhanced.json

Usage: python src/cold_start_enhance.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from forecast_model import load_and_align_data, _build_neighbor_map

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_ALPHA = 10.0
FEATURE_BLEND = 0.4          # weight on feature regression vs neighbour average
NUMERIC_FEATURES = ['through_lanes', 'speed_limit', 'lane_width']
CATEGORICAL_FEATURES = ['aadt_band', 'functional_class',
                        'roadway_width_band', 'surface_type']


def fit_feature_regression(train, network):
    """Ridge regression: road features -> mean volume, fitted on known stations."""
    known_means = train.groupby('station_key')['volume'].mean().reset_index()
    known_means.columns = ['station_key', 'mean_volume']
    df = known_means.merge(network, on='station_key')
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('Unknown')

    X = pd.get_dummies(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
                       columns=CATEGORICAL_FEATURES)
    y = df['mean_volume'].values
    model = Ridge(alpha=RIDGE_ALPHA).fit(X, y)
    print(f"  Feature regression R² (in-sample): {model.score(X, y):.3f}")
    return model, X.columns


def predict_cold_baselines(model, feature_cols, network, cold_stations):
    """Predict a mean volume level for each cold-start station from road features."""
    cold = network[network['station_key'].isin(cold_stations)].copy()
    for col in CATEGORICAL_FEATURES:
        cold[col] = cold[col].fillna('Unknown')

    X = pd.get_dummies(cold[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
                       columns=CATEGORICAL_FEATURES)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    preds = np.maximum(model.predict(X), 10.0)
    return dict(zip(cold['station_key'], preds))


def build_full_history(train, val):
    """Concatenate observed training and validation history on a common schema."""
    cols = ['station_key', 'direction_code', 'timestamp', 'volume',
            'hour', 'day_of_week', 'month', 'is_weekend']
    tr = train.copy()
    tr['hour'] = tr['actual_hour']          # identity when HOUR_OFFSET == 0
    tr = tr[cols].copy()
    va = val[cols].copy()
    hist = pd.concat([tr, va], ignore_index=True)
    hist['year'] = hist['timestamp'].dt.year
    return hist


def main():
    print("=" * 68)
    print("  COLD-START ENHANCEMENT")
    print("=" * 68)

    train, val, network, edges, submission = load_and_align_data()

    train_stations = set(train['station_key'].unique())
    sub_stations = set(submission['station_key'].unique())
    cold_stations = sorted(sub_stations - train_stations)
    print(f"\nCold-start stations in submission: {len(cold_stations)}")

    neighbor_map = _build_neighbor_map(edges)
    cold_neighbors = {
        s: [(n, w) for n, w in neighbor_map.get(s, []) if n in train_stations]
        for s in cold_stations
    }
    connected = sum(1 for v in cold_neighbors.values() if v)
    print(f"  with known neighbours: {connected}/{len(cold_stations)}")

    print("\n--- Feature regression ---")
    reg, feature_cols = fit_feature_regression(train, network)
    feature_means = predict_cold_baselines(reg, feature_cols, network, cold_stations)

    print("\n--- Synthetic history + station profiles ---")
    history = build_full_history(train, val)
    print(f"  observed history rows: {len(history):,}")
    print(f"  window: {history['timestamp'].min()} .. {history['timestamp'].max()}")

    known_station_means = train.groupby('station_key')['volume'].mean()

    cold_baselines, cold_hour, cold_month, cold_dow, cold_trends = {}, {}, {}, {}, {}
    n_enhanced = n_fallback = 0

    for stn in cold_stations:
        neighbours = cold_neighbors[stn]
        feature_mean = feature_means.get(stn, 500.0)

        if not neighbours:
            # Fallback: feature regression + global profiles (handled downstream)
            n_fallback += 1
            cold_trends[stn] = None       # signals "use global trend"
            continue

        names = [n for n, _ in neighbours]
        weights = {n: w for n, w in neighbours}

        nb = history[history['station_key'].isin(names)].copy()
        if nb.empty:
            n_fallback += 1
            cold_trends[stn] = None
            continue
        nb['weight'] = nb['station_key'].map(weights)

        profile = (nb.groupby(['hour', 'day_of_week', 'month', 'is_weekend'])
                     .apply(lambda g: np.average(g['volume'], weights=g['weight']),
                            include_groups=False)
                     .reset_index(name='synthetic_volume'))

        neighbour_avg = np.average([known_station_means.get(n, 500.0) for n, _ in neighbours],
                                   weights=[w for _, w in neighbours])
        blended_mean = FEATURE_BLEND * feature_mean + (1 - FEATURE_BLEND) * neighbour_avg
        scale = blended_mean / max(neighbour_avg, 1.0)
        profile['synthetic_volume'] *= scale

        stn_mean = profile['synthetic_volume'].mean()
        if stn_mean < 1:
            n_fallback += 1
            cold_trends[stn] = None
            continue

        # Station-specific multiplicative factors, replicated across directions
        for (h, w), vol in profile.groupby(['hour', 'is_weekend'])['synthetic_volume'].mean().items():
            for d in ('1', '3', '5', '7'):
                cold_hour[(stn, d, int(h), bool(w))] = vol / stn_mean
        for m, vol in profile.groupby('month')['synthetic_volume'].mean().items():
            for d in ('1', '3', '5', '7'):
                cold_month[(stn, d, int(m))] = vol / stn_mean
        for dw, vol in profile.groupby('day_of_week')['synthetic_volume'].mean().items():
            for d in ('1', '3', '5', '7'):
                cold_dow[(stn, d, int(dw))] = vol / stn_mean
        for d in ('1', '3', '5', '7'):
            asym = 1.025 if d in ('1', '3') else 0.975
            cold_baselines[(stn, d)] = stn_mean * asym

        # Trend inherited from neighbours (2025 H1 vs 2024 H1)
        trends, tw = [], []
        for n, w in neighbours:
            a = history[(history.station_key == n) & (history.year == 2024)
                        & (history.month.isin([1, 2, 3, 4, 5, 6]))]['volume'].mean()
            b = history[(history.station_key == n) & (history.year == 2025)
                        & (history.month.isin([1, 2, 3, 4, 5, 6]))]['volume'].mean()
            if a and a > 0 and not np.isnan(b):
                trends.append(b / a)
                tw.append(w)
        cold_trends[stn] = float(np.average(trends, weights=tw)) if trends else None
        n_enhanced += 1

    # Resolve fallback trends to the global value
    _, global_trend = _global_trend(train, val)
    cold_trends = {k: (v if v is not None else global_trend) for k, v in cold_trends.items()}

    print(f"  enhanced (neighbour-based): {n_enhanced}")
    print(f"  fallback (feature-only):    {n_fallback}")
    vals = list(cold_trends.values())
    print(f"  trends: mean={np.mean(vals):.4f} range=[{min(vals):.3f}, {max(vals):.3f}]")

    artifact = {
        'cold_baselines': {str(k): v for k, v in cold_baselines.items()},
        'cold_hour_factors': {str(k): v for k, v in cold_hour.items()},
        'cold_month_factors': {str(k): v for k, v in cold_month.items()},
        'cold_dow_factors': {str(k): v for k, v in cold_dow.items()},
        'cold_trends': cold_trends,
    }
    out = OUTPUT_DIR / 'cold_start_enhanced.json'
    out.write_text(json.dumps(artifact))
    print(f"\n  wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    print(f"    baselines={len(cold_baselines)} hour={len(cold_hour)} "
          f"month={len(cold_month)} dow={len(cold_dow)}")
    return artifact


def _global_trend(train, val):
    """Local helper mirroring forecast_model.compute_trend's global value."""
    from forecast_model import compute_trend
    return compute_trend(train, val)


if __name__ == '__main__':
    main()
