"""
Master Pipeline: Generate Final Submission
===========================================
Orchestrates all model layers and produces the submission CSV.

Steps:
1. Build multiplicative base model (Layers 1-3)
2. Generate base predictions on full validation → get residuals
3. Train LightGBM on validation residuals (Layer 4)
4. Generate submission predictions (base + LGB correction)
5. Compute prediction intervals (90% CI)
6. Assign reliability scores
7. Validate and save submission

Author: Traffic Pattern Analysis Model
Date: 2025-08-17
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add model source to path
sys.path.insert(0, str(Path(__file__).parent))

from forecast_model import (
    load_and_align_data, compute_station_baselines, compute_temporal_profiles,
    compute_station_adjustments, compute_trend
)
from lgb_residual import prepare_lgb_features, train_lgb_residual_model, apply_lgb_correction
from uncertainty import (
    compute_prediction_intervals, generate_intervals, compute_reliability_scores
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cold_start_enhanced():
    """
    Load the synthetic-history cold-start profiles produced by
    src/cold_start_enhance.py.

    Required: cold-start stations get station-specific hour, day-of-week and
    month profiles from this artifact rather than the network-average profiles.
    Without it the submission differs materially (mean forecast 776.6 vs 798.2),
    so a missing file is a hard error rather than a silent fallback.
    """
    path = OUTPUT_DIR / 'cold_start_enhanced.json'
    if not path.exists():
        sys.exit(f"Missing {path}.\nRun first: python3 src/cold_start_enhance.py")
    with open(path) as fh:
        return json.load(fh)


def predict_base(df, components, cold_enhanced, train_stations):
    """
    Generate base multiplicative predictions for a DataFrame.

    Known stations use observed baselines, profiles and station-level
    adjustments. Cold-start stations use the synthetic-history baseline and
    station-specific profiles, falling back to the network-average factor for
    any cell the synthetic history did not cover.
    """
    baselines = components['baselines']
    hour_factors = components['hour_factors']
    dow_factors = components['dow_factors']
    month_factors = components['month_factors']
    station_hour_adj = components['station_hour_adj']
    station_month_adj = components['station_month_adj']
    per_station_trend = components['per_station_trend']
    global_trend = components['global_trend']

    cold_baselines = cold_enhanced['cold_baselines']
    cold_hour = cold_enhanced['cold_hour_factors']
    cold_dow = cold_enhanced['cold_dow_factors']
    cold_month = cold_enhanced['cold_month_factors']
    cold_trends = cold_enhanced['cold_trends']

    predictions = np.zeros(len(df))

    for i, row in enumerate(df.to_dict('records')):
        stn = row['station_key']
        d = str(row['direction_code'])
        ts = row.get('timestamp')
        h = int(row['hour']) if 'hour' in row else int(ts.hour)
        dow = int(row['day_of_week']) if 'day_of_week' in row else int(ts.dayofweek)
        m = int(row['month']) if 'month' in row else int(ts.month)
        wknd = bool(row['is_weekend']) if 'is_weekend' in row else (dow >= 5)
        yr = int(row.get('year') or (ts.year if ts is not None else 2025))

        if stn in train_stations:
            baseline = baselines.get((stn, d), 500.0)
            factors = (hour_factors.get((h, wknd), 1.0)
                       * dow_factors.get(dow, 1.0)
                       * month_factors.get(m, 1.0)
                       * station_hour_adj.get((stn, d, h, wknd), 1.0)
                       * station_month_adj.get((stn, d, m), 1.0))
            trend = per_station_trend.get(stn, global_trend) if yr >= 2025 else 1.0
        else:
            key = f"('{stn}', '{d}')"
            baseline = cold_baselines.get(key, baselines.get((stn, d), 500.0))
            factors = (
                cold_hour.get(f"('{stn}', '{d}', {h}, {wknd})",
                              hour_factors.get((h, wknd), 1.0))
                * cold_dow.get(f"('{stn}', '{d}', {dow})",
                               dow_factors.get(dow, 1.0))
                * cold_month.get(f"('{stn}', '{d}', {m})",
                                 month_factors.get(m, 1.0)))
            trend = cold_trends.get(stn, global_trend) if yr >= 2025 else 1.0

        predictions[i] = max(baseline * factors * trend, 0.0)

    return predictions


def main():
    print("=" * 70)
    print("  TRAFFIC FORECASTING - MASTER PIPELINE")
    print("=" * 70)
    
    # ============================================================
    # STEP 1: Build base model
    # ============================================================
    train, val, network, edges, submission = load_and_align_data()
    
    baselines, cold_stations = compute_station_baselines(train, val, network, edges)
    hour_factors, dow_factors, month_factors = compute_temporal_profiles(train, val)
    station_hour_adj, station_month_adj = compute_station_adjustments(
        train, hour_factors, dow_factors, month_factors
    )
    per_station_trend, global_trend = compute_trend(train, val)
    
    components = {
        'baselines': baselines,
        'hour_factors': hour_factors,
        'dow_factors': dow_factors,
        'month_factors': month_factors,
        'station_hour_adj': station_hour_adj,
        'station_month_adj': station_month_adj,
        'per_station_trend': per_station_trend,
        'global_trend': global_trend,
    }
    
    cold_enhanced = load_cold_start_enhanced()
    train_stations = set(train['station_key'].unique())

    # ============================================================
    # STEP 2: Generate base predictions on validation
    # ============================================================
    print("\n--- Step 2: Base predictions on validation ---")
    val_preds = predict_base(val, components, cold_enhanced, train_stations)
    val_residuals = val['volume'].values - val_preds
    
    val_mae = np.mean(np.abs(val_residuals))
    val_rmse = np.sqrt(np.mean(val_residuals**2))
    print(f"  Base model on validation: MAE={val_mae:.2f}, RMSE={val_rmse:.2f}")
    
    # ============================================================
    # STEP 3: Train LightGBM on residuals
    # ============================================================
    val_with_preds = val.copy()
    val_with_preds['predicted'] = val_preds
    
    lgb_model, improvement, importance = train_lgb_residual_model(
        val_with_preds, network, val_preds, val['volume'].values
    )
    
    # ============================================================
    # STEP 4: Generate submission predictions
    # ============================================================
    print("\n--- Step 4: Generating submission predictions ---")
    
    # Add temporal features to submission
    sub = submission.copy()
    sub['hour'] = sub['timestamp'].dt.hour
    sub['day_of_week'] = sub['timestamp'].dt.dayofweek
    sub['month'] = sub['timestamp'].dt.month
    sub['is_weekend'] = sub['day_of_week'].isin([5, 6])
    sub['year'] = sub['timestamp'].dt.year
    
    # Base predictions
    base_preds = predict_base(sub, components, cold_enhanced, train_stations)
    print(f"  Base predictions: mean={base_preds.mean():.1f}")
    
    # Apply LightGBM correction
    sub_features = prepare_lgb_features(sub, network, base_preds)
    # LightGBM was fitted on validation residuals, which contain warm stations
    # only, so its correction is trusted less on cold-start rows.
    lgb_corrections = lgb_model.predict(sub_features)
    blend = np.where(sub['station_key'].isin(train_stations), 0.7, 0.3)
    final_preds = np.maximum(base_preds + blend * lgb_corrections, 0.0)
    print(f"  After LGB correction: mean={final_preds.mean():.1f}")
    
    # ============================================================
    # STEP 5: Prediction Intervals
    # ============================================================
    val_with_preds['hour'] = val_with_preds['hour'].astype(int)
    val_with_preds['is_weekend'] = val_with_preds['is_weekend'].astype(bool)
    
    # Compute interval parameters from validation residuals
    # Use LGB-corrected validation residuals for better calibration
    val_lgb_features = prepare_lgb_features(val_with_preds, network, val_preds)
    val_corrected = apply_lgb_correction(val_preds, val_lgb_features, lgb_model, blend_weight=0.7)
    val_corrected_residuals = val['volume'].values - val_corrected
    val_with_preds['predicted'] = val_corrected  # update for interval computation
    
    interval_params = compute_prediction_intervals(
        val_with_preds, val_corrected_residuals, network, cold_stations
    )
    
    # Generate intervals for submission
    lower_90, upper_90 = generate_intervals(
        final_preds, sub, interval_params, cold_stations, cold_inflation=1.4
    )
    
    # ============================================================
    # STEP 6: Reliability Scores
    # ============================================================
    train_station_counts = train.groupby('station_key').size().to_dict()
    
    enhanced_stations = {k.strip("()").replace("'", "").split(", ")[0]
                         for k in cold_enhanced['cold_baselines']}
    reliability_scores = compute_reliability_scores(
        sub, cold_stations, per_station_trend, train_station_counts, network,
        enhanced_stations=enhanced_stations
    )
    
    # ============================================================
    # STEP 7: Assemble and validate submission
    # ============================================================
    print("\n--- Step 7: Assembling Final Submission ---")
    
    # Populate submission columns
    submission['forecast_volume'] = np.round(final_preds, 2)
    submission['lower_90'] = np.round(lower_90, 2)
    submission['upper_90'] = np.round(upper_90, 2)
    submission['reliability_score'] = np.round(reliability_scores, 4)
    
    # Final constraint enforcement
    submission['forecast_volume'] = submission['forecast_volume'].clip(lower=0)
    submission['lower_90'] = submission['lower_90'].clip(lower=0)
    submission['upper_90'] = submission['upper_90'].clip(lower=0)
    
    # Ensure lower <= forecast <= upper
    submission['lower_90'] = submission[['lower_90', 'forecast_volume']].min(axis=1)
    submission['upper_90'] = submission[['upper_90', 'forecast_volume']].max(axis=1)
    submission['reliability_score'] = submission['reliability_score'].clip(0, 1)
    
    # Check for any NaN/inf
    for col in ['forecast_volume', 'lower_90', 'upper_90', 'reliability_score']:
        n_bad = (~np.isfinite(submission[col])).sum()
        if n_bad > 0:
            print(f"  WARNING: {col} has {n_bad} non-finite values! Fixing...")
            submission[col] = submission[col].fillna(submission['forecast_volume'].median())
    
    # Save
    output_path = OUTPUT_DIR / 'submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"\n  ✓ Submission saved to: {output_path}")
    print(f"    Rows: {len(submission):,}")
    print(f"    Forecast volume: mean={submission['forecast_volume'].mean():.1f}, "
          f"range=[{submission['forecast_volume'].min():.1f}, {submission['forecast_volume'].max():.1f}]")
    print(f"    Lower 90: mean={submission['lower_90'].mean():.1f}")
    print(f"    Upper 90: mean={submission['upper_90'].mean():.1f}")
    print(f"    Mean interval width: {(submission['upper_90'] - submission['lower_90']).mean():.1f}")
    print(f"    Reliability: mean={submission['reliability_score'].mean():.3f}")
    
    return submission


if __name__ == '__main__':
    submission = main()
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
