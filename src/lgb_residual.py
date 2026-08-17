"""
LightGBM Residual Refinement Model (Layer 4)
=============================================
Trains on the residuals (actual - multiplicative_forecast) from validation data
to learn corrections that the multiplicative model misses.

Features:
- Temporal: hour, day_of_week, month, is_weekend, week_of_year
- Station context: through_lanes, speed_limit, aadt_band, functional_class
- Interactions: hour×weekend, volume_level_bucket
- Multiplicative model prediction (as a feature for heteroscedasticity)

Key design decisions:
- Train on validation residuals (since we want to correct for patterns
  the base model systematically gets wrong)
- Use station FEATURES not station IDs (to generalize to cold-start)
- Regularized to avoid overfitting (max_depth=6, min_child=50)
- If improvement < 5% on held-out data, discard the layer
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path


def prepare_lgb_features(df, network, base_predictions=None):
    """
    Prepare feature matrix for LightGBM.
    
    Args:
        df: DataFrame with hour, day_of_week, month, is_weekend, station_key, direction_code
        network: Network features DataFrame
        base_predictions: Optional array of base model predictions (for residual learning)
    
    Returns:
        Feature DataFrame ready for LightGBM
    """
    features = pd.DataFrame(index=df.index)
    
    # Temporal features
    features['hour'] = df['hour'].astype(int)
    features['day_of_week'] = df['day_of_week'].astype(int)
    features['month'] = df['month'].astype(int)
    features['is_weekend'] = df['is_weekend'].astype(int)
    
    # Cyclical encoding for hour (sin/cos)
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    # Interaction features
    features['hour_x_weekend'] = features['hour'] * features['is_weekend']
    features['is_rush_hour'] = ((features['hour'] >= 7) & (features['hour'] <= 9) | 
                                 (features['hour'] >= 16) & (features['hour'] <= 18)).astype(int)
    features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 5)).astype(int)
    
    # Week of year (for seasonality within months)
    if 'timestamp' in df.columns:
        features['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    
    # Station context features (from network)
    # Merge on station_key
    network_features = network.set_index('station_key')[
        ['through_lanes', 'speed_limit', 'lane_width', 'aadt_band', 
         'functional_class', 'roadway_width_band', 'surface_type']
    ].copy()
    
    # Encode categoricals as integers
    aadt_order = {'<5k': 0, '5k-10k': 1, '10k-25k': 2, '25k-50k': 3, '50k-100k': 4, '100k+': 5}
    network_features['aadt_encoded'] = network_features['aadt_band'].map(aadt_order).fillna(2)
    
    fc_order = {'Local': 0, 'Minor Collector': 1, 'Major Collector': 2, 
                'Minor Arterial': 3, 'Other Principal Arterials': 4,
                'Other Expressways & Freeway': 5, 'Interstate': 6}
    network_features['fc_encoded'] = network_features['functional_class'].map(fc_order).fillna(3)
    
    width_order = {'<24': 0, '24-35': 1, '36-47': 2, '48-59': 3, '60+': 4}
    network_features['width_encoded'] = network_features['roadway_width_band'].map(width_order).fillna(2)
    
    surface_order = {'Surface Treated': 0, 'Asphalt': 1, 'Composite': 2, 'Concrete': 3}
    network_features['surface_encoded'] = network_features['surface_type'].map(surface_order).fillna(2)
    
    # Join network features to data
    stn_features = df['station_key'].map(
        lambda s: network_features.loc[s] if s in network_features.index else None
    )
    
    for col in ['through_lanes', 'speed_limit', 'lane_width', 'aadt_encoded', 
                'fc_encoded', 'width_encoded', 'surface_encoded']:
        features[col] = df['station_key'].map(
            lambda s, c=col: network_features.loc[s][c] if s in network_features.index else np.nan
        )
    
    # Base model prediction (helps LightGBM learn heteroscedastic corrections)
    if base_predictions is not None:
        features['base_prediction'] = base_predictions
        features['log_base_pred'] = np.log1p(np.maximum(base_predictions, 0))
    
    # Direction code
    features['direction_code'] = df['direction_code'].astype(int)
    
    return features


def train_lgb_residual_model(val, network, base_predictions, actual_volumes,
                              val_split_month=5):
    """
    Train LightGBM on residuals from the multiplicative model.
    
    Uses months 1–4 as training, month 5–6 as test within validation.
    
    Args:
        val: Validation DataFrame
        network: Network features DataFrame
        base_predictions: Base model predictions for validation set
        actual_volumes: Actual volumes from validation set
        val_split_month: Month to split train/test within validation
    
    Returns:
        model: Trained LightGBM model
        improvement_pct: Percentage improvement in MAE over base model
        feature_importance: Feature importance DataFrame
    """
    print("\n--- Layer 4: LightGBM Residual Refinement ---")
    
    # Compute residuals
    residuals = actual_volumes - base_predictions
    
    # Prepare features
    features = prepare_lgb_features(val, network, base_predictions)
    
    # Split within validation: train on months 1–4, test on months 5–6
    train_mask = val['month'] < val_split_month
    test_mask = val['month'] >= val_split_month
    
    X_train = features[train_mask]
    y_train = residuals[train_mask]
    X_test = features[test_mask]
    y_test = residuals[test_mask]
    
    print(f"  Train: {len(X_train):,} rows (months < {val_split_month})")
    print(f"  Test: {len(X_test):,} rows (months >= {val_split_month})")
    
    # LightGBM parameters (conservative to avoid overfitting)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': 42,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train with early stopping
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=callbacks,
    )
    
    # Evaluate improvement
    test_preds_residual = model.predict(X_test)
    
    # Base model MAE on test split
    base_mae_test = np.mean(np.abs(y_test))  # residuals are (actual - base), MAE of residuals = base MAE
    
    # Corrected predictions
    corrected_preds = base_predictions[test_mask] + test_preds_residual
    actual_test = actual_volumes[test_mask]
    corrected_mae = np.mean(np.abs(actual_test - corrected_preds))
    base_mae_on_test = np.mean(np.abs(actual_test - base_predictions[test_mask]))
    
    improvement_pct = (base_mae_on_test - corrected_mae) / base_mae_on_test * 100
    
    print(f"\n  Test Results:")
    print(f"    Base model MAE (test split): {base_mae_on_test:.2f}")
    print(f"    With LightGBM correction:    {corrected_mae:.2f}")
    print(f"    Improvement: {improvement_pct:.1f}%")
    print(f"    Best iteration: {model.best_iteration}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 Features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:20s} : {row['importance']:.1f}")
    
    # Decision: keep or discard?
    if improvement_pct < 3.0:
        print(f"\n  ⚠ Improvement ({improvement_pct:.1f}%) < 3% threshold. Consider discarding.")
    else:
        print(f"\n  ✓ LightGBM improves MAE by {improvement_pct:.1f}%. Keeping.")
    
    return model, improvement_pct, importance


def apply_lgb_correction(predictions, features, lgb_model, blend_weight=0.7):
    """
    Apply LightGBM residual correction to base predictions.
    
    Args:
        predictions: Base model predictions
        features: Feature DataFrame for the targets
        lgb_model: Trained LightGBM model
        blend_weight: How much of the LGB correction to apply (0=none, 1=full)
    
    Returns:
        Corrected predictions
    """
    lgb_corrections = lgb_model.predict(features)
    corrected = predictions + blend_weight * lgb_corrections
    return np.maximum(corrected, 0.0)


if __name__ == '__main__':
    from forecast_model import build_model, validate_model
    
    # Build base model
    components, train, val, network, edges, submission = build_model()
    
    # Generate base predictions on full validation set
    print("\n--- Generating base predictions on full validation ---")
    val_df = val.copy()
    val_df['dir_str'] = val_df['direction_code'].astype(str)
    val_df['is_weekend_bool'] = val_df['is_weekend'].astype(bool)
    
    base_preds = np.zeros(len(val_df))
    for i in range(len(val_df)):
        row = val_df.iloc[i]
        stn = row['station_key']
        d = str(row['direction_code'])
        h = int(row['hour'])
        dow = int(row['day_of_week'])
        m = int(row['month'])
        wknd = bool(row['is_weekend'])
        
        baseline = components['baselines'].get((stn, d), 500.0)
        hf = components['hour_factors'].get((h, wknd), 1.0)
        df = components['dow_factors'].get(dow, 1.0)
        mf = components['month_factors'].get(m, 1.0)
        sha = components['station_hour_adj'].get((stn, d, h, wknd), 1.0)
        sma = components['station_month_adj'].get((stn, d, m), 1.0)
        trend = components['per_station_trend'].get(stn, components['global_trend'])
        
        base_preds[i] = max(baseline * hf * df * mf * sha * sma * trend, 0.0)
    
    print(f"  Base predictions: mean={base_preds.mean():.1f}, MAE={np.mean(np.abs(val_df['volume'].values - base_preds)):.2f}")
    
    # Train LightGBM
    lgb_model, improvement, importance = train_lgb_residual_model(
        val_df, network, base_preds, val_df['volume'].values
    )
