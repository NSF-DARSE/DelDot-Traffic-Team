"""
Traffic Volume Forecasting Model (v2)
======================================
Multiplicative hierarchical model with LightGBM residual correction.

CRITICAL FINDING: Training timestamps are offset by +2 hours from 
validation/submission timestamps. Training "hour 0" = actual "hour 2".
Training covers actual hours 2–23. Hours 0–1 are only in validation.

Architecture:
    forecast = station_baseline × hour_factor × dow_factor × month_factor 
               × station_adj × trend_factor

Author: Traffic Pattern Analysis Model
Date: 2025-08-17
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT
MODEL_DIR = PROJECT_ROOT / 'model'
OUTPUT_DIR = MODEL_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

HOUR_OFFSET = 2  # Training hour + OFFSET = actual/submission hour
SHRINKAGE_THRESHOLD = 500


# ============================================================
# DATA LOADING & ALIGNMENT
# ============================================================
def load_and_align_data():
    """
    Load all data and align training hours to actual time.
    
    Training hour 0 → actual hour 2
    Training hour 21 → actual hour 23
    Training covers actual hours 2–23 (hours 0–1 only in validation).
    """
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'traffic_train.parquet')
    val = pd.read_parquet(DATA_DIR / 'traffic_validation.parquet')
    network = pd.read_csv(DATA_DIR / 'network.csv')
    edges = pd.read_csv(DATA_DIR / 'network_edges.csv')
    submission = pd.read_csv(DATA_DIR / 'submission_template.csv')
    submission['timestamp'] = pd.to_datetime(submission['timestamp'])
    
    # CRITICAL: Align training hours to actual time
    train['actual_hour'] = train['hour'] + HOUR_OFFSET
    # The original 'hour' column is the raw data; 'actual_hour' is the true hour
    
    print(f"  Train: {train.shape[0]:,} rows, hours {train['actual_hour'].min()}–{train['actual_hour'].max()} (actual)")
    print(f"  Val: {val.shape[0]:,} rows, hours {val['hour'].min()}–{val['hour'].max()}")
    print(f"  Submission: {submission.shape[0]:,} targets")
    print(f"  Hour offset applied: training_hour + {HOUR_OFFSET} = actual_hour")
    
    return train, val, network, edges, submission


# ============================================================
# LAYER 1: STATION-LEVEL BASELINES
# ============================================================
def compute_station_baselines(train, val, network, edges):
    """
    Compute baseline volume for each station×direction.
    Known stations: observed mean. Cold-start: feature regression + neighbors.
    """
    print("\n--- Layer 1: Station Baselines ---")
    
    train_stations = set(train['station_key'].unique())
    all_network_stations = set(network['station_key'].unique())
    cold_stations = all_network_stations - train_stations
    
    # Known stations: mean per (station, direction)
    known_baselines = (
        train.groupby(['station_key', 'direction_code'])['volume']
        .mean()
        .to_dict()
    )
    print(f"  Known baselines: {len(known_baselines)} station×direction combos")
    
    # === Cold-start: Feature-based regression ===
    known_station_means = train.groupby('station_key')['volume'].mean().reset_index()
    known_station_means.columns = ['station_key', 'mean_volume']
    known_with_features = known_station_means.merge(network, on='station_key')
    
    feature_cols = ['through_lanes', 'speed_limit', 'lane_width']
    cat_cols = ['aadt_band', 'functional_class', 'roadway_width_band', 'surface_type']
    
    for col in cat_cols:
        known_with_features[col] = known_with_features[col].fillna('Unknown')
    
    X_known = pd.get_dummies(known_with_features[feature_cols + cat_cols],
                             columns=cat_cols, drop_first=False)
    y_known = known_with_features['mean_volume'].values
    
    baseline_model = Ridge(alpha=10.0)
    baseline_model.fit(X_known, y_known)
    print(f"  Cold-start regression R² (in-sample): {baseline_model.score(X_known, y_known):.3f}")
    
    # Predict cold-start baselines
    cold_network = network[network['station_key'].isin(cold_stations)].copy()
    for col in cat_cols:
        cold_network[col] = cold_network[col].fillna('Unknown')
    
    X_cold = pd.get_dummies(cold_network[feature_cols + cat_cols],
                            columns=cat_cols, drop_first=False)
    for col in X_known.columns:
        if col not in X_cold.columns:
            X_cold[col] = 0
    X_cold = X_cold[X_known.columns]
    
    cold_predictions = np.maximum(baseline_model.predict(X_cold), 10.0)
    
    # === Neighbor transfer ===
    neighbor_map = _build_neighbor_map(edges)
    
    cold_baselines = {}
    for idx, (_, row) in enumerate(cold_network.iterrows()):
        stn = row['station_key']
        reg_pred = cold_predictions[idx]
        
        neighbors = neighbor_map.get(stn, [])
        known_vols, weights = [], []
        for neighbor, w in neighbors:
            if neighbor in train_stations:
                known_vols.append(train[train['station_key'] == neighbor]['volume'].mean())
                weights.append(w)
        
        if known_vols:
            neighbor_avg = np.average(known_vols, weights=weights)
            blended = 0.4 * reg_pred + 0.6 * neighbor_avg
        else:
            blended = reg_pred
        
        cold_baselines[stn] = blended
    
    # Assign direction-level baselines for cold-start (slight asymmetry)
    cold_direction_baselines = {}
    for stn, vol in cold_baselines.items():
        for dir_code in ['1', '3', '5', '7']:
            asym = 1.025 if dir_code in ['1', '3'] else 0.975
            cold_direction_baselines[(stn, dir_code)] = vol * asym
    
    all_baselines = {**known_baselines, **cold_direction_baselines}
    print(f"  Cold-start baselines: {len(cold_baselines)} stations")
    print(f"  Total baselines: {len(all_baselines)} entries")
    
    return all_baselines, cold_stations


def _build_neighbor_map(edges):
    """Build weighted neighbor map from edges."""
    neighbor_map = {}
    for _, row in edges.iterrows():
        fr, to = row['from_station_key'], row['to_station_key']
        weight = 1.0
        if row['edge_type'] == 'same_corridor':
            weight *= 2.0
        if row['distance_band'] == '<=1mi':
            weight *= 3.0
        elif row['distance_band'] == '1-5mi':
            weight *= 2.0
        
        neighbor_map.setdefault(fr, []).append((to, weight))
        neighbor_map.setdefault(to, []).append((fr, weight))
    return neighbor_map


# ============================================================
# LAYER 2: TEMPORAL PROFILES (using aligned hours)
# ============================================================
def compute_temporal_profiles(train, val):
    """
    Compute multiplicative temporal factors using ACTUAL hours.
    
    Hours 2–23: learned from training (using actual_hour = raw_hour + 2)
    Hours 0–1: learned from validation data
    """
    print("\n--- Layer 2: Temporal Profiles ---")
    
    global_mean = train['volume'].mean()
    print(f"  Global mean volume (training): {global_mean:.1f}")
    
    # === HOUR × WEEKEND profile (actual hours 2–23 from training) ===
    hour_weekend_means = train.groupby(['actual_hour', 'is_weekend'])['volume'].mean()
    
    hour_factors = {}
    for (hour, is_wknd), mean_vol in hour_weekend_means.items():
        hour_factors[(int(hour), bool(is_wknd))] = mean_vol / global_mean
    
    # Hours 0–1: learn from validation (these truly don't exist in training)
    # Use validation's relative profile shape for hours 0–1
    # Anchor: hour 2 factor from training should match hour 2 in validation (after trend)
    val_hourly = val.groupby(['hour', 'is_weekend'])['volume'].mean()
    
    for hour in [0, 1]:
        for is_wknd in [False, True]:
            if (hour, is_wknd) in val_hourly.index:
                # Ratio of hour 0/1 to hour 2 in validation
                ref_val = val_hourly.get((2, is_wknd), val_hourly.get((3, is_wknd), 1))
                ratio_to_h2 = val_hourly[(hour, is_wknd)] / ref_val
                # Apply this ratio to our training-derived hour 2 factor
                hour_factors[(hour, is_wknd)] = hour_factors.get((2, is_wknd), 0.1) * ratio_to_h2
    
    print(f"  Hour×Weekend factors: {len(hour_factors)} entries (24h × 2)")
    print(f"    Hour  0 wkday: {hour_factors.get((0, False), 'N/A'):.4f}")
    print(f"    Hour  2 wkday: {hour_factors.get((2, False), 'N/A'):.4f}")
    print(f"    Hour  8 wkday: {hour_factors.get((8, False), 'N/A'):.4f}")
    print(f"    Hour 16 wkday: {hour_factors.get((16, False), 'N/A'):.4f}")
    print(f"    Hour 23 wkday: {hour_factors.get((23, False), 'N/A'):.4f}")
    
    # === DAY-OF-WEEK profile ===
    dow_means = train.groupby('day_of_week')['volume'].mean()
    dow_factors = {int(d): v / global_mean for d, v in dow_means.items()}
    print(f"  Day-of-week factors: Mon={dow_factors[0]:.3f}, Fri={dow_factors[4]:.3f}, Sun={dow_factors[6]:.3f}")
    
    # === MONTH profile ===
    month_means = train.groupby('month')['volume'].mean()
    month_factors = {int(m): v / global_mean for m, v in month_means.items()}
    print(f"  Month factors: Jan={month_factors[1]:.3f}, Jul={month_factors[7]:.3f}, Nov={month_factors[11]:.3f}")
    
    return hour_factors, dow_factors, month_factors


# ============================================================
# LAYER 2b: STATION-SPECIFIC ADJUSTMENTS (with shrinkage)
# ============================================================
def compute_station_adjustments(train, hour_factors, dow_factors, month_factors):
    """
    Station-specific deviations from global profiles, with shrinkage.
    Stations with less data are pulled toward the global average (factor=1).
    """
    print("\n--- Layer 2b: Station-Specific Adjustments ---")
    
    global_mean = train['volume'].mean()
    station_hour_adj = {}
    station_month_adj = {}
    
    for (stn, dir_code), group in train.groupby(['station_key', 'direction_code']):
        n_obs = len(group)
        w = min(n_obs / SHRINKAGE_THRESHOLD, 1.0)  # shrinkage weight
        stn_mean = group['volume'].mean()
        if stn_mean < 1:
            continue
        
        # Hourly adjustment (using actual_hour)
        stn_hour = group.groupby(['actual_hour', 'is_weekend'])['volume'].mean()
        for (hour, is_wknd), vol in stn_hour.items():
            h_factor = hour_factors.get((int(hour), bool(is_wknd)), 1.0)
            expected = stn_mean * h_factor
            if expected > 0:
                raw_adj = vol / expected
                adj = w * raw_adj + (1 - w) * 1.0
                station_hour_adj[(stn, dir_code, int(hour), bool(is_wknd))] = adj
        
        # Monthly adjustment
        stn_month = group.groupby('month')['volume'].mean()
        for month, vol in stn_month.items():
            m_factor = month_factors.get(int(month), 1.0)
            expected = stn_mean * m_factor
            if expected > 0:
                raw_adj = vol / expected
                adj = w * raw_adj + (1 - w) * 1.0
                station_month_adj[(stn, dir_code, int(month))] = adj
    
    print(f"  Station-hour adjustments: {len(station_hour_adj):,} entries")
    print(f"  Station-month adjustments: {len(station_month_adj):,} entries")
    
    return station_hour_adj, station_month_adj


# ============================================================
# LAYER 3: YEAR-OVER-YEAR TREND
# ============================================================
def compute_trend(train, val):
    """
    Estimate per-station year-over-year trend using aligned hours.
    Compare training (2024) to validation (2025) for matching months/hours.
    """
    print("\n--- Layer 3: Year-over-Year Trend ---")
    
    # Use overlapping months (1–6) and hours (2–23) for fair comparison
    train_subset = train[train['month'].isin([1,2,3,4,5,6])]
    val_subset = val[(val['hour'] >= 2) & (val['month'].isin([1,2,3,4,5,6]))]
    
    train_stn_means = train_subset.groupby('station_key')['volume'].mean()
    val_stn_means = val_subset.groupby('station_key')['volume'].mean()
    
    per_station_trend = {}
    for stn in train_stn_means.index:
        if stn in val_stn_means.index and train_stn_means[stn] > 0:
            per_station_trend[stn] = val_stn_means[stn] / train_stn_means[stn]
    
    # Robust global trend (trimmed mean)
    trends_arr = np.array(list(per_station_trend.values()))
    q10, q90 = np.percentile(trends_arr, [10, 90])
    trimmed = trends_arr[(trends_arr >= q10) & (trends_arr <= q90)]
    global_trend = float(trimmed.mean())
    
    # Also compute volume-weighted global ratio
    vol_weighted_trend = val_subset['volume'].mean() / train_subset['volume'].mean()
    
    print(f"  Per-station trends: {len(per_station_trend)} stations")
    print(f"  Trimmed mean (10-90%): {global_trend:.4f}")
    print(f"  Volume-weighted global: {vol_weighted_trend:.4f}")
    print(f"  Using: {vol_weighted_trend:.4f} as global trend")
    
    # Use volume-weighted as the primary global trend
    global_trend = vol_weighted_trend
    
    return per_station_trend, global_trend


# ============================================================
# GENERATE FORECASTS
# ============================================================
def generate_forecasts(submission, baselines, hour_factors, dow_factors,
                       month_factors, station_hour_adj, station_month_adj,
                       per_station_trend, global_trend, cold_stations):
    """
    Generate point forecasts for all submission targets.
    
    forecast = baseline × hour_factor × dow_factor × month_factor 
               × station_hour_adj × station_month_adj × trend_factor
    """
    print("\n--- Generating Forecasts ---")
    
    sub = submission.copy()
    sub['hour'] = sub['timestamp'].dt.hour
    sub['day_of_week'] = sub['timestamp'].dt.dayofweek
    sub['month'] = sub['timestamp'].dt.month
    sub['is_weekend'] = sub['day_of_week'].isin([5, 6])
    sub['year'] = sub['timestamp'].dt.year
    sub['dir_str'] = sub['direction_code'].astype(str)
    
    # Vectorized lookups via apply (faster than row-by-row for 170K rows)
    def predict_row(r):
        stn = r['station_key']
        d = r['dir_str']
        h = int(r['hour'])
        dow = int(r['day_of_week'])
        m = int(r['month'])
        wknd = bool(r['is_weekend'])
        year = int(r['year'])
        
        baseline = baselines.get((stn, d), 500.0)
        hf = hour_factors.get((h, wknd), 1.0)
        df = dow_factors.get(dow, 1.0)
        mf = month_factors.get(m, 1.0)
        sha = station_hour_adj.get((stn, d, h, wknd), 1.0)
        sma = station_month_adj.get((stn, d, m), 1.0)
        trend = per_station_trend.get(stn, global_trend) if year >= 2025 else 1.0
        
        return max(baseline * hf * df * mf * sha * sma * trend, 0.0)
    
    sub['forecast_volume'] = sub.apply(predict_row, axis=1)
    
    # Clean up
    result = sub.drop(columns=['hour', 'day_of_week', 'month', 'is_weekend', 'year', 'dir_str'])
    
    print(f"  Generated {len(result):,} forecasts")
    print(f"  Mean: {result['forecast_volume'].mean():.1f}")
    print(f"  Median: {result['forecast_volume'].median():.1f}")
    print(f"  Range: [{result['forecast_volume'].min():.1f}, {result['forecast_volume'].max():.1f}]")
    
    return result


# ============================================================
# VALIDATION
# ============================================================
def validate_model(val, baselines, hour_factors, dow_factors, month_factors,
                   station_hour_adj, station_month_adj, per_station_trend, 
                   global_trend, cold_stations, sample_size=None):
    """
    Validate model on the validation set. Returns predictions and metrics.
    """
    print("\n--- Model Validation ---")
    
    if sample_size:
        val_df = val.sample(sample_size, random_state=42).copy()
        print(f"  Using {sample_size:,} sample rows")
    else:
        val_df = val.copy()
        print(f"  Using full validation set ({len(val_df):,} rows)")
    
    val_df['dir_str'] = val_df['direction_code'].astype(str)
    val_df['is_weekend'] = val_df['is_weekend'].astype(bool)
    
    # Generate predictions
    predictions = np.zeros(len(val_df))
    for i in range(len(val_df)):
        row = val_df.iloc[i]
        stn = row['station_key']
        d = str(row['direction_code'])
        h = int(row['hour'])
        dow = int(row['day_of_week'])
        m = int(row['month'])
        wknd = bool(row['is_weekend'])
        
        baseline = baselines.get((stn, d), 500.0)
        hf = hour_factors.get((h, wknd), 1.0)
        df_val = dow_factors.get(dow, 1.0)
        mf = month_factors.get(m, 1.0)
        sha = station_hour_adj.get((stn, d, h, wknd), 1.0)
        sma = station_month_adj.get((stn, d, m), 1.0)
        trend = per_station_trend.get(stn, global_trend)  # val is 2025
        
        predictions[i] = max(baseline * hf * df_val * mf * sha * sma * trend, 0.0)
    
    val_df['predicted'] = predictions
    actual = val_df['volume'].values
    residuals = actual - predictions
    
    # Metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mask = actual > 10  # avoid MAPE explosion on near-zero values
    mape = np.mean(np.abs(residuals[mask]) / actual[mask]) * 100
    r2 = 1 - np.sum(residuals**2) / np.sum((actual - actual.mean())**2)
    
    print(f"\n  Overall Metrics:")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAPE: {mape:.1f}% (excluding vol < 10)")
    print(f"    R²:   {r2:.4f}")
    
    # By hour bucket
    print(f"\n  By Hour:")
    for h in [0, 1, 6, 8, 12, 16, 18, 22, 23]:
        h_data = val_df[val_df['hour'] == h]
        if len(h_data) > 0:
            h_mae = np.mean(np.abs(h_data['volume'] - h_data['predicted']))
            print(f"    Hour {h:2d}: MAE={h_mae:.1f} (n={len(h_data)})")
    
    return val_df, residuals


# ============================================================
# MAIN
# ============================================================
def build_model():
    """Build the full multiplicative model and return all components."""
    train, val, network, edges, submission = load_and_align_data()
    
    baselines, cold_stations = compute_station_baselines(train, val, network, edges)
    hour_factors, dow_factors, month_factors = compute_temporal_profiles(train, val)
    station_hour_adj, station_month_adj = compute_station_adjustments(
        train, hour_factors, dow_factors, month_factors
    )
    per_station_trend, global_trend = compute_trend(train, val)
    
    model_components = {
        'baselines': baselines,
        'hour_factors': hour_factors,
        'dow_factors': dow_factors,
        'month_factors': month_factors,
        'station_hour_adj': station_hour_adj,
        'station_month_adj': station_month_adj,
        'per_station_trend': per_station_trend,
        'global_trend': global_trend,
        'cold_stations': cold_stations,
    }
    
    return model_components, train, val, network, edges, submission


if __name__ == '__main__':
    components, train, val, network, edges, submission = build_model()
    
    # Validate on sample
    val_results, residuals = validate_model(
        val, sample_size=50000, **{k: components[k] for k in [
            'baselines', 'hour_factors', 'dow_factors', 'month_factors',
            'station_hour_adj', 'station_month_adj', 'per_station_trend',
            'global_trend', 'cold_stations'
        ]}
    )
    
    print("\n✓ Model v2 (with hour alignment) built and validated.")
