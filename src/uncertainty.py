"""
Uncertainty Estimation Module
==============================
Generates 90% prediction intervals and reliability scores.

Approach:
1. Empirical quantiles: compute 5th/95th percentile of residuals
   per (station_group × hour_bucket × weekend) cell
2. Scale intervals by predicted volume level (heteroscedastic)
3. Calibrate on held-out validation data to achieve ~90% coverage
4. Assign reliability scores based on data availability and prediction difficulty

Author: Traffic Pattern Analysis Model
Date: 2026-08-17
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PREDICTION INTERVALS
# ============================================================
def compute_prediction_intervals(val_df, residuals, network, cold_stations):
    """
    Compute empirical prediction interval parameters from validation residuals.
    
    Groups residuals by (volume_bucket × hour_bucket × is_weekend) and computes
    the 5th and 95th percentiles. This gives heteroscedastic intervals:
    wider for high-volume/uncertain conditions.
    
    Args:
        val_df: Validation DataFrame with 'predicted' column
        residuals: Array of (actual - predicted) residuals
        network: Network DataFrame
        cold_stations: Set of cold-start station keys
    
    Returns:
        interval_params: Dict with parameters for generating intervals
    """
    print("\n--- Uncertainty: Computing Prediction Intervals ---")
    
    df = val_df.copy()
    df['residual'] = residuals
    df['abs_residual'] = np.abs(residuals)
    
    # Create grouping buckets
    # Volume level buckets (based on predicted volume)
    df['vol_bucket'] = pd.cut(df['predicted'], 
                               bins=[0, 50, 200, 500, 1000, 2000, 5000, float('inf')],
                               labels=['0-50', '50-200', '200-500', '500-1k', '1k-2k', '2k-5k', '5k+'])
    
    # Hour buckets
    df['hour_bucket'] = pd.cut(df['hour'],
                                bins=[-1, 5, 9, 13, 17, 21, 24],
                                labels=['night', 'morning', 'midday', 'afternoon', 'evening', 'late'])
    
    # Compute quantiles per cell
    interval_params = {}
    
    for (vol_b, hour_b, wknd), group in df.groupby(['vol_bucket', 'hour_bucket', 'is_weekend'], observed=True):
        if len(group) < 20:
            continue
        q05 = np.percentile(group['residual'], 5)
        q95 = np.percentile(group['residual'], 95)
        mean_pred = group['predicted'].mean()
        cv = group['abs_residual'].mean() / max(mean_pred, 1)  # coefficient of variation
        
        interval_params[(str(vol_b), str(hour_b), bool(wknd))] = {
            'q05': q05,
            'q95': q95,
            'cv': cv,
            'n': len(group),
            'mean_pred': mean_pred,
        }
    
    # Also compute overall fallback percentiles
    overall_q05 = np.percentile(residuals, 5)
    overall_q95 = np.percentile(residuals, 95)
    overall_cv = np.mean(np.abs(residuals)) / max(df['predicted'].mean(), 1)
    
    interval_params['__fallback__'] = {
        'q05': overall_q05,
        'q95': overall_q95,
        'cv': overall_cv,
        'n': len(df),
        'mean_pred': df['predicted'].mean(),
    }
    
    print(f"  Computed intervals for {len(interval_params) - 1} cells + 1 fallback")
    print(f"  Overall: q05={overall_q05:.1f}, q95={overall_q95:.1f}, CV={overall_cv:.3f}")
    
    # Calibration check: what's the actual coverage?
    coverage = _check_coverage(df, interval_params)
    print(f"  Empirical coverage check: {coverage:.1f}% (target: 90%)")
    
    return interval_params


def _check_coverage(df, interval_params):
    """Check coverage of the interval params on the same data (optimistic)."""
    covered = 0
    total = 0
    
    for _, row in df.iterrows():
        pred = row['predicted']
        actual = row['volume']
        
        vol_b = str(row['vol_bucket'])
        hour_b = str(row['hour_bucket'])
        wknd = bool(row['is_weekend'])
        
        params = interval_params.get((vol_b, hour_b, wknd), interval_params['__fallback__'])
        
        lower = pred + params['q05']
        upper = pred + params['q95']
        lower = max(lower, 0)
        
        if lower <= actual <= upper:
            covered += 1
        total += 1
        
        if total >= 10000:  # sample for speed
            break
    
    return covered / total * 100


def generate_intervals(predictions, submission_df, interval_params, cold_stations,
                       cold_inflation=1.5):
    """
    Generate prediction intervals for submission targets.
    
    Args:
        predictions: Point forecasts (array)
        submission_df: Submission DataFrame with temporal features
        interval_params: Interval parameters from compute_prediction_intervals
        cold_stations: Set of cold-start station keys
        cold_inflation: Multiplier to widen intervals for cold-start stations
    
    Returns:
        lower_90: Array of lower bounds
        upper_90: Array of upper bounds
    """
    print("\n--- Generating Prediction Intervals ---")
    
    n = len(predictions)
    lower_90 = np.zeros(n)
    upper_90 = np.zeros(n)
    
    # Pre-compute buckets
    hour = submission_df['timestamp'].dt.hour.values
    is_weekend = submission_df['timestamp'].dt.dayofweek.isin([5, 6]).values
    
    # Hour bucket mapping
    def hour_to_bucket(h):
        if h <= 5: return 'night'
        elif h <= 9: return 'morning'
        elif h <= 13: return 'midday'
        elif h <= 17: return 'afternoon'
        elif h <= 21: return 'evening'
        else: return 'late'
    
    # Volume bucket mapping
    def vol_to_bucket(v):
        if v <= 50: return '0-50'
        elif v <= 200: return '50-200'
        elif v <= 500: return '200-500'
        elif v <= 1000: return '500-1k'
        elif v <= 2000: return '1k-2k'
        elif v <= 5000: return '2k-5k'
        else: return '5k+'
    
    fallback = interval_params['__fallback__']
    
    for i in range(n):
        pred = predictions[i]
        h_bucket = hour_to_bucket(hour[i])
        v_bucket = vol_to_bucket(pred)
        wknd = bool(is_weekend[i])
        
        params = interval_params.get((v_bucket, h_bucket, wknd), fallback)
        
        # Base interval from empirical quantiles
        low = pred + params['q05']
        high = pred + params['q95']
        
        # Inflate for cold-start stations
        stn = submission_df.iloc[i]['station_key']
        if stn in cold_stations:
            width = high - low
            center = (high + low) / 2
            low = center - width * cold_inflation / 2
            high = center + width * cold_inflation / 2
        
        # Enforce constraints
        lower_90[i] = max(low, 0.0)
        upper_90[i] = max(high, lower_90[i])  # upper >= lower
        
        # Ensure forecast is within bounds
        if pred < lower_90[i]:
            lower_90[i] = pred * 0.5
        if pred > upper_90[i]:
            upper_90[i] = pred * 1.5
    
    print(f"  Intervals generated for {n:,} targets")
    print(f"  Mean width: {np.mean(upper_90 - lower_90):.1f}")
    print(f"  Median width: {np.median(upper_90 - lower_90):.1f}")
    
    return lower_90, upper_90


# ============================================================
# RELIABILITY SCORES
# ============================================================
def compute_reliability_scores(submission_df, cold_stations, per_station_trend,
                               train_station_counts, network,
                               enhanced_stations=frozenset()):
    """
    Assign reliability scores (0–1) based on prediction difficulty.
    
    Factors:
    - Data availability (cold-start → low reliability)
    - Temporal extrapolation (far-future → slightly lower)
    - Hour type (night hours more variable relative to mean)
    - Station consistency (high CV stations → lower)
    
    Args:
        submission_df: Submission DataFrame
        cold_stations: Set of cold-start station keys
        per_station_trend: Dict of per-station trend ratios
        train_station_counts: Dict of station → observation count
        network: Network DataFrame
    
    Returns:
        reliability_scores: Array of floats [0, 1]
    """
    print("\n--- Computing Reliability Scores ---")
    
    n = len(submission_df)
    scores = np.ones(n) * 0.80  # base score
    
    hours = submission_df['timestamp'].dt.hour.values
    months = submission_df['timestamp'].dt.month.values
    years = submission_df['timestamp'].dt.year.values
    stations = submission_df['station_key'].values
    
    for i in range(n):
        stn = stations[i]
        h = hours[i]
        m = months[i]
        yr = years[i]
        
        score = 0.80  # Base reliability
        
        # Cold-start penalty. Stations with a reconstructed station-specific
        # profile are penalised less than those falling back to network
        # averages, because more is known about them.
        if stn in cold_stations:
            score -= 0.22 if stn in enhanced_stations else 0.30
        
        # Limited training data penalty
        elif stn in train_station_counts:
            count = train_station_counts[stn]
            if count < 5000:  # limited history
                score -= 0.12
            elif count < 8000:
                score -= 0.05
        
        # Night hour penalty (more variable relative to low mean)
        if h <= 4 or h >= 22:
            score -= 0.08
        
        # Hours 0–1 penalty (only learned from validation, not training)
        if h <= 1:
            score -= 0.05
        
        # Far-future penalty (later in 2025 = more uncertainty)
        if yr >= 2025 and m >= 10:
            score -= 0.05
        elif yr >= 2025 and m >= 7:
            score -= 0.03
        
        # Peak hours are more predictable (regular commute patterns)
        if 7 <= h <= 9 or 16 <= h <= 18:
            score += 0.05
        
        # Stations with very high trend variance (unreliable trend estimate)
        if stn in per_station_trend:
            trend = per_station_trend[stn]
            if abs(trend - 0.9) > 0.3:  # far from typical
                score -= 0.05
        
        # Clamp to valid range
        scores[i] = np.clip(score, 0.10, 0.95)
    
    print(f"  Reliability scores computed for {n:,} targets")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Distribution:")
    for bucket in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
        count = ((scores >= bucket[0]) & (scores < bucket[1])).sum()
        print(f"    [{bucket[0]:.1f}, {bucket[1]:.1f}): {count:,} ({count/n*100:.1f}%)")
    
    return scores
