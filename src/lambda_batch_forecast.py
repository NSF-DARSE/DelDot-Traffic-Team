"""
Batch Forecast Lambda — Daily Rolling 30-Day Predictions
=========================================================
Triggered by EventBridge daily at 5am ET (9:00 UTC).
Generates forecasts for every station x direction x hour over the next 30 days
and writes an enriched CSV to S3 for QuickSight.

Uses the SAME prediction logic as the on-demand API, including the
enhanced cold-start path (station-specific profiles built from synthetic
history reconstructed from neighboring stations).

Output:
    s3://<bucket>/quicksight/rolling/forecast_latest.csv          (current)
    s3://<bucket>/quicksight/rolling/history/forecast_<date>.csv  (archive)

Invocation is IAM-authenticated only (EventBridge, or `aws lambda invoke`).
There is no public function URL.
"""

import json
import os
import csv
import io
import boto3
from datetime import datetime, timedelta, date

S3_BUCKET = os.environ.get('MODEL_BUCKET', 'deldot-traffic-forecasting-062905933333')
MODEL_KEY = os.environ.get('MODEL_KEY', 'models/v2.1/model_artifact.json')
OUTPUT_KEY = 'quicksight/rolling/forecast_latest.csv'

MODEL = None
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DIR_LABELS = {'1': 'Northbound', '3': 'Eastbound', '5': 'Southbound', '7': 'Westbound'}


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    s3 = boto3.client('s3')
    resp = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    MODEL = json.loads(resp['Body'].read().decode('utf-8'))
    meta = MODEL.get('metadata', {})
    print(f"Model loaded: v{meta.get('version')} "
          f"enhanced_cold_start={'cold_baselines_enhanced' in MODEL}")
    return MODEL


def predict_single(model, station, direction, hour, dow, month, year, is_weekend,
                   cold_set, has_enhanced_map):
    """
    Multiplicative forecast. Mirrors the API Lambda logic exactly.

    Returns (forecast, lower_90, upper_90, reliability, path)
    """
    is_cold = station in cold_set

    if not is_cold:
        baseline = model['baselines'].get(f"('{station}', '{direction}')", 500.0)
        hf = model['hour_factors'].get(f"({hour}, {is_weekend})", 1.0)
        df = model['dow_factors'].get(str(dow), 1.0)
        mf = model['month_factors'].get(str(month), 1.0)
        sha = model['station_hour_adj'].get(
            f"('{station}', '{direction}', {hour}, {is_weekend})", 1.0)
        sma = model['station_month_adj'].get(
            f"('{station}', '{direction}', {month})", 1.0)
        base_trend = model['per_station_trend'].get(station, model['global_trend'])
        path = 'known_station'
        has_enhanced = False
    else:
        b_key = f"('{station}', '{direction}')"
        has_enhanced = has_enhanced_map.get(b_key, False)
        if has_enhanced:
            baseline = model['cold_baselines_enhanced'][b_key]
            hf = model['cold_hour_factors_enhanced'].get(
                f"('{station}', '{direction}', {hour}, {is_weekend})",
                model['hour_factors'].get(f"({hour}, {is_weekend})", 1.0))
            df = model['cold_dow_factors_enhanced'].get(
                f"('{station}', '{direction}', {dow})",
                model['dow_factors'].get(str(dow), 1.0))
            mf = model['cold_month_factors_enhanced'].get(
                f"('{station}', '{direction}', {month})",
                model['month_factors'].get(str(month), 1.0))
            base_trend = model['cold_trends_enhanced'].get(station, model['global_trend'])
            path = 'cold_start_enhanced'
        else:
            baseline = model['baselines'].get(b_key, 500.0)
            hf = model['hour_factors'].get(f"({hour}, {is_weekend})", 1.0)
            df = model['dow_factors'].get(str(dow), 1.0)
            mf = model['month_factors'].get(str(month), 1.0)
            base_trend = model['global_trend']
            path = 'cold_start_fallback'
        sha = sma = 1.0

    # Compound the trend for years beyond the 2025 forecast base
    years_from_base = max(year - 2025, 0)
    trend = base_trend * (base_trend ** years_from_base) if year >= 2025 else 1.0

    forecast = max(baseline * hf * df * mf * sha * sma * trend, 0.0)

    # Uncertainty (same rules as the API)
    if forecast < 50:
        cv = 0.8
    elif forecast < 200:
        cv = 0.5
    elif forecast < 500:
        cv = 0.3
    elif forecast < 1000:
        cv = 0.22
    elif forecast < 2000:
        cv = 0.18
    else:
        cv = 0.15
    if is_cold:
        cv *= 1.4
    if hour <= 4 or hour >= 22:
        cv *= 1.3

    std = forecast * cv
    lower = max(forecast - 1.645 * std, 0.0)
    upper = forecast + 1.645 * std

    # Reliability (same rules as the API)
    reliability = 0.80
    if is_cold:
        reliability -= 0.22 if has_enhanced else 0.30
    if hour <= 1 or hour >= 22:
        reliability -= 0.08
    if years_from_base >= 1:
        reliability -= 0.10
    elif year == 2025 and month >= 10:
        reliability -= 0.05
    elif year == 2025 and month >= 7:
        reliability -= 0.03
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        reliability += 0.05
    reliability = max(0.10, min(0.95, reliability))

    return forecast, lower, upper, reliability, path


def get_all_station_directions(model):
    """Union of known baselines and enhanced cold-start baselines."""
    pairs = set()
    for src in ('baselines', 'cold_baselines_enhanced'):
        for key in model.get(src, {}):
            parts = key.strip("()").replace("'", "").split(", ")
            if len(parts) == 2:
                pairs.add((parts[0], parts[1]))
    return sorted(pairs)


def time_of_day_label(hour):
    if hour < 6:
        return 'Night (12am-6am)'
    if hour < 10:
        return 'Morning Rush (6am-10am)'
    if hour < 15:
        return 'Midday (10am-3pm)'
    if hour < 19:
        return 'Evening Rush (3pm-7pm)'
    return 'Evening (7pm-12am)'


def lambda_handler(event, context):
    model = load_model()

    today = date.today()
    forecast_days = int(os.environ.get('FORECAST_DAYS', '30'))
    print(f"Generating {forecast_days}d forecast from {today}")

    station_dirs = get_all_station_directions(model)
    cold_set = set(model.get('cold_stations', []))
    enhanced_baselines = model.get('cold_baselines_enhanced', {})
    has_enhanced_map = {k: True for k in enhanced_baselines}

    print(f"Station x direction pairs: {len(station_dirs)}")
    print(f"Cold-start stations: {len(cold_set)}, enhanced entries: {len(enhanced_baselines)}")

    rows = []
    path_counts = {}
    generated_at = datetime.utcnow().isoformat() + 'Z'

    for day_offset in range(forecast_days):
        d = today + timedelta(days=day_offset)
        dow, month, year = d.weekday(), d.month, d.year
        is_weekend = dow >= 5
        date_str = d.isoformat()

        for station, direction in station_dirs:
            for hour in range(24):
                fc, lo, hi, rel, path = predict_single(
                    model, station, direction, hour, dow, month, year,
                    is_weekend, cold_set, has_enhanced_map)
                path_counts[path] = path_counts.get(path, 0) + 1

                rows.append({
                    'station_key': station,
                    'direction_code': direction,
                    'direction_label': DIR_LABELS.get(direction, direction),
                    'date': date_str,
                    'hour': hour,
                    'day_of_week': DAY_NAMES[dow],
                    'month': month,
                    'year': year,
                    'is_weekend': 'Weekend' if is_weekend else 'Weekday',
                    'time_of_day': time_of_day_label(hour),
                    'forecast_volume': round(fc, 1),
                    'lower_90': round(lo, 1),
                    'upper_90': round(hi, 1),
                    'reliability_score': round(rel, 3),
                    'interval_width': round(hi - lo, 1),
                    'station_type': ('New Station (Cold-Start)' if station in cold_set
                                     else 'Established Station'),
                    'prediction_path': path,
                    'model_version': model.get('metadata', {}).get('version', 'unknown'),
                    'generated_at': generated_at,
                })

    print(f"Generated {len(rows):,} rows. Path breakdown: {path_counts}")

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    payload = out.getvalue().encode('utf-8')

    s3 = boto3.client('s3')
    s3.put_object(Bucket=S3_BUCKET, Key=OUTPUT_KEY, Body=payload, ContentType='text/csv')
    archive_key = f"quicksight/rolling/history/forecast_{today.isoformat()}.csv"
    s3.put_object(Bucket=S3_BUCKET, Key=archive_key, Body=payload, ContentType='text/csv')

    size_mb = len(payload) / (1024 * 1024)
    print(f"Wrote s3://{S3_BUCKET}/{OUTPUT_KEY} ({size_mb:.1f} MB)")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Generated {len(rows):,} forecasts for {forecast_days} days',
            'date_range': f'{today} to {today + timedelta(days=forecast_days-1)}',
            'station_directions': len(station_dirs),
            'path_breakdown': path_counts,
            'model_version': model.get('metadata', {}).get('version'),
            'output': f's3://{S3_BUCKET}/{OUTPUT_KEY}',
            'size_mb': round(size_mb, 1),
        })
    }
