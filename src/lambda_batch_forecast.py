"""
Batch Forecast Lambda — Daily Rolling 30-Day Predictions
=========================================================
Runs daily (triggered by EventBridge at 5am ET).
Generates forecasts for all 118 stations × 2 directions × 24 hours × 30 days.
Writes enriched CSV to S3 for QuickSight consumption.

Output: s3://deldot-traffic-forecasting-062905933333/quicksight/rolling/forecast_latest.csv
"""

import json
import os
import csv
import io
import boto3
from datetime import datetime, timedelta, date

S3_BUCKET = os.environ.get('MODEL_BUCKET', 'deldot-traffic-forecasting-062905933333')
MODEL_KEY = os.environ.get('MODEL_KEY', 'models/v2/model_artifact.json')
OUTPUT_KEY = 'quicksight/rolling/forecast_latest.csv'

# Global model cache
MODEL = None


def load_model():
    """Load model from S3 (cached across warm invocations)."""
    global MODEL
    if MODEL is not None:
        return MODEL
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    MODEL = json.loads(response['Body'].read().decode('utf-8'))
    print(f"Model loaded: v{MODEL['metadata']['version']}")
    return MODEL


def predict_single(model, station, direction, hour, dow, month, year, is_weekend):
    """Generate a single forecast."""
    key = f"('{station}', '{direction}')"
    baseline = model['baselines'].get(key, 500.0)
    
    h_key = f"({hour}, {is_weekend})"
    h_factor = model['hour_factors'].get(h_key, 1.0)
    d_factor = model['dow_factors'].get(str(dow), 1.0)
    m_factor = model['month_factors'].get(str(month), 1.0)
    
    sh_key = f"('{station}', '{direction}', {hour}, {is_weekend})"
    sh_adj = model['station_hour_adj'].get(sh_key, 1.0)
    sm_key = f"('{station}', '{direction}', {month})"
    sm_adj = model['station_month_adj'].get(sm_key, 1.0)
    
    # Trend: compound annual decline
    # Base year is 2024. Each year adds another ~10% decline.
    years_from_base = max(year - 2024, 0)
    base_trend = model['per_station_trend'].get(station, model['global_trend'])
    trend = base_trend ** years_from_base if years_from_base > 0 else 1.0
    
    forecast = max(baseline * h_factor * d_factor * m_factor * sh_adj * sm_adj * trend, 0.0)
    
    # Uncertainty
    is_cold = station in model.get('cold_stations', [])
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
        cv *= 1.5
    if hour <= 4 or hour >= 22:
        cv *= 1.3
    
    std_est = forecast * cv
    lower_90 = max(forecast - 1.645 * std_est, 0.0)
    upper_90 = forecast + 1.645 * std_est
    
    # Reliability
    reliability = 0.80
    if is_cold:
        reliability -= 0.30
    if hour <= 1 or hour >= 22:
        reliability -= 0.08
    if years_from_base >= 2:
        reliability -= 0.10
    elif years_from_base >= 1:
        reliability -= 0.05
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        reliability += 0.05
    reliability = max(0.10, min(0.95, reliability))
    
    return forecast, lower_90, upper_90, reliability


def get_all_stations_directions(model):
    """Extract unique (station, direction) pairs from model baselines."""
    pairs = set()
    for key in model['baselines'].keys():
        # Keys look like "('STN_0067', '1')"
        parts = key.strip("()").replace("'", "").split(", ")
        if len(parts) == 2:
            pairs.add((parts[0], parts[1]))
    return sorted(pairs)


def time_of_day_label(hour):
    """Human-readable time category."""
    if hour < 6:
        return 'Night (12am-6am)'
    elif hour < 10:
        return 'Morning Rush (6am-10am)'
    elif hour < 15:
        return 'Midday (10am-3pm)'
    elif hour < 19:
        return 'Evening Rush (3pm-7pm)'
    else:
        return 'Evening (7pm-12am)'


def day_name(dow):
    """Day of week number to name."""
    names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return names[dow]


def lambda_handler(event, context):
    """
    Generate 30-day rolling forecasts for all stations and write to S3.
    """
    model = load_model()
    
    # Determine date range: today + 30 days
    today = date.today()
    forecast_days = int(os.environ.get('FORECAST_DAYS', '30'))
    
    print(f"Generating forecasts: {today} to {today + timedelta(days=forecast_days-1)}")
    
    # Get all station-direction pairs
    station_dirs = get_all_stations_directions(model)
    print(f"Stations × Directions: {len(station_dirs)}")
    
    # Direction code to label
    dir_labels = {'1': 'Northbound', '3': 'Eastbound', '5': 'Southbound', '7': 'Westbound'}
    
    # Cold-start set
    cold_set = set(model.get('cold_stations', []))
    
    # Generate forecasts
    rows = []
    for day_offset in range(forecast_days):
        current_date = today + timedelta(days=day_offset)
        dow = current_date.weekday()
        month = current_date.month
        year = current_date.year
        is_weekend = dow >= 5
        date_str = current_date.isoformat()
        
        for station, direction in station_dirs:
            for hour in range(24):
                forecast, lower, upper, reliability = predict_single(
                    model, station, direction, hour, dow, month, year, is_weekend
                )
                
                rows.append({
                    'station_key': station,
                    'direction_code': direction,
                    'direction_label': dir_labels.get(direction, direction),
                    'date': date_str,
                    'hour': hour,
                    'day_of_week': day_name(dow),
                    'month': month,
                    'year': year,
                    'is_weekend': 'Weekend' if is_weekend else 'Weekday',
                    'time_of_day': time_of_day_label(hour),
                    'forecast_volume': round(forecast, 1),
                    'lower_90': round(lower, 1),
                    'upper_90': round(upper, 1),
                    'reliability_score': round(reliability, 3),
                    'interval_width': round(upper - lower, 1),
                    'station_type': 'New Station (Cold-Start)' if station in cold_set else 'Established Station',
                    'generated_at': datetime.utcnow().isoformat() + 'Z',
                })
    
    print(f"Generated {len(rows):,} forecast rows")
    
    # Write CSV to memory
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    # Upload to S3
    s3 = boto3.client('s3')
    csv_bytes = output.getvalue().encode('utf-8')
    
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=OUTPUT_KEY,
        Body=csv_bytes,
        ContentType='text/csv'
    )
    
    # Also save a dated copy for history
    dated_key = f"quicksight/rolling/history/forecast_{today.isoformat()}.csv"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=dated_key,
        Body=csv_bytes,
        ContentType='text/csv'
    )
    
    size_mb = len(csv_bytes) / (1024 * 1024)
    print(f"Uploaded to s3://{S3_BUCKET}/{OUTPUT_KEY} ({size_mb:.1f} MB)")
    print(f"Archived to s3://{S3_BUCKET}/{dated_key}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Generated {len(rows):,} forecasts for {forecast_days} days',
            'date_range': f'{today} to {today + timedelta(days=forecast_days-1)}',
            'stations': len(station_dirs),
            'output': f's3://{S3_BUCKET}/{OUTPUT_KEY}',
            'size_mb': round(size_mb, 1)
        })
    }
