"""
Delaware DOT Traffic Forecast API — Lambda Handler
====================================================
Serves real-time traffic volume predictions from the multiplicative model.

Endpoints:
    GET /forecast?station=STN_0067&direction=1&date=2025-09-15&hour=16
    GET /forecast?station=STN_0067&direction=1&date=2025-09-15
    GET /health

Returns:
    {
        "station": "STN_0067",
        "direction": "1",
        "timestamp": "2025-09-15T16:00:00",
        "forecast_volume": 1245.3,
        "lower_90": 980.1,
        "upper_90": 1510.5,
        "reliability_score": 0.78,
        "model_version": "2.0"
    }
"""

import json
import os
import boto3
from datetime import datetime, date

# Global model cache (persists across warm Lambda invocations)
MODEL = None
S3_BUCKET = os.environ.get('MODEL_BUCKET', 'deldot-traffic-forecasting-062905933333')
MODEL_KEY = os.environ.get('MODEL_KEY', 'models/v2/model_artifact.json')


def load_model():
    """Load model artifact from S3 (cached in global variable)."""
    global MODEL
    if MODEL is not None:
        return MODEL
    
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    MODEL = json.loads(response['Body'].read().decode('utf-8'))
    print(f"Model loaded: version={MODEL['metadata']['version']}, "
          f"baselines={len(MODEL['baselines'])}")
    return MODEL


def predict(station, direction, timestamp_str):
    """
    Generate a single forecast using the multiplicative model.
    
    Args:
        station: e.g., "STN_0067"
        direction: e.g., "1"
        timestamp_str: ISO format "2025-09-15T16:00:00"
    
    Returns:
        dict with forecast_volume, lower_90, upper_90, reliability_score
    """
    model = load_model()
    
    # Parse timestamp
    ts = datetime.fromisoformat(timestamp_str)
    hour = ts.hour
    dow = ts.weekday()  # 0=Monday
    month = ts.month
    is_weekend = dow >= 5
    year = ts.year
    
    # Layer 1: Station baseline
    key = f"('{station}', '{direction}')"
    baseline = model['baselines'].get(key, 500.0)
    
    # Layer 2: Temporal factors
    h_key = f"({hour}, {is_weekend})"
    h_factor = model['hour_factors'].get(h_key, 1.0)
    d_factor = model['dow_factors'].get(str(dow), 1.0)
    m_factor = model['month_factors'].get(str(month), 1.0)
    
    # Station-specific adjustments
    sh_key = f"('{station}', '{direction}', {hour}, {is_weekend})"
    sh_adj = model['station_hour_adj'].get(sh_key, 1.0)
    sm_key = f"('{station}', '{direction}', {month})"
    sm_adj = model['station_month_adj'].get(sm_key, 1.0)
    
    # Layer 3: Trend
    if year >= 2025:
        trend = model['per_station_trend'].get(station, model['global_trend'])
    else:
        trend = 1.0
    
    # Combine
    forecast = max(baseline * h_factor * d_factor * m_factor * sh_adj * sm_adj * trend, 0.0)
    
    # Uncertainty estimation (simplified for Lambda)
    # Use coefficient of variation based on volume level and time
    is_cold_start = station in model.get('cold_stations', [])
    
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
    
    # Inflate for cold-start
    if is_cold_start:
        cv *= 1.5
    
    # Night hours are more variable
    if hour <= 4 or hour >= 22:
        cv *= 1.3
    
    # 90% interval: approximately ±1.645 × std
    std_est = forecast * cv
    lower_90 = max(forecast - 1.645 * std_est, 0.0)
    upper_90 = forecast + 1.645 * std_est
    
    # Reliability score
    reliability = 0.80
    if is_cold_start:
        reliability -= 0.30
    if hour <= 1 or hour >= 22:
        reliability -= 0.08
    if year >= 2025 and month >= 10:
        reliability -= 0.05
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        reliability += 0.05
    reliability = max(0.10, min(0.95, reliability))
    
    return {
        'forecast_volume': round(forecast, 1),
        'lower_90': round(lower_90, 1),
        'upper_90': round(upper_90, 1),
        'reliability_score': round(reliability, 3),
    }


def lambda_handler(event, context):
    """
    Main Lambda handler. Supports API Gateway proxy integration.
    """
    try:
        # Handle API Gateway proxy format
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        params = event.get('queryStringParameters') or {}
        
        # Health check
        if path == '/health' or path == '/':
            return response(200, {
                'status': 'healthy',
                'model_version': '2.0',
                'service': 'deldot-traffic-forecast'
            })
        
        # Forecast endpoint
        if path == '/forecast':
            # Required parameters
            station = params.get('station')
            direction = params.get('direction', '1')
            date_str = params.get('date')
            hour = params.get('hour')
            
            if not station:
                return response(400, {'error': 'Missing required parameter: station'})
            if not date_str:
                return response(400, {'error': 'Missing required parameter: date (YYYY-MM-DD)'})
            
            # If hour specified, return single forecast
            if hour is not None:
                ts = f"{date_str}T{int(hour):02d}:00:00"
                result = predict(station, direction, ts)
                result.update({
                    'station': station,
                    'direction': direction,
                    'timestamp': ts,
                    'model_version': '2.0'
                })
                return response(200, result)
            
            # If no hour, return full day (24 forecasts)
            results = []
            for h in range(24):
                ts = f"{date_str}T{h:02d}:00:00"
                result = predict(station, direction, ts)
                result['hour'] = h
                results.append(result)
            
            return response(200, {
                'station': station,
                'direction': direction,
                'date': date_str,
                'forecasts': results,
                'model_version': '2.0'
            })
        
        return response(404, {'error': f'Unknown path: {path}'})
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return response(500, {'error': str(e)})


def response(status_code, body):
    """Format API Gateway proxy response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
        'body': json.dumps(body)
    }
