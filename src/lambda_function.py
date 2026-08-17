"""
Delaware DOT Traffic Forecast API — Lambda Handler
====================================================
Serves traffic volume predictions from the multiplicative model.

Endpoints:
    GET /health
        Service + model version check.

    GET /forecast?station=STN_0067&direction=1&date=2026-09-15&hour=16
        Single-hour forecast. Omit `hour` for a full 24-hour profile.

    GET /explain?station=STN_0067&direction=1&date=2026-09-15&hour=16
        Full factor-by-factor decomposition of a single prediction.
        Every prediction is auditable: shows each multiplicative factor,
        its meaning, and the running product.

Auth: API key required (enforced at API Gateway via `x-api-key` header).

Model paths:
    - Known stations       -> observed baseline + observed profiles + station adj
    - Cold-start enhanced  -> synthetic-history baseline + station-specific profiles
    - Cold-start fallback  -> feature-regression baseline + global profiles
"""

import json
import os
import boto3
from datetime import datetime

MODEL = None
S3_BUCKET = os.environ.get('MODEL_BUCKET', 'deldot-traffic-forecasting-062905933333')
MODEL_KEY = os.environ.get('MODEL_KEY', 'models/v2.1/model_artifact.json')


def load_model():
    """Load model artifact from S3 (cached across warm invocations)."""
    global MODEL
    if MODEL is not None:
        return MODEL
    s3 = boto3.client('s3')
    resp = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    MODEL = json.loads(resp['Body'].read().decode('utf-8'))
    meta = MODEL.get('metadata', {})
    print(f"Model loaded: v{meta.get('version')} "
          f"(enhanced_cold_start={'cold_baselines_enhanced' in MODEL})")
    return MODEL


def model_version(model):
    """Read version from the artifact rather than hardcoding it."""
    return model.get('metadata', {}).get('version', 'unknown')


def _parse_time(timestamp_str):
    ts = datetime.fromisoformat(timestamp_str)
    dow = ts.weekday()          # 0=Monday
    return {
        'hour': ts.hour,
        'dow': dow,
        'month': ts.month,
        'year': ts.year,
        'is_weekend': dow >= 5,
    }


def _uncertainty(forecast, hour, is_cold):
    """Heteroscedastic interval: CV scales inversely with volume level."""
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
        cv *= 1.4          # widen for synthetic-history stations
    if hour <= 4 or hour >= 22:
        cv *= 1.3          # night hours are relatively more variable

    std = forecast * cv
    lower = max(forecast - 1.645 * std, 0.0)
    upper = forecast + 1.645 * std
    return lower, upper, cv


def _reliability(hour, year, month, is_cold, has_enhanced):
    """Factor-based reliability score in [0.10, 0.95]."""
    score = 0.80
    deductions = []

    if is_cold:
        if has_enhanced:
            score -= 0.22
            deductions.append(('cold_start_with_synthetic_profile', -0.22))
        else:
            score -= 0.30
            deductions.append(('cold_start_no_neighbors', -0.30))

    if hour <= 1 or hour >= 22:
        score -= 0.08
        deductions.append(('late_night_hour', -0.08))

    years_ahead = max(year - 2025, 0)
    if years_ahead >= 1:
        score -= 0.10
        deductions.append(('beyond_forecast_horizon', -0.10))
    elif year == 2025 and month >= 10:
        score -= 0.05
        deductions.append(('late_in_forecast_year', -0.05))
    elif year == 2025 and month >= 7:
        score -= 0.03
        deductions.append(('mid_forecast_year', -0.03))

    if 7 <= hour <= 9 or 16 <= hour <= 18:
        score += 0.05
        deductions.append(('peak_hour_regular_pattern', +0.05))

    return max(0.10, min(0.95, score)), deductions


def predict_detailed(station, direction, timestamp_str):
    """
    Generate a forecast plus a full factor decomposition.

    Returns a dict with the prediction AND the explanation of how it was built.
    """
    model = load_model()
    t = _parse_time(timestamp_str)
    hour, dow, month, year, wknd = t['hour'], t['dow'], t['month'], t['year'], t['is_weekend']

    cold_set = set(model.get('cold_stations', []))
    is_cold = station in cold_set

    factors = []

    if not is_cold:
        # ---- Known station: observed data ----
        path = 'known_station'
        baseline = model['baselines'].get(f"('{station}', '{direction}')")
        if baseline is None:
            return None, f"Unknown station/direction combination: {station}/{direction}"

        hf = model['hour_factors'].get(f"({hour}, {wknd})", 1.0)
        df = model['dow_factors'].get(str(dow), 1.0)
        mf = model['month_factors'].get(str(month), 1.0)
        sha = model['station_hour_adj'].get(f"('{station}', '{direction}', {hour}, {wknd})", 1.0)
        sma = model['station_month_adj'].get(f"('{station}', '{direction}', {month})", 1.0)
        trend = model['per_station_trend'].get(station, model['global_trend']) if year >= 2025 else 1.0

        factors = [
            ('station_baseline', baseline, f"Observed average volume for {station} dir {direction}"),
            ('hour_of_day', hf, f"Hour {hour} on a {'weekend' if wknd else 'weekday'} vs network average"),
            ('day_of_week', df, f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]} vs weekly average"),
            ('month_of_year', mf, f"Month {month} seasonal effect"),
            ('station_hour_adjustment', sha, "This station's own hourly shape vs the network's"),
            ('station_month_adjustment', sma, "This station's own seasonal shape vs the network's"),
            ('year_over_year_trend', trend, "Observed traffic trend applied to 2025+ forecasts"),
        ]
        has_enhanced = False

    else:
        # ---- Cold-start station ----
        b_key = f"('{station}', '{direction}')"
        enh_baseline = model.get('cold_baselines_enhanced', {}).get(b_key)
        has_enhanced = enh_baseline is not None

        if has_enhanced:
            path = 'cold_start_enhanced'
            baseline = enh_baseline
            hf = model['cold_hour_factors_enhanced'].get(
                f"('{station}', '{direction}', {hour}, {wknd})",
                model['hour_factors'].get(f"({hour}, {wknd})", 1.0))
            df = model['cold_dow_factors_enhanced'].get(
                f"('{station}', '{direction}', {dow})",
                model['dow_factors'].get(str(dow), 1.0))
            mf = model['cold_month_factors_enhanced'].get(
                f"('{station}', '{direction}', {month})",
                model['month_factors'].get(str(month), 1.0))
            trend = model['cold_trends_enhanced'].get(station, model['global_trend']) if year >= 2025 else 1.0
            src = "synthetic history reconstructed from neighboring stations"
        else:
            path = 'cold_start_fallback'
            baseline = model['baselines'].get(b_key, 500.0)
            hf = model['hour_factors'].get(f"({hour}, {wknd})", 1.0)
            df = model['dow_factors'].get(str(dow), 1.0)
            mf = model['month_factors'].get(str(month), 1.0)
            trend = model['global_trend'] if year >= 2025 else 1.0
            src = "road-feature regression (no connected neighbors available)"

        sha = sma = 1.0  # folded into the cold-start profiles
        factors = [
            ('station_baseline', baseline, f"Estimated volume level from {src}"),
            ('hour_of_day', hf, f"Hour {hour} on a {'weekend' if wknd else 'weekday'} ({'station-specific' if has_enhanced else 'network-average'} profile)"),
            ('day_of_week', df, f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]} effect"),
            ('month_of_year', mf, f"Month {month} seasonal effect"),
            ('year_over_year_trend', trend, "Trend inherited from neighboring stations" if has_enhanced else "Network-wide average trend"),
        ]

    # Multiplicative combination with a running product for auditability
    forecast = baseline
    steps = []
    running = baseline
    for i, (name, value, desc) in enumerate(factors):
        if i == 0:
            steps.append({'factor': name, 'value': round(value, 4),
                          'running_total': round(running, 1), 'meaning': desc})
            continue
        running *= value
        steps.append({'factor': name, 'value': round(value, 4),
                      'running_total': round(running, 1), 'meaning': desc})

    forecast = max(running, 0.0)
    lower, upper, cv = _uncertainty(forecast, hour, is_cold)
    reliability, rel_factors = _reliability(hour, year, month, is_cold, has_enhanced)

    result = {
        'station': station,
        'direction': direction,
        'timestamp': timestamp_str,
        'forecast_volume': round(forecast, 1),
        'lower_90': round(lower, 1),
        'upper_90': round(upper, 1),
        'reliability_score': round(reliability, 3),
        'model_version': model_version(model),
    }

    explanation = {
        'prediction_path': path,
        'is_cold_start': is_cold,
        'decomposition': steps,
        'formula': ' x '.join(f['factor'] for f in
                              [{'factor': n} for n, _, _ in factors]),
        'uncertainty': {
            'coefficient_of_variation': round(cv, 3),
            'interval_method': '90% interval = forecast +/- 1.645 * (forecast * CV)',
            'lower_90': round(lower, 1),
            'upper_90': round(upper, 1),
        },
        'reliability': {
            'base_score': 0.80,
            'adjustments': [{'reason': r, 'delta': d} for r, d in rel_factors],
            'final_score': round(reliability, 3),
        },
        'privacy_note': 'Station identifiers are anonymous. Data is aggregate hourly '
                        'vehicle counts only - no individual vehicles or persons are tracked.',
    }
    return {'prediction': result, 'explanation': explanation}, None


def predict(station, direction, timestamp_str):
    """Thin wrapper returning just the prediction fields."""
    out, err = predict_detailed(station, direction, timestamp_str)
    if err:
        return None, err
    return out['prediction'], None


def lambda_handler(event, context):
    try:
        path = event.get('path', '/')
        params = event.get('queryStringParameters') or {}
        model = load_model()

        if path in ('/', '/health'):
            return response(200, {
                'status': 'healthy',
                'service': 'deldot-traffic-forecast',
                'model_version': model_version(model),
                'enhanced_cold_start': 'cold_baselines_enhanced' in model,
                'endpoints': ['/health', '/forecast', '/explain'],
            })

        station = params.get('station')
        direction = params.get('direction', '1')
        date_str = params.get('date')
        hour = params.get('hour')

        if path in ('/forecast', '/explain'):
            if not station:
                return response(400, {'error': 'Missing required parameter: station'})
            if not date_str:
                return response(400, {'error': 'Missing required parameter: date (YYYY-MM-DD)'})

        if path == '/explain':
            if hour is None:
                return response(400, {'error': '/explain requires an hour parameter (0-23)'})
            out, err = predict_detailed(station, direction, f"{date_str}T{int(hour):02d}:00:00")
            if err:
                return response(404, {'error': err})
            return response(200, out)

        if path == '/forecast':
            if hour is not None:
                result, err = predict(station, direction, f"{date_str}T{int(hour):02d}:00:00")
                if err:
                    return response(404, {'error': err})
                return response(200, result)

            forecasts = []
            for h in range(24):
                result, err = predict(station, direction, f"{date_str}T{h:02d}:00:00")
                if err:
                    return response(404, {'error': err})
                result['hour'] = h
                forecasts.append(result)
            return response(200, {
                'station': station,
                'direction': direction,
                'date': date_str,
                'forecasts': forecasts,
                'model_version': model_version(model),
            })

        return response(404, {'error': f'Unknown path: {path}'})

    except Exception as e:
        print(f"ERROR: {e}")
        return response(500, {'error': str(e)})


def response(status_code, body):
    return {
        'statusCode': status_code,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(body),
    }
