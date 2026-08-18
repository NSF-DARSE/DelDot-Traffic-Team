"""
Build the deployable model artifact for AWS Lambda
===================================================
Serialises the fitted multiplicative model plus the enhanced cold-start profiles
into a single JSON file that both Lambda functions load from S3.

JSON is used deliberately instead of pickle: no library-version coupling, no
arbitrary-code-execution risk on load, and the artifact stays human-inspectable.

Tuple keys are stringified (e.g. "('STN_0067', '1')") because JSON object keys
must be strings; the Lambda handlers rebuild the same key strings when looking
values up.

Usage:
    python src/cold_start_enhance.py      # must run first
    python src/build_artifact.py [version]

Output: output/model_artifact_v<version>.json
"""

import json
import sys
from pathlib import Path

from forecast_model import (
    load_and_align_data,
    compute_station_baselines,
    compute_temporal_profiles,
    compute_station_adjustments,
    compute_trend,
    HOUR_OFFSET,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'

DESCRIPTION = ('Multiplicative traffic forecasting model with enhanced cold-start '
               'profiles built from synthetic neighbour history')


def stringify(d):
    """JSON object keys must be strings; tuple keys become their repr."""
    return {str(k): v for k, v in d.items()}


def main(version='2.2'):
    print("=" * 68)
    print(f"  BUILD MODEL ARTIFACT v{version}")
    print("=" * 68)

    train, val, network, edges, submission = load_and_align_data()
    baselines, cold_stations = compute_station_baselines(train, val, network, edges)
    hour_f, dow_f, month_f = compute_temporal_profiles(train, val)
    sh_adj, sm_adj = compute_station_adjustments(train, hour_f, dow_f, month_f)
    per_station_trend, global_trend = compute_trend(train, val)

    cold_path = OUTPUT_DIR / 'cold_start_enhanced.json'
    if not cold_path.exists():
        sys.exit(f"Missing {cold_path}. Run: python src/cold_start_enhance.py")
    cold = json.loads(cold_path.read_text())

    artifact = {
        # Base model
        'baselines': stringify(baselines),
        'hour_factors': stringify(hour_f),
        'dow_factors': stringify(dow_f),
        'month_factors': stringify(month_f),
        'station_hour_adj': stringify(sh_adj),
        'station_month_adj': stringify(sm_adj),
        'per_station_trend': per_station_trend,
        'global_trend': global_trend,
        'cold_stations': sorted(cold_stations),
        # Enhanced cold-start (keys already stringified upstream)
        'cold_baselines_enhanced': cold['cold_baselines'],
        'cold_hour_factors_enhanced': cold['cold_hour_factors'],
        'cold_month_factors_enhanced': cold['cold_month_factors'],
        'cold_dow_factors_enhanced': cold['cold_dow_factors'],
        'cold_trends_enhanced': cold['cold_trends'],
        'metadata': {
            'version': version,
            'hour_offset': HOUR_OFFSET,
            'train_rows': int(len(train)),
            'train_stations': int(train['station_key'].nunique()),
            'train_hours_covered': sorted(int(h) for h in train['actual_hour'].unique()),
            'cold_start_stations': len(cold_stations),
            'global_trend': global_trend,
            'description': DESCRIPTION,
        },
    }

    out = OUTPUT_DIR / f'model_artifact_v{version}.json'
    out.write_text(json.dumps(artifact))

    print(f"\n  wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    for k, v in artifact.items():
        if isinstance(v, dict) and k != 'metadata':
            print(f"    {k:32s} n={len(v)}")
    print(f"\n  metadata: hour_offset={HOUR_OFFSET}, global_trend={global_trend:.4f}, "
          f"train_rows={len(train):,}")
    return out


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '2.2')
