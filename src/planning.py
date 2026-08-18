"""
Deterministic Planning Tools
=============================
Answers the operational questions a traffic agency actually asks, by ranking
model forecasts. No language model is involved: given the same inputs these
functions always return the same answer, and every figure is reproducible.

    best_hours   -> "when is the quietest time to travel on <date>?"
    best_window  -> "which <N>-hour window has the lowest impact?"

Both take a `predict_fn(station, direction, iso_timestamp) -> float` so they
reuse exactly the same forecasting path as `/forecast` and `/explain`. Numbers
therefore agree across every endpoint.

These functions are the authority for ranking. The conversational layer may only
parse the question and phrase the result; it must never choose the answer.
"""

from datetime import date as _date, datetime, timedelta

MAX_RANGE_DAYS = 31          # guard against unbounded scans
MAX_DURATION_HOURS = 24 * 14


def _parse_date(s, field):
    try:
        return datetime.strptime(str(s)[:10], '%Y-%m-%d').date()
    except (ValueError, TypeError):
        raise ValueError(f"{field} must be a date in YYYY-MM-DD form, got {s!r}")


def _hourly_series(predict_fn, station, direction, start, end):
    """Forecast every hour in [start, end] inclusive of both dates."""
    series = []
    day = start
    while day <= end:
        for hour in range(24):
            ts = f"{day.isoformat()}T{hour:02d}:00:00"
            series.append({
                'timestamp': ts,
                'date': day.isoformat(),
                'hour': hour,
                'day_of_week': day.strftime('%A'),
                'forecast_volume': round(float(predict_fn(station, direction, ts)), 1),
            })
        day += timedelta(days=1)
    return series


# ----------------------------------------------------------------------
# best_hours — quietest hours on a single date
# ----------------------------------------------------------------------
def best_hours(predict_fn, station, direction, date, top=3,
               earliest=6, latest=22):
    """
    Rank hours on `date` by forecast volume, ascending.

    `earliest`/`latest` bound the search because the unconstrained answer is
    always the middle of the night, which is true but operationally useless for
    a travel question. Defaults to 06:00-22:00; pass earliest=0, latest=23 to
    consider the whole day.
    """
    d = _parse_date(date, 'date')
    earliest, latest = int(earliest), int(latest)
    if not (0 <= earliest <= 23 and 0 <= latest <= 23):
        raise ValueError("earliest and latest must be hours between 0 and 23")
    if earliest > latest:
        raise ValueError("earliest must not be after latest")
    top = max(1, min(int(top), 24))

    series = _hourly_series(predict_fn, station, direction, d, d)
    considered = [r for r in series if earliest <= r['hour'] <= latest]
    if not considered:
        raise ValueError("no hours fall inside the requested window")

    ranked = sorted(considered, key=lambda r: r['forecast_volume'])
    peak = max(series, key=lambda r: r['forecast_volume'])
    quietest = ranked[0]

    return {
        'question': 'quietest_hours',
        'station': station,
        'direction': direction,
        'date': d.isoformat(),
        'day_of_week': d.strftime('%A'),
        'search_window': f"{earliest:02d}:00-{latest:02d}:00",
        'quietest_hours': [
            {'hour': r['hour'], 'time': f"{r['hour']:02d}:00",
             'forecast_volume': r['forecast_volume']}
            for r in ranked[:top]
        ],
        'busiest_hour_of_day': {
            'hour': peak['hour'], 'time': f"{peak['hour']:02d}:00",
            'forecast_volume': peak['forecast_volume'],
        },
        'saving_vs_peak': round(peak['forecast_volume'] - quietest['forecast_volume'], 1),
        'basis': 'Forecast hourly volumes ranked ascending. Lower volume implies '
                 'lighter traffic, not a guarantee of shorter journey time.',
    }


# ----------------------------------------------------------------------
# best_window — lowest-impact contiguous window for planned work
# ----------------------------------------------------------------------
def best_window(predict_fn, station, direction, start, end,
                duration_hours=24, align='day', top=3):
    """
    Find the contiguous `duration_hours` window in [start, end] carrying the
    least total forecast volume.

    align='day'  -> only windows beginning at midnight (answers "which day?")
    align='hour' -> every hourly start position (answers "which window?")

    Total volume over the window is used as the impact proxy: it approximates
    the number of vehicles affected by a closure of that length.
    """
    s = _parse_date(start, 'start')
    e = _parse_date(end, 'end')
    if e < s:
        raise ValueError("end must not be before start")
    span_days = (e - s).days + 1
    if span_days > MAX_RANGE_DAYS:
        raise ValueError(f"date range is {span_days} days; maximum is {MAX_RANGE_DAYS}")

    duration_hours = int(duration_hours)
    if not (1 <= duration_hours <= MAX_DURATION_HOURS):
        raise ValueError(f"duration_hours must be between 1 and {MAX_DURATION_HOURS}")
    if align not in ('day', 'hour'):
        raise ValueError("align must be 'day' or 'hour'")

    series = _hourly_series(predict_fn, station, direction, s, e)
    n = len(series)
    if duration_hours > n:
        raise ValueError(
            f"duration_hours ({duration_hours}) exceeds the {n} hours in the range")

    step = 24 if align == 'day' else 1
    candidates = []
    for i in range(0, n - duration_hours + 1, step):
        block = series[i:i + duration_hours]
        total = round(sum(r['forecast_volume'] for r in block), 1)
        candidates.append({
            'start': block[0]['timestamp'],
            'end': block[-1]['timestamp'],
            'start_day': block[0]['day_of_week'],
            'duration_hours': duration_hours,
            'affected_vehicles': total,
            'mean_hourly_volume': round(total / duration_hours, 1),
            'peak_hourly_volume': round(max(r['forecast_volume'] for r in block), 1),
        })

    if not candidates:
        raise ValueError("no candidate windows fit the requested duration")

    ranked = sorted(candidates, key=lambda c: c['affected_vehicles'])
    best, worst = ranked[0], ranked[-1]
    top = max(1, min(int(top), len(ranked)))

    return {
        'question': 'lowest_impact_window',
        'station': station,
        'direction': direction,
        'search_start': s.isoformat(),
        'search_end': e.isoformat(),
        'duration_hours': duration_hours,
        'alignment': align,
        'candidates_evaluated': len(candidates),
        'recommended': best,
        'ranked': ranked[:top],
        'worst_choice': worst,
        'avoided_vehicles_vs_worst': round(
            worst['affected_vehicles'] - best['affected_vehicles'], 1),
        'basis': 'Total forecast volume over each contiguous window, ranked '
                 'ascending. An analytical estimate for planning comparison, '
                 'not an engineering-grade operational recommendation.',
    }


# ----------------------------------------------------------------------
# Relative-date helpers, used when resolving natural-language questions
# ----------------------------------------------------------------------
def resolve_relative_range(phrase, today=None):
    """
    Map a few common relative phrases to concrete dates. Returns
    (start, end) as ISO strings, or None if the phrase is not recognised.

    Deliberately small: the language model resolves anything more elaborate,
    and its output is validated against these bounds.
    """
    today = today or _date.today()
    p = (phrase or '').strip().lower()

    if p == 'today':
        return today.isoformat(), today.isoformat()
    if p == 'tomorrow':
        d = today + timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if p in ('this week', 'the rest of this week'):
        end = today + timedelta(days=(6 - today.weekday()))
        return today.isoformat(), end.isoformat()
    if p == 'next week':
        start = today + timedelta(days=(7 - today.weekday()))
        return start.isoformat(), (start + timedelta(days=6)).isoformat()
    if p in ('next month', 'the next month', 'next 30 days'):
        return today.isoformat(), (today + timedelta(days=29)).isoformat()
    if p in ('next fortnight', 'next two weeks'):
        return today.isoformat(), (today + timedelta(days=13)).isoformat()
    return None
