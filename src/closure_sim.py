"""
Road Closure Impact Simulator
==============================
Estimates what a planned closure does to the closed station AND to its
neighbours, by combining the volume forecast with a capacity proxy and a
diversion model over the supplied station graph.

Answers the question a DOT actually worries about: not "how many drivers are
affected" but "what breaks downstream".

    Closing 2 of 3 lanes at STN_0053 diverts ~1,100 veh/hr at peak. Weighted
    across its neighbours, STN_0066 absorbs ~480, taking it from 61% to 87% of
    estimated capacity -- the binding constraint.

WHAT THIS IS NOT
----------------
This is not a traffic assignment model. Proper assignment needs an
origin-destination matrix, measured link capacities and route-choice modelling;
none of those are available in the supplied data. The station graph is adjacency
only -- 186 edges labelled `same_corridor` or `proximity` with coarse distance
bands, no flow direction and no routable topology.

Outputs are therefore analytical estimates for comparing options, not
engineering-grade operational, safety or capacity recommendations.

THE TWO ASSUMPTIONS THAT MATTER
-------------------------------
1. Capacity proxy. Derived from observed throughput: for each station,
   per-lane capacity is the larger of (a) the 90th percentile of observed
   per-lane peaks across stations of the same functional class and (b) the
   station's own observed peak per lane. A road can evidently carry at least
   what it has already carried. Observed peaks are demand-limited, so this
   understates true capacity -- conservative in the safe direction. Override
   with real values via `capacity_per_lane` when they are available.

2. Diversion rate. What fraction of turned-away traffic reroutes rather than
   not travelling at all. This cannot be calibrated here: the data contains no
   closure event. It is an explicit input, defaulting to 0.7, and every result
   is reported across a sensitivity range so the reader can see how much the
   conclusion depends on it.
"""

from datetime import datetime, timedelta

DEFAULT_DIVERSION_RATE = 0.70
DEFAULT_SENSITIVITY = (0.50, 0.70, 0.90)

# A neighbour at 85% of capacity is treated as approaching its limit; over 100%
# it cannot absorb the traffic and a queue forms there too.
UTIL_WARN = 0.85
UTIL_OVER = 1.00

MAX_DURATION_HOURS = 24 * 14


# ----------------------------------------------------------------------
# Capacity model (built offline, baked into the deployed artifact)
# ----------------------------------------------------------------------
def build_capacity_model(observed, network):
    """
    Derive a per-lane capacity proxy for every station.

    Args:
        observed: DataFrame with station_key, direction_code, volume
        network:  DataFrame with station_key, through_lanes, functional_class

    Returns:
        (capacity_per_lane: dict, station_lanes: dict)
    """
    import numpy as np

    peaks = (observed.groupby(['station_key', 'direction_code'])['volume']
             .quantile(0.999).reset_index(name='peak'))
    peaks = peaks.merge(
        network[['station_key', 'through_lanes', 'functional_class']],
        on='station_key', how='left')
    peaks['through_lanes'] = peaks['through_lanes'].fillna(2).clip(lower=1)
    peaks['per_lane'] = peaks['peak'] / peaks['through_lanes']
    peaks['functional_class'] = peaks['functional_class'].fillna('Unknown')

    class_p90 = peaks.groupby('functional_class')['per_lane'].quantile(0.90).to_dict()
    global_p90 = float(peaks['per_lane'].quantile(0.90))

    own_peak = peaks.groupby('station_key')['per_lane'].max().to_dict()
    lanes = (network.set_index('station_key')['through_lanes']
             .fillna(2).clip(lower=1).to_dict())
    cls = network.set_index('station_key')['functional_class'].fillna('Unknown').to_dict()

    capacity = {}
    for stn in network['station_key']:
        by_class = class_p90.get(cls.get(stn, 'Unknown'), global_p90)
        capacity[stn] = float(max(by_class, own_peak.get(stn, 0.0)))
    return capacity, {k: float(v) for k, v in lanes.items()}


def build_neighbor_weights(edges):
    """
    Weighted adjacency for diversion. Same weighting used for cold-start
    reconstruction: a same-corridor neighbour is a more plausible alternative
    route than a merely nearby one, and closer is more plausible than farther.
    """
    weights = {}
    for _, r in edges.iterrows():
        w = 2.0 if r['edge_type'] == 'same_corridor' else 1.0
        band = r['distance_band']
        w *= 3.0 if band == '<=1mi' else (2.0 if band == '1-5mi' else 1.0)
        weights.setdefault(r['from_station_key'], []).append([r['to_station_key'], w])
        weights.setdefault(r['to_station_key'], []).append([r['from_station_key'], w])
    return weights


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
def simulate_closure(predict_fn, station, direction, start, duration_hours,
                     capacity_per_lane, station_lanes, neighbor_weights,
                     lanes_closed=None, diversion_rate=DEFAULT_DIVERSION_RATE,
                     sensitivity=DEFAULT_SENSITIVITY, known_stations=None):
    """
    Simulate a closure and its knock-on effect on neighbouring stations.

    Args:
        predict_fn(station, direction, iso_ts) -> forecast volume
        start: ISO date or datetime string for the closure start
        duration_hours: length of the closure
        lanes_closed: lanes taken out of service; defaults to all of them
        diversion_rate: fraction of turned-away traffic that reroutes
        sensitivity: diversion rates to also report

    Returns a dict describing the closed station, each affected neighbour, the
    binding constraint, and a sensitivity table.
    """
    duration_hours = int(duration_hours)
    if not (1 <= duration_hours <= MAX_DURATION_HOURS):
        raise ValueError(f"duration_hours must be between 1 and {MAX_DURATION_HOURS}")
    if not (0.0 <= float(diversion_rate) <= 1.0):
        raise ValueError("diversion_rate must be between 0 and 1")

    try:
        t0 = datetime.fromisoformat(str(start)[:19]) if 'T' in str(start) \
            else datetime.fromisoformat(str(start)[:10])
    except ValueError:
        raise ValueError(f"start must be an ISO date or datetime, got {start!r}")

    lanes = station_lanes.get(station)
    if lanes is None:
        raise ValueError(f"unknown station: {station}")
    lanes = int(lanes)
    lanes_closed = lanes if lanes_closed is None else int(lanes_closed)
    if not (1 <= lanes_closed <= lanes):
        raise ValueError(f"lanes_closed must be between 1 and {lanes} for {station}")

    per_lane = capacity_per_lane.get(station, 0.0)
    cap_before = per_lane * lanes
    lanes_open = lanes - lanes_closed
    cap_during = per_lane * lanes_open

    # --- neighbours eligible to receive diverted traffic ---
    raw = neighbor_weights.get(station, [])
    nbrs = [(n, w) for n, w in raw
            if n != station and (known_stations is None or n in known_stations)]
    total_w = sum(w for _, w in nbrs)

    # --- hour by hour over the closure ---
    hours = []
    nbr_recv = {n: 0.0 for n, _ in nbrs}
    nbr_peak_recv = {n: 0.0 for n, _ in nbrs}
    for k in range(duration_hours):
        ts = (t0 + timedelta(hours=k)).strftime('%Y-%m-%dT%H:00:00')
        demand = float(predict_fn(station, direction, ts))
        throughput = min(demand, cap_during)
        unserved = max(demand - throughput, 0.0)
        diverted = unserved * float(diversion_rate)
        hours.append({
            'timestamp': ts,
            'demand': round(demand, 1),
            'throughput': round(throughput, 1),
            'unserved': round(unserved, 1),
            'diverted': round(diverted, 1),
            'suppressed': round(unserved - diverted, 1),
            'over_capacity': bool(demand > cap_during),
        })
        if total_w > 0 and diverted > 0:
            for n, w in nbrs:
                share = diverted * (w / total_w)
                nbr_recv[n] += share
                nbr_peak_recv[n] = max(nbr_peak_recv[n], share)

    tot_demand = sum(h['demand'] for h in hours)
    tot_through = sum(h['throughput'] for h in hours)
    tot_unserved = sum(h['unserved'] for h in hours)
    tot_diverted = sum(h['diverted'] for h in hours)
    queue_hours = sum(1 for h in hours if h['over_capacity'])

    # --- neighbour utilisation at the closure's peak diversion hour ---
    peak_hour = max(hours, key=lambda h: h['diverted']) if hours else None
    neighbours = []
    floored = []
    for n, w in sorted(nbrs, key=lambda x: -x[1]):
        n_lanes = int(station_lanes.get(n, 2))
        n_cap_raw = capacity_per_lane.get(n, 0.0) * n_lanes
        base = float(predict_fn(n, direction, peak_hour['timestamp'])) if peak_hour else 0.0

        # A station's capacity is at least what we forecast it carries. Where the
        # proxy falls below the forecast the proxy is wrong, not the road
        # overloaded -- this happens for cold-start stations, which have no
        # observed peak to floor the class estimate and whose volume was
        # reconstructed from neighbours that may be of a different road class.
        # Report it rather than emitting a >100% baseline utilisation.
        capacity_basis = 'observed_proxy'
        n_cap = n_cap_raw
        if n_cap_raw > 0 and base > n_cap_raw:
            n_cap = base
            capacity_basis = 'floored_to_forecast'
            floored.append(n)

        recv = nbr_peak_recv.get(n, 0.0)
        util_b = (base / n_cap) if n_cap else 0.0
        util_a = ((base + recv) / n_cap) if n_cap else 0.0
        status = ('over_capacity' if util_a > UTIL_OVER
                  else 'approaching_capacity' if util_a >= UTIL_WARN
                  else 'within_capacity')
        neighbours.append({
            'station': n,
            'diversion_weight': round(w / total_w, 3) if total_w else 0.0,
            'lanes': n_lanes,
            'estimated_capacity': round(n_cap, 1),
            'capacity_basis': capacity_basis,
            'baseline_at_peak': round(base, 1),
            'received_at_peak': round(recv, 1),
            'total_received_over_closure': round(nbr_recv.get(n, 0.0), 1),
            'utilisation_before': round(util_b, 3),
            'utilisation_after': round(util_a, 3),
            'status': status,
        })

    # Prefer a binding constraint whose capacity we actually trust; fall back to
    # the highest utilisation overall if every candidate had to be floored.
    trusted = [x for x in neighbours if x['capacity_basis'] == 'observed_proxy']
    pool = trusted or neighbours
    binding = max(pool, key=lambda x: x['utilisation_after']) if pool else None

    # --- sensitivity on the diversion assumption ---
    sens = []
    for rate in sensitivity:
        rows = []
        for n, w in nbrs:
            n_lanes = int(station_lanes.get(n, 2))
            n_cap = capacity_per_lane.get(n, 0.0) * n_lanes
            base = float(predict_fn(n, direction, peak_hour['timestamp'])) if peak_hour else 0.0
            recv = (peak_hour['unserved'] * rate * (w / total_w)) if (peak_hour and total_w) else 0.0
            rows.append(((base + recv) / n_cap) if n_cap else 0.0)
        sens.append({
            'diversion_rate': rate,
            'total_diverted': round(tot_unserved * rate, 1),
            'max_neighbour_utilisation': round(max(rows), 3) if rows else None,
        })

    return {
        'question': 'closure_impact',
        'closed_station': {
            'station': station,
            'direction': direction,
            'lanes_total': lanes,
            'lanes_closed': lanes_closed,
            'lanes_open': lanes_open,
            'capacity_per_lane': round(per_lane, 1),
            'capacity_normal': round(cap_before, 1),
            'capacity_during_closure': round(cap_during, 1),
        },
        'window': {
            'start': hours[0]['timestamp'] if hours else None,
            'end': hours[-1]['timestamp'] if hours else None,
            'duration_hours': duration_hours,
        },
        'totals': {
            'demand': round(tot_demand, 1),
            'served': round(tot_through, 1),
            'turned_away': round(tot_unserved, 1),
            'diverted': round(tot_diverted, 1),
            'suppressed': round(tot_unserved - tot_diverted, 1),
            'hours_over_capacity': queue_hours,
        },
        'peak_diversion_hour': peak_hour,
        'neighbour_impacts': neighbours,
        'binding_constraint': binding,
        'sensitivity': sens,
        'capacity_caveats': {
            'stations_floored_to_forecast': floored,
            'note': ('For these stations the capacity proxy fell below the '
                     'forecast, so capacity was floored to the forecast and '
                     'their utilisation is a lower bound. This affects '
                     'cold-start stations, which have no observed peak and whose '
                     'volume was reconstructed from neighbours possibly of a '
                     'different road class.') if floored else None,
        },
        'assumptions': {
            'diversion_rate': float(diversion_rate),
            'capacity_basis': 'Observed throughput proxy: per-lane capacity is the '
                              'greater of the class 90th percentile of observed '
                              'per-lane peaks and the station\'s own observed peak. '
                              'Understates true capacity because observed peaks are '
                              'demand-limited.',
            'diversion_basis': 'Turned-away traffic is split between rerouting and '
                               'not travelling. Distributed across graph neighbours '
                               'weighted by edge type and distance band. No closure '
                               'event exists in the data to calibrate this, so a '
                               'sensitivity range is reported.',
            'not_modelled': ['origin-destination flows', 'route choice',
                             'signal retiming', 'queue spillback between stations',
                             'time-of-day rerouting behaviour'],
        },
        'basis': 'Analytical estimate for comparing closure options. Not an '
                 'engineering-grade operational, safety or capacity recommendation.',
    }
