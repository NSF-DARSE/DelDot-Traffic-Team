# Scenario Analysis — Methodology & Results

## Overview

Five operational scenarios were analyzed to estimate traffic volume impacts,
uncertainty bounds, recovery times, and reliability for common DOT situations.
Stations were selected based on their characteristics and relevance to each
scenario type.

These are **analytical estimates**, not engineering-grade operational predictions.

---

## Scenario Summaries

| ID | Scenario | Station | Impact | Recovery | Reliability |
|----|----------|---------|--------|----------|-------------|
| 01 | Major road closure | STN_0108 | -27.5% [-42.5%, -17.5%] | 48h | 0.65 |
| 02 | Special event surge | STN_0037 | +40.2% [+25.2%, +60.2%] | 6h | 0.55 |
| 03 | Severe winter storm | STN_0066 | -45.0% [-60.0%, -30.0%] | 36h | 0.60 |
| 04 | Construction zone | STN_0053 | -15.0% [-22.0%, -8.0%] | 168h (1 wk) | 0.70 |
| 05 | Corridor cascade | STN_0114 | -25.0% [-40.0%, -15.0%] | 4h | 0.55 |

---

## Detailed Scenario Analysis

### SCENARIO_01: Major Road Closure (Emergency Bridge Repair)

**Station:** STN_0108 — 5-lane Interstate, 65 mph, 100k+ AADT  
**Normal volume:** 4,872 veh/hr average; 7,112 veh/hr peak

**Event:** Emergency repair requiring closure of 4 of 5 lanes (80% capacity reduction).

**Analysis:**
- During peak hours (6am–6pm): demand exceeds reduced capacity. Traffic queues
  form; approximately 40% of drivers divert to alternate routes. Measured volume
  at the station drops 40%.
- During off-peak hours: demand is below the reduced 1-lane capacity (~1,500 veh/hr).
  Minor speed reduction but most traffic passes through. Volume drops ~15%.
- Weighted average across 24 hours: **-27.5%**

**Recovery:** After lanes reopen:
- Queue clearance: 2–4 hours
- Driver re-routing habits: 1–2 days for all drivers to return from alternates
- Total: **48 hours** to normalize

**Uncertainty:** Wide bounds [-42.5%, -17.5%] reflect dependence on:
- Availability and quality of alternate routes
- Time of day of closure
- Whether diversion information is provided (VMS, apps)

---

### SCENARIO_02: Special Event (Holiday Weekend Traffic Surge)

**Station:** STN_0037 — strongest weekend uplift in network (+18% vs weekday)

**Normal volume:** 294 veh/hr average; weekends average 330 veh/hr

**Event:** Major holiday weekend (July 4th type) or large venue event near this station.

**Analysis:**
- This station already shows the highest weekend uplift (+18%), suggesting
  recreational/event-driven traffic patterns
- Transportation literature indicates major events add 15–30% above normal
  weekend volumes for nearby corridors
- Conservative estimate: 25% above normal weekend level
- Combined effect vs. overall baseline: **+40.2%**

**Recovery:** Event traffic dissipates within **6 hours** of event conclusion.

**Uncertainty:** Wide range [+25.2%, +60.2%] because:
- No event data exists in training set
- Event magnitude depends on type, time, weather
- Lowest reliability of all scenarios (0.55)

---

### SCENARIO_03: Severe Weather (Winter Storm / Ice)

**Station:** STN_0066 — 2-lane Interstate, 55 mph, 50k–100k AADT

**Normal volume:** 1,849 veh/hr average

**Event:** Major winter storm with significant icing.

**Analysis:**
- Transportation weather research consistently shows:
  - Light snow: -10 to -20% volume
  - Heavy snow: -20 to -40% volume
  - Ice storms: -40 to -70% volume
- This station is high-speed (55 mph), 2-lane, likely rural — maximally
  exposed to weather sensitivity
- Major ice event estimate: **-45%** reduction (volume drops to ~1,017 veh/hr)

**Recovery:** 
- Storm duration: ~24 hours
- Road treatment and ice clearing: 6–12 hours post-storm
- Traffic returns to normal: **36 hours** total from storm onset

**Uncertainty:** [-60%, -30%] depends on:
- Storm severity and duration
- Speed of road treatment response
- Whether roads are formally closed vs. advisory conditions

---

### SCENARIO_04: Construction Zone (Long-term Lane Closure)

**Station:** STN_0053 — 3-lane Interstate, 55 mph, 50k–100k AADT  
**Connected corridor:** STN_0066 (1–5 mi), STN_0118 (1–5 mi)

**Normal volume:** 2,237 veh/hr average

**Event:** Long-term construction project (6+ months), one lane permanently closed.

**Analysis:**
- Capacity reduction: 3 lanes → 2 lanes (33% capacity loss)
- Unlike sudden closures, long-term construction allows driver adaptation:
  - Week 1: -20 to -25% (initial shock, route exploration)
  - Weeks 2–4: -15 to -20% (stabilizing)
  - Month 2+: **-15%** (equilibrium — permanent diversion established)
- Connected corridor stations (STN_0066, STN_0118) expected to see +5 to +8%
  increase as they absorb diverted traffic

**Recovery:** After construction ends:
- Driver habits take 1–2 weeks to shift back
- **168 hours** (1 week) to full normalization

**Reliability: 0.70** (highest of all scenarios) — construction impacts are the
most well-studied and predictable because duration is long enough for patterns
to stabilize.

---

### SCENARIO_05: Corridor Disruption (Upstream Cascade / Spillback)

**Corridor:** STN_0064 (downstream, 1,864 avg) → STN_0114 (upstream, 1,900 avg)  
**Distance:** ≤1 mile apart, same corridor

**Event:** Major multi-lane incident on STN_0064 (downstream) during peak hours.

**Analysis:**
- STN_0064 (incident site): volume drops -50 to -70% immediately as lanes blocked
- Queue forms and propagates upstream (shockwave)
- STN_0114 (upstream, <1 mile): throughput reduced as queue reaches this point
- Cascade effect on STN_0114: **-25%** reduction during incident
- This demonstrates the spatial propagation of disruptions through a corridor

**Recovery:**
- Typical major incident duration: 1–3 hours
- Queue dissipation after lanes clear: 30–90 minutes
- **4 hours** total from incident onset to normal flow at upstream station

**Uncertainty:** [-40%, -15%] reflects high variability in:
- Incident severity (fender bender vs. multi-vehicle)
- Number of lanes blocked
- Incident clearance time
- Whether HOV/shoulder is available as bypass

---

## Methodology

### Station Selection
Stations were chosen based on characteristics relevant to each scenario:
- **Road closure:** Highest-volume Interstate (maximum impact)
- **Special event:** Strongest weekend uplift (event-sensitive location)
- **Weather:** High-speed 2-lane Interstate (weather-vulnerable)
- **Construction:** Multi-lane corridor with connected alternates
- **Cascade:** Same-corridor stations ≤1 mile apart (spillback physics)

### Impact Estimation Approach
1. **Capacity analysis:** Compare demand (from model) to reduced capacity
2. **Diversion modeling:** Estimate fraction that diverts based on alternate availability
3. **Literature calibration:** Cross-reference with published transportation research
   on similar scenarios (HCM, TRB papers on work zones, weather, incidents)
4. **Temporal weighting:** Average across peak/off-peak hours proportionally

### Uncertainty Bounds
- Lower bound: pessimistic scenario (worse conditions, fewer alternatives)
- Upper bound: optimistic scenario (quick response, good alternates)
- Width reflects the inherent unpredictability of the event type

### Recovery Time Estimation
- Based on event duration + clearance time + pattern normalization
- Short events (incidents): 4–6 hours
- Medium events (weather, closures): 36–48 hours
- Long events (construction): 1 week (168 hours)

### Reliability Scores
Reflect confidence in the estimate based on:
- Availability of analogous data (construction > weather > events > incidents)
- Variability of the phenomenon
- Quality of assumptions

| Score Range | Meaning |
|-------------|---------|
| 0.70 | High confidence — well-studied, predictable patterns |
| 0.60–0.65 | Moderate — solid methodology but significant unknowns |
| 0.55 | Lower confidence — high variability, no historical analogues |

---

## Limitations

1. **No historical event/weather data** — scenarios rely on literature, not
   observed events in this dataset
2. **No origin-destination information** — diversion estimates are approximate
3. **Static model** — does not capture real-time queue dynamics or route choice
4. **Single-station reporting** — corridor effects are noted in assumptions but
   only one station value is reported per scenario
5. **No capacity model** — lane capacities are inferred from literature (≈1,800
   veh/hr/lane) rather than calibrated from this dataset
