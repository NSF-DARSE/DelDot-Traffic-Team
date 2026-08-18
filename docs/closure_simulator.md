# Road Closure Impact Simulator

Estimates what a planned closure does to the closed station **and to its
neighbours**, by combining the volume forecast with a capacity proxy and a
diversion model over the supplied station graph.

Built 2026-08-18. Live at `/simulate-closure`, and reachable in natural language
through `/ask`.

---

## The question it answers

`/best-window` already answers *when* to close a road with least impact. This
answers the question a DOT actually worries about: **what breaks downstream.**

> Closing two of three lanes at STN_0053 for 72 hours from 30 August turns away
> 36,880 vehicles — 25,815 divert, 11,065 don't travel — and the road runs over
> capacity for 38 of those hours. The binding constraint is STN_0066, which rises
> from 76.3% to 86.2% of estimated capacity. The other four neighbours stay below
> 56%.

## What it is not

**Not a traffic assignment model.** Proper assignment needs an
origin-destination matrix, measured link capacities and route-choice modelling.
None exist in the supplied data. The station graph is adjacency only — 186 edges
labelled `same_corridor` or `proximity` with coarse distance bands, no flow
direction and no routable topology.

Outputs are analytical estimates for comparing options, consistent with the
constraint the case study places on scenario results.

## Usage

```bash
# Deterministic
GET /simulate-closure?station=STN_0053&direction=1&start=2026-08-30
    &duration_hours=72&lanes_closed=2&diversion_rate=0.7

# Natural language (same computation underneath)
GET /ask?q=what happens if I close two of three lanes at STN_0053 for 72 hours
         starting August 30 2026
```

`lanes_closed` defaults to a full closure. `diversion_rate` defaults to 0.70.

## How it works

For each hour of the closure:

```
demand      = forecast volume at the closed station
capacity    = (lanes_total - lanes_closed) x capacity_per_lane
throughput  = min(demand, capacity)
turned_away = demand - throughput
diverted    = turned_away x diversion_rate
suppressed  = turned_away x (1 - diversion_rate)
```

Diverted volume is distributed across graph neighbours weighted by edge type
(`same_corridor` ×2, `proximity` ×1) and distance band (`≤1mi` ×3, `1-5mi` ×2,
`5-15mi` ×1) — the same weighting used for cold-start reconstruction, on the same
reasoning that a same-corridor neighbour is a more plausible alternative route
than a merely nearby one.

Each neighbour's utilisation is then reported before and after, and flagged
`within_capacity` (<85%), `approaching_capacity` (85–100%) or `over_capacity`
(>100%). The **binding constraint** is the highest post-diversion utilisation
among neighbours whose capacity estimate is trusted.

## The two assumptions that matter

### Capacity is derived from observed throughput, not textbook values

Per-lane capacity for each station is the greater of:

1. the 90th percentile of observed per-lane peaks across stations of the same
   functional class, and
2. that station's own observed peak per lane — a road can evidently carry at
   least what it has already carried.

Calibrated from the data rather than asserted. Observed per-lane peaks vary
roughly threefold by road class, which a single textbook constant would miss:

| Functional class | Median observed peak per lane |
|---|---:|
| Local | 1,552 |
| Other Expressways & Freeway | 1,496 |
| Interstate | 1,419 |
| Minor Collector | 757 |
| Other Principal Arterials | 646 |
| Major Collector | 475 |
| Minor Arterial | 481 |

This also settled a data question: `through_lanes` is **per direction**, not
total. STN_0108 records 8,771 veh/hr in one direction across 5 lanes, which is
only coherent at 1,754 per lane.

Because observed peaks are demand-limited, this understates true capacity —
conservative in the safe direction. Override with `capacity_per_lane` when real
values are available.

### The diversion rate cannot be calibrated here

What fraction of turned-away traffic reroutes rather than not travelling at all
is an assumption. There is no closure event anywhere in the data to fit it
against.

So it is an explicit input, and every result reports a sensitivity range:

| Diversion rate | Max neighbour utilisation (STN_0066 closure) |
|---:|---:|
| 50% | 89% |
| 70% | 95% |
| 90% | 102% |

That range is the honest answer: at the low end the network absorbs the closure,
at the high end a neighbour tips over capacity. The conclusion depends on the
assumption, and the output says so rather than hiding it behind a single number.

## A model finding this surfaced

Building the simulator exposed a genuine inconsistency in the cold-start
reconstruction, unrelated to closures.

`STN_0068` is a cold-start station: a **2-lane Other Principal Arterial**, so its
capacity proxy is about 1,835 veh/hr. But its volume was reconstructed from
neighbours that include **Interstates**, giving it a forecast peak near 4,400 —
roughly 240% of what its own geometry can carry.

The reconstruction is volume-weighted by neighbour proximity but **not
capacity-aware**: a 2-lane arterial can inherit a motorway's volume level. 52 of
127 stations are cold-start and therefore exposed to this.

Handled two ways:

- **In the simulator**, capacity is floored to the forecast where the proxy falls
  below it. Utilisation is then reported as a lower bound with
  `capacity_basis: floored_to_forecast`, and such stations are excluded from
  binding-constraint selection in favour of stations whose capacity is trusted.
  Before this fix STN_0068 reported 162% utilisation *before any closure*, which
  is impossible and would have been the headline number.
- **In the model**, left unchanged. Making the cold-start reconstruction
  capacity-aware would alter the submission, and it is recorded here as
  follow-up work rather than changed the day before a deadline.

## Guardrail interaction

Closure narration goes through the same numeric-containment check as every other
generated sentence. Building it required two precision fixes to the allow-list,
both cases where the model was correct and the guardrail was wrong:

1. **Ratios expressed as percentages.** A utilisation of `0.862` is naturally
   written "86.2%". Now permitted for any ratio value up to 10.0, which also
   covers over-capacity readings such as `1.018` → "101.8%". This only ever
   licences ×100 of a value that genuinely appears in the result.
2. **Range hyphens read as minus signs.** `"389-1188"` and `"08:00-20:00"` were
   parsed as containing `-1188` and `-20`, causing spurious rejections. A leading
   minus now counts only when not preceded by a digit or word character.

Re-verified after both changes that fabricated volumes, invented percentages and
impossible utilisations are still rejected.

## Output structure

| Field | Contents |
|---|---|
| `closed_station` | lanes closed/open, capacity before and during |
| `totals` | demand, served, turned away, diverted, suppressed, hours over capacity |
| `peak_diversion_hour` | the worst hour, with its demand and unserved volume |
| `neighbour_impacts` | per neighbour: weight, capacity, utilisation before/after, status |
| `binding_constraint` | the neighbour that limits the plan |
| `sensitivity` | max neighbour utilisation across diversion rates |
| `capacity_caveats` | stations whose capacity was floored to forecast |
| `assumptions` | diversion rate, capacity basis, and what is explicitly not modelled |

## Scenario relevance

Scenarios 1, 4 and 5 derived their percentages from capacity reasoning done by
hand. Those figures can now be computed instead, with the diversion assumption
made explicit and its sensitivity reported. The submitted
`scenario_template_filled.csv` was left unchanged for this deadline; recomputing
it from the simulator is the natural next step.
