# Traffic Pattern Analysis — Problem & Data Brief

**Purpose of this document.** Self-contained description of the challenge, the supplied data, and
everything measured from it, so that a person or an LLM with no prior exposure can reason about
solution designs. Every number below was computed directly from the supplied files (commands in
Appendix B). Statements that are inference rather than measurement are marked **[hypothesis]**.

Generated 2026-08-17 against the files in `Traffic_Pattern_Analysis/`.

---

## 1. Executive summary

Forecast hourly traffic volume for 170,956 specific (station, direction, timestamp) target cells,
supply a 90% prediction interval and a reliability score for each, answer 5 scenario questions, and
document the method.

Five things dominate the difficulty, in order of impact:

1. **The training file's timestamps are offset by +2 hours relative to validation and the targets.**
   Aligning them cuts baseline MAE from 304 to 130 (−57%). This is the single largest free win in
   the dataset and it is not documented anywhere in the supplied materials.
2. **43 of the 118 target stations have zero labeled history.** They account for 29% of target rows
   and are roughly **2.7× harder** than stations with history (measured: sMAPE 55% vs 25%).
3. **The scored horizon is 6 months beyond the last label** (Jul–Dec 2025), with no covariates
   available for that period other than the calendar and static road attributes.
4. **Missing data is whole-day outages, not scattered hours.** Median 66 missing days per
   station-direction in 2024; 72 of 150 series have an outage run of 30+ consecutive days.
5. **Uncertainty is heteroscedastic and drifts.** Naive residual quantiles deliver only 79–80%
   coverage on a real holdout; they need ~1.6× inflation to hit the promised 90%.

Measured reference points (see §9 for full tables):

| Setting | MAE | sMAPE |
|---|---:|---:|
| Naive profile, timestamps unaligned | 304 | 58% |
| Same profile, timestamps aligned (+2h) | 130 | 25% |
| Aligned + seasonal factor, realistic holdout | 105 | 21% |
| In-sample oracle (upper bound for this feature set) | 108 | 24% |
| Cold-start simulation (no own history) | 357 | 55% |

---

## 2. The task

### 2.1 What must be produced

| Deliverable | Source template | Status |
|---|---|---|
| Forecast file | `submission_template.csv` (170,956 rows) | 4 output columns entirely empty |
| Scenario file | `scenario_template.csv` (5 rows) | entirely empty; **scenario prompts are not in the repo** |
| Presentation / demo | — | free form |
| Reproducible code | — | free form |
| Technical write-up | — | method, validation, assumptions, limitations, operations |

Forecast columns to populate, per row: `forecast_volume`, `lower_90`, `upper_90`,
`reliability_score`. Columns to leave byte-identical: `target_id`, `station_key`, `direction_code`,
`timestamp`.

### 2.2 Scoring weights (from `README.md`)

| Area | Weight |
|---|---:|
| Forecast performance | 35% |
| Scenario analysis | 20% |
| Uncertainty and reliability | 15% |
| Robustness | 15% |
| Reproducibility and operational readiness | 10% |
| Traffic analysis and technical quality | 5% |

**Read this allocation carefully.** Only 35% rewards point accuracy. 50% (uncertainty + robustness +
scenarios) rewards behaviour on the hard slices — cold-start stations, difficult periods, incomplete
observations — and honest interval calibration. A model with slightly worse MAE but well-calibrated
intervals and a defensible cold-start strategy scores higher than a tuned gradient booster with
over-confident intervals.

The README explicitly names the hidden slices: *"limited-history locations, difficult operating
periods, incomplete observations, uncertainty, and supplied scenario questions."* Exact evaluation
rows, slice membership and thresholds are not disclosed. **[hypothesis]** Robustness is likely scored
by re-computing the metric on those subsets, so per-slice error should be tracked during development,
not just the global number.

### 2.3 Hard rules (`rules_and_data_use.md`)

Violations invalidate the submission, not just cost points:

- **Prediction-time information only.** No future observations, target values, or event outcomes
  known only after the forecast timestamp. This forbids random k-fold CV as a validation design —
  splits must be temporal.
- **No de-anonymisation.** `station_key` is authoritative. No external sources, public archives,
  APIs, or pattern matching to identify the underlying stations or recover withheld observations.
  External traffic datasets may not be used to reconstruct identities or targets. (General technical
  references, libraries, transportation research and AI tools are allowed.)
- **Structural integrity of the submission.** Every target row present, no duplicates, no reordering,
  ID columns unaltered.
- **Numeric invariants** (enforced by `validate_submission.py`): all four outputs finite and
  populated; `forecast_volume >= 0`; `lower_90 >= 0`;
  `lower_90 <= forecast_volume <= upper_90`; `0 <= reliability_score <= 1`.
- **Scenario framing.** Results are *analytical estimates*, never engineering-grade operational,
  safety, capacity, or design recommendations. This wording matters in the write-up.
- **Reproducibility.** Document dependencies, preprocessing, model procedure, random seeds.
- Synthetic/augmented data derived from the supplied data is allowed if reproducible.

---

## 3. File inventory

| File | Size | Shape | Notes |
|---|---:|---|---|
| `traffic_train.parquet` | 3.1 MB | 902,902 × 9 | labels, calendar 2024 |
| `traffic_validation.parquet` | 1.8 MB | 556,800 × 9 | labels, Jan–Jun 2025 |
| `submission_template.csv` | 7.5 MB | 170,956 × 8 | the target set |
| `network.csv` | 7.7 KB | 127 × 8 | static station attributes |
| `network_edges.csv` | 6.8 KB | 186 × 4 | station adjacency graph |
| `data_dictionary.csv` | 988 B | 21 field definitions | |
| `scenario_template.csv` | 235 B | 5 × 8 | empty scenario rows |
| `submission_schema.md`, `rules_and_data_use.md`, `README.md` | | | the rules |
| `validate_submission.py` | 5.6 KB | | format gate, exit 0 on pass |
| `quickstart.ipynb` | 2.9 KB | | load-and-join sanity check only |

Total labelled data is small: ~1.46M rows, ~100 MB in pandas memory. Everything fits comfortably in
RAM; there is no big-data problem here, only a structure problem.

**Environment note.** The parquet files need `pyarrow` or `fastparquet`, neither of which is
installed in the local `python3` (which has pandas 3.0.3, numpy 2.4.4, scikit-learn 1.8.0).
`pip install pyarrow` before anything else.

---

## 4. Schemas

### 4.1 The two labelled panels (identical schema)

| Column | dtype | Notes |
|---|---|---|
| `station_key` | str | `STN_0001` … `STN_0127`, opaque |
| `timestamp` | datetime64[us] | hourly, **naive — no timezone** |
| `direction_code` | **str** | `'1'`,`'3'`,`'5'`,`'7'` |
| `direction` | str | North / East / South / West |
| `volume` | float64 | vehicles in the hour; **no NaNs in either file** |
| `hour`, `day_of_week`, `month` | int32 | redundant with `timestamp` — verified consistent |
| `is_weekend` | bool | `day_of_week >= 5` — verified consistent |

Direction encoding: **1 = North, 3 = East, 5 = South, 7 = West**. Every station carries exactly two
directions, always an opposing pair (N/S or E/W). Verified: all 75 labeled and all 118 target
stations have exactly 2 directions.

The derived calendar columns match the timestamp exactly in both files, so they carry no extra
information — but see §5.1, because "matches the timestamp" does not mean "matches reality".

`volume` has **no missing values anywhere**. Missing observations manifest as *absent rows*, never as
null volumes. Any imputation logic keyed on NaN will silently do nothing.

### 4.2 `submission_template.csv`

| Column | dtype on read | Notes |
|---|---|---|
| `target_id` | str | `T000001` … , unique, no duplicates |
| `station_key` | str | 118 distinct |
| `direction_code` | **int64** | ← type mismatch with the parquet files |
| `timestamp` | str | `%Y-%m-%d %H:%M:%S` |
| `forecast_volume`, `lower_90`, `upper_90`, `reliability_score` | float64 | 100% null |

> **Join trap.** `direction_code` is a *string* in the parquets and an *int64* from the CSVs. A merge
> across the two without casting produces zero matches and no error. `validate_submission.py:27`
> normalises by stripping a trailing `.0`, so a float-formatted code passes validation — but a naive
> merge in your own pipeline will not.

### 4.3 `network.csv` — static station context

One row per station, **no direction dimension** (both directions of a station share attributes).
127 stations: 75 labeled, 43 cold targets, and **9 stations that are neither labeled nor scored**
(context/graph filler only).

| Field | Missing | Values |
|---|---:|---|
| `through_lanes` | 0 | 1–5 (mode 2: 91/127) |
| `speed_limit` | 0 | 20–65 mph |
| `lane_width` | 0 | 6–18 ft (mode 12: 94/127) |
| `aadt_band` | 0 | `<5k`, `5k-10k`, `10k-25k`, `25k-50k`, `50k-100k`, `100k+` |
| `roadway_width_band` | 0 | `<24`, `24-35`, `36-47`, `48-59`, `60+` |
| `functional_class` | **35 (27.6%)** | Interstate, Other Expressways & Freeway, Other Principal Arterials, Minor Arterial, Major/Minor Collector, Local |
| `surface_type` | **8 (6.3%)** | Asphalt, Composite, Concrete, Surface Treated |

The vocabulary (functional class names, AADT, mph, lane widths in feet) is FHWA/HPMS terminology, so
this is US state-DOT continuous-count-station data. **[hypothesis]** — but a well-founded one, and it
matters because it justifies using a US federal holiday calendar as a feature (§7.4). Using the
holiday calendar is a general calendar reference, not an attempt to de-anonymise stations.

### 4.4 `network_edges.csv` — station graph

186 undirected edges, all 127 stations connected (no isolated nodes), mean degree 2.9.

| Field | Values |
|---|---|
| `edge_type` | `proximity` (131), `same_corridor` (55) |
| `distance_band` | `<=1mi` (32), `1-5mi` (120), `5-15mi` (34) |

No coordinates are supplied. This graph is the only spatial information available, and it is the
primary lever for the cold-start problem (§6, §8).

---

## 5. Structural findings — the things that are not in the documentation

### 5.1 FINDING 1 — the training file is shifted by +2 hours

**Evidence.**

- `traffic_train.parquet` contains **only hours 0–21**. Each of those 22 hours has exactly 41,041
  rows (22 × 41,041 = 902,902). Hours 22 and 23 are entirely absent. `traffic_validation.parquet`
  has all 24 hours, exactly 23,200 rows each.
- The mean diurnal profiles of the two files do not match at lag 0 but match almost perfectly at
  lag 2:

  | shift applied to train | profile correlation vs validation |
  |---|---:|
  | 0 h | 0.794 |
  | +1 h | 0.939 |
  | **+2 h** | **0.9998** |
  | +3 h | 0.943 |

- The lag-2 alignment holds in **every quarter** (r = 0.991–0.999), in **11 of 12 months**, and is
  the best single lag for **116 of 150** station-direction series (21 pairs prefer lag 1, 13 prefer
  lag 3 — the peak is broad, so per-series noise moves the argmax by one).
- The unshifted train profile peaks at label-hour 14 and bottoms at label-hour 0, which is not a
  physically plausible traffic curve. Validation peaks at 16:00 and bottoms at 02:00, which is.
- Decisive test — a per-series `hour × day-of-week` mean profile learned on 2024 and scored on the
  Jan–Jun 2025 labels:

  | shift | MAE | RMSE | MAPE | sMAPE |
  |---|---:|---:|---:|---:|
  | −1 h | 397.9 | 640.3 | 414.5% | 73.1% |
  | 0 h | 304.4 | 502.4 | 192.3% | 58.4% |
  | +1 h | 203.8 | 350.4 | 75.1% | 40.7% |
  | **+2 h** | **129.6** | **254.5** | **37.7%** | **24.9%** |
  | +3 h | 199.5 | 332.9 | 70.3% | 41.2% |

  For reference, fitting the profile *on the validation data itself* (an in-sample oracle) gives
  MAE 107.6. The +2h-aligned out-of-sample model is within 20% of that ceiling; the unaligned one is
  3× away from it.

**Interpretation.** The 2024 labels were written with a clock offset — most likely a
UTC/local-time or hour-beginning/hour-ending mismatch introduced when the anonymised extract was
built. **[hypothesis]** The truncation at hour 21 is the same artifact: the file covers real clock
hours 02:00–23:59 and real hours 00:00–01:59 of 2024 were dropped when the window was cut.

**Consequences you must handle.**

1. Add `+2h` to every `traffic_train.parquet` timestamp before joining it with validation, with the
   target set, or with any calendar feature. Recompute `hour`, `day_of_week`, `month`, `is_weekend`
   after shifting — the supplied columns are consistent with the *wrong* timestamp.
2. **After the shift, there is no 2024 training data at all for target hours 00:00 and 01:00.**
   That is 8.33% of validation rows (46,400 rows: 23,200 at hour 0 and 23,200 at hour 1) with no
   matching 2024 cell, and it is the entire explanation for the 91.7% coverage figure of a
   `pair × hour × dow` lookup. Validation (Jan–Jun 2025) is the **only** source of night-hour
   behaviour, and the target file samples hours 0 and 1 at the same rate as every other hour
   (≈4.2% each). Plan for it: borrow the hour-0/1 shape from validation, or model hour as a smooth
   cyclical function so those hours are interpolated rather than looked up.
3. **Open ambiguity.** The 2024-dated target rows (§6.3) sit inside the shifted-clock period. It is
   unknown whether the hidden ground truth for those rows uses the validation clock or the train
   clock. They are only 81 rows out of 170,956 (0.05%), so the exposure is negligible — but say so
   in the write-up rather than leaving it silent, and consider blending the two conventions for
   those rows only.
4. Sanity-check any external replication of this finding before building on it. It is the highest-
   leverage claim in this document and it rests entirely on the profile-alignment evidence above.

Everything from §5.2 onward is computed on the **shifted** training data.

### 5.2 FINDING 2 — missingness is whole-day, never partial

Grouping by (station, direction, date):

- Every present day in the training file has **exactly 22 observations**. Every present day in
  validation has **exactly 24**. There are no partially-observed days anywhere in either file.
- Missing days per station-direction series:

  | | train (366 days) | validation (181 days) |
  |---|---:|---:|
  | mean missing days | 92.4 | 26.3 |
  | median | 66.5 | 14 |
  | max | 272 | 131 |
  | median longest consecutive run | 24 days | 2 days |
  | series with a ≥30-day outage | **72 / 150** | 22 / 150 |
  | median number of separate outage runs | 13 | 11 |

- Series start/stop mid-year: 30 series have no data before February 2024, 16 stop more than a month
  before year end. Rows per series in 2024 range 2,068 → 7,898 (a fully-observed 2024 at 22 h/day
  would be 8,052).

This is the classic signature of a **loop-detector or count-station outage**: the station goes dark
for weeks, then returns. It is not random missingness, so imputation methods that assume MCAR are
mis-specified, and any per-series model must tolerate long gaps and cold restarts.

Zero volumes are a different phenomenon and appear to be genuine: 0.90% of train rows and 1.61% of
validation rows are exactly 0, in runs of at most 7 hours (train) and 13 hours (validation), with
**no run reaching 24 hours**. They cluster in the small hours at low-volume stations. Treat zeros as
real observations, not as sensor failures.

### 5.3 FINDING 3 — the target set is three different problems in one file

Cross-referencing the 170,956 target rows against both labelled panels: **zero target rows overlap
the labelled data.** The target set decomposes as:

| Slice | Rows | Share | Stations | Period | Nature |
|---|---:|---:|---:|---|---|
| **A. Warm future** | 120,571 | 70.5% | 75 | 2025-07-01 → 2025-12-31 | 1–6 month extrapolation from own history |
| **B. Cold future** | 26,575 | 15.5% | 41 | 2025-07-01 → 2025-12-31 | no own history *and* out of sample in time |
| **C. Cold backfill** | 23,810 | 13.9% | 43 | 2024-01-15 → 2025-06-30 | no own history, but *contemporaneous* with observed neighbours |

Slice C is qualitatively different and easy to under-exploit: for those timestamps you know exactly
what every labelled station in the network was doing at that moment. It is a spatial-interpolation
problem (nowcast from neighbours), not a forecasting problem, and it should use a different model
from slices A and B. 13.9% of the score sits there.

### 5.4 FINDING 4 — targets are a sparse random sample, not contiguous series

The target set is **not** a dense grid. Per (station, direction, date) the template carries between
1 and 8 rows (mode 2–3), out of 24 possible hours. Sampling is uniform across hour-of-day (each hour
4.06–4.38% of rows) and across day-of-week (13.3–15.1%).

- 236 station-direction pairs, median 752 target rows each, range **3 → 1,067**.
- 118 stations, median ~1,500 rows each, but **`STN_0020` and `STN_0127` have only 6 rows apiece**
  and `STN_0123` / `STN_0095` have 75 / 115. Extreme per-station weight imbalance.
- All 184 days of Jul–Dec 2025 appear; 40–1,014 rows per day.
- Only 81 rows fall in calendar 2024, spread over 7 stations.

Practical implications: recursive / autoregressive forecasting is awkward (you would be predicting
scattered isolated hours), whereas a direct regression on calendar + station features maps naturally
onto the target set. And because rows are sampled uniformly, the global metric is effectively a
row-weighted average that over-weights high-row-count stations — but the undisclosed slice metrics
may not be, so do not let two 6-row stations rot.

---

## 6. The cold-start problem in detail

43 of 118 target stations (36%) have **zero** labelled observations. They carry 50,385 target rows
(29.5% of the file).

### 6.1 What is known about them

Only `network.csv` attributes and `network_edges.csv` adjacency. Their attribute distributions are
close to those of the labelled stations, which is what makes transfer viable:

| Attribute | cold (43) | labeled (75) |
|---|---|---|
| `through_lanes` = 2 | 32 (74%) | 52 (69%) |
| `functional_class` missing | 15 (35%) | 18 (24%) |
| `aadt_band` = `10k-25k` | 17 (40%) | 24 (32%) |
| `aadt_band` = `50k-100k` | 5 (12%) | 15 (20%) |
| `surface_type` = Composite | 30 (70%) | 37 (49%) |
| median `speed_limit` | 50 mph | 50 mph |

There is no cold station whose attribute combination is unrepresented among labelled stations, so
every cold station has plausible donors. Cold stations are mildly over-represented in the
missing-`functional_class` group, which slightly compounds the difficulty.

### 6.2 Graph position

| | value |
|---|---|
| cold stations with ≥1 edge | 43 / 43 (mean degree 3.0) |
| cold stations with ≥1 **labeled** neighbour | 41 / 43 |
| cold stations with a **labeled same-corridor** neighbour | 20 / 43 |
| cold stations with **no** labeled neighbour | 2 |
| cold stations within 2 hops of a labeled station | **43 / 43** |

So a neighbour-based transfer covers 95% of cold stations at one hop and **100% at two hops** — the
two exceptions, `STN_0041` (neighbours `STN_0019`, `STN_0058`) and `STN_0055` (neighbours `STN_0028`,
`STN_0074`, `STN_0127`), each reach 2–4 labeled stations at distance 2. No cold station is
information-isolated. The strongest edge type (`same_corridor`) is available for only 47% of them,
so the rest must lean on `proximity` edges and attribute similarity.

**Where the cold stations live.** They appear in *neither* labelled file: the training and validation
files contain the **exact same 75 stations** (verified set equality), so validation adds six months of
time, not new locations. The 43 cold stations exist only as rows in `network.csv`,
`network_edges.csv`, and the target file.

### 6.3 How well can cold start actually be done?

Simulated by leave-one-station-out on the 75 labelled stations: hide a station's history entirely,
predict it from (level prior = median level of labelled stations sharing its `aadt_band`) ×
(shape = mean normalised `hour × dow` profile of its labelled graph neighbours), and score against
its real Jan–Jun 2025 values.

| Cold-start strategy | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| Flat level prior only (no shape) | 542.4 | 797.4 | 80.1% |
| **aadt-band level × neighbour shape** | **356.5** | **625.3** | **55.3%** |
| *(reference)* warm model with own history | 129.6 | 254.5 | 24.9% |

Two readings. First, the shape transfer is worth a 34% MAE reduction over a flat prior — the diurnal
curve genuinely transfers between neighbouring stations. Second, even a reasonable cold-start method
is **2.7× worse** than a warm model, and that gap is irreducible with the supplied information. This
number is the honest basis for the `reliability_score` on cold stations and for their interval
widths; anything claiming near-warm confidence on a cold station is miscalibrated by construction.

### 6.4 Level priors: which attributes predict volume

Correlation with log mean hourly volume across the 75 labelled stations:

| Feature | corr |
|---|---:|
| `aadt_band` (ordinal) | 0.661 |
| `speed_limit` | 0.508 |
| `through_lanes` | 0.485 |
| `lane_width` | 0.143 |

`aadt_band` is the strongest single prior — **but two of its six levels are unreliable.** Comparing
each band against the AADT implied by the observed data (mean hourly volume × 24 hours × 2
directions):

| `aadt_band` | labeled stations | median implied AADT | verdict |
|---|---:|---:|---|
| `<5k` | 13 | **28,664** | **inconsistent** |
| `5k-10k` | 3 | **16,294** | **inconsistent** |
| `10k-25k` | 24 | 16,563 | consistent |
| `25k-50k` | 16 | 35,955 | consistent |
| `50k-100k` | 15 | 71,699 | consistent |
| `100k+` | 4 | 135,556 | consistent |

The four upper bands are well calibrated. `<5k` is not a low-volume band at all — its 13 labelled
members range from 86 to 2,159 mean hourly volume (implied AADT 4k to 104k). **[hypothesis]** `<5k`
is functioning as an "unknown/default" bucket rather than a real measurement, and `5k-10k` (n=3) is
too sparse to trust. Treat both as missing-value indicators, not as ordinal levels; the 7 cold
stations in `<5k` and 3 in `5k-10k` need their level prior from the graph instead.

---

## 7. Traffic statistics

### 7.1 Volume distribution (training data)

| stat | value |
|---|---:|
| mean | 890.5 |
| std | 1029.5 |
| min | 0 |
| p1 / p5 | 1 / 10 |
| p25 / median / p75 | 206 / 574 / 1,203 |
| p95 / p99 / p99.9 | 2,778 / 5,175 / 7,706 |
| max | 8,954 |
| exact zeros | 0.90% (validation: 1.61%) |

Strongly right-skewed with a hard floor at 0 → model in log space or with a Tweedie/Poisson-family
objective; a plain Gaussian MSE fit will systematically over-predict the quiet hours and produce
negative lower bounds that then have to be clipped (which quietly destroys interval calibration).

Series levels vary by **60×**: mean hourly volume per station-direction ranges 82 → 4,887 (median
615, p95 2,164). Any global-MAE objective is therefore dominated by a handful of high-volume
stations. Per-series normalisation (predict shape × level, or fit in log space) is close to
mandatory — it is also what makes the cold-start transfer of §6.3 work.

### 7.2 Diurnal profile (true clock, from validation)

| hour | mean volume | | hour | mean volume |
|---:|---:|---|---:|---:|
| 0 | 162 | | 12 | 1,187 |
| 1 | 108 | | 13 | 1,199 |
| 2 | **86** (min) | | 14 | 1,275 |
| 3 | 92 | | 15 | 1,368 |
| 4 | 149 | | 16 | **1,392** (peak) |
| 5 | 331 | | 17 | 1,335 |
| 6 | 665 | | 18 | 1,094 |
| 7 | 997 | | 19 | 872 |
| 8 | 1,058 | | 20 | 702 |
| 9 | 1,029 | | 21 | 540 |
| 10 | 1,052 | | 22 | 397 |
| 11 | 1,122 | | 23 | 267 |

A single dominant afternoon peak (15:00–17:00) with a morning shoulder around 07:00–08:00 and a
midday plateau — the classic mixed commuter/commercial arterial shape rather than a pure commuter
double peak. Peak-to-trough ratio is **16×**, which is why hour-of-day is the single most important
feature and why absolute-error metrics behave very differently at 03:00 and 16:00.

### 7.3 Weekly and annual cycles

Day-of-week means (training): Mon 887, Tue 888, Wed 896, Thu 940, **Fri 999**, Sat 849, **Sun 769**.
A modest ±12% weekly swing — far smaller than the diurnal swing, but the *shape* differs by day too
(weekends lose the morning shoulder), so `hour × day_of_week` interaction beats additive terms.

Monthly seasonal index, normalised so 1.00 = that series' annual mean:

| month | 2024 (train) | 2025 (validation) |
|---:|---:|---:|
| 1 | 0.827 | 0.870 |
| 2 | 0.913 | 0.889 |
| 3 | 0.956 | 0.959 |
| 4 | 0.990 | 1.034 |
| 5 | 1.052 | 1.100 |
| 6 | 1.087 | 1.134 |
| 7 | 1.092 | — |
| 8 | **1.106** | — |
| 9 | 1.022 | — |
| 10 | 1.012 | — |
| 11 | 0.947 | — |
| 12 | 0.919 | — |

Summer peak, winter trough, ~34% peak-to-trough. **The scored horizon (Jul–Dec 2025) spans the
annual maximum and the descent into winter** — the seasonal factor is not optional there. The only
available estimate of Jul–Dec seasonality is the 2024 curve, and 2025's H1 seasonal amplitude is
running slightly *stronger* than 2024's (1.134 vs 1.087 in June), so a naive 2024 seasonal transfer
will slightly under-shoot the summer peak.

### 7.4 Year-over-year drift

Per station-direction, matching months, 2025 vs 2024 (144 pairs with data in both):

| month | median ratio | pairs |
|---:|---:|---:|
| Jan | 0.945 | 104 |
| Feb | 0.860 | 122 |
| Mar | 0.895 | 121 |
| Apr | 0.916 | 128 |
| May | 0.913 | 132 |
| Jun | 0.905 | 130 |
| **overall** | **0.907** (IQR 0.875–0.938) | 144 |

Traffic in H1 2025 runs **~9% below** the same months of 2024, consistently across stations
(interquartile range only 0.875–0.938 — this is a network-wide trend, not a few outliers). Extending
2024 levels into H2 2025 without a level correction will over-predict.

> **Do not confuse two different corrections.** (a) *Year-over-year level*: 2025 sits ~9% below 2024
> — real, and it must be applied when carrying 2024 levels forward. (b) *Seasonal mix*: a profile
> averaged over all of 2024 is anchored to the annual mean, so when it is used to predict a
> high-season month it needs a factor *above* 1. In the holdout of §9.2 the two effects partly cancel
> and the empirically best flat multiplier is ×1.05–1.08 — that is not evidence against the −9%
> drift, it is the drift and the seasonal mix netting out. Model them as separate terms.

### 7.5 Difficult periods

Weekday-only, per-day network-mean index (1.00 = normal). Lowest days:

| 2024 | index | likely cause | 2025 | index | likely cause |
|---|---:|---|---|---:|---|
| 01-19 | **0.464** | severe winter storm **[hypothesis]** | 01-06 | **0.246** | severe winter storm **[hypothesis]** |
| 12-25 | 0.493 | Christmas | 02-12 | 0.552 | winter weather **[hypothesis]** |
| 01-01 | 0.563 | New Year | 01-01 | 0.611 | New Year |
| 01-16 | 0.626 | winter weather **[hypothesis]** | 01-07 | 0.694 | storm aftermath **[hypothesis]** |
| 11-28 | 0.675 | Thanksgiving | 01-20 | 0.711 | MLK Day |
| 05-27 | 0.724 | Memorial Day | 02-06 | 0.846 | winter weather **[hypothesis]** |
| 09-02 | 0.806 | Labor Day | 03-10 | 0.852 | |
| 07-04 | 0.830 | Independence Day | | | |

Highest weekdays are all late-June-to-August Fridays (up to 1.35).

Two distinct regimes are visible and they need different treatment:

- **Holidays are deterministic and forecastable.** US federal holidays cut volume 17–50%. A holiday
  flag (plus day-before/day-after) is cheap and directly improves the H2 2025 forecast, which
  contains Independence Day, Labor Day, Thanksgiving, and Christmas. In the §9.2 holdout, the single
  worst day was **2025-05-26 (Memorial Day), MAE 294.6 vs 118 overall** — a 2.5× error spike on a
  date that was known a year in advance.
- **Weather is stochastic and not forecastable at 6 months.** The 2025-01-06 collapse to 0.246 of
  normal cannot be predicted from the supplied data at any horizon. This is precisely what the
  interval and the reliability score exist to express — and it means winter target rows deserve
  *asymmetric* downside width (traffic can collapse to 25% of normal, but a storm never doubles it).

### 7.6 Correlation structure

Raw hourly correlations are inflated by the shared diurnal cycle, so these are computed on
**residuals after removing each series' `hour × dow` profile, in log space**:

| relationship | median residual corr |
|---|---:|
| same station, opposite direction (raw) | 0.916 |
| `same_corridor` edge | **0.844** |
| `proximity` edge | 0.617 |
| random station pair | 0.483 |
| edge with `distance_band = <=1mi` | 0.778 |
| edge with `1-5mi` | 0.673 |
| edge with `5-15mi` | 0.639 |

Three things follow. (1) The supplied graph carries **real** signal: corridor edges are far more
informative than proximity edges, which are in turn more informative than random pairs — the edge
types are not decorative. (2) Even unrelated stations share a residual correlation of 0.48, i.e.
there is a strong **network-wide common factor** (weather, regional demand). A latent daily
network factor is a compact, high-value feature — and it is the mechanism that makes slice C
(cold backfill, §5.3) tractable, since that factor is *observed* for those timestamps. (3) Because
errors are correlated across stations, aggregate uncertainty cannot be computed as if rows were
independent.

Temporal persistence of the network-mean residual: lag 1 h = 0.82, lag 24 h = 0.42, lag 168 h = 0.20.
Deviations are day-scale events, largely washed out after a week — consistent with weather.

### 7.8 Graph topology — what the two edge types actually are

The data provider (DelDOT) describes the edge types as: *"`proximity` means the locations are
spatially close in the underlying network, while `same_corridor` means they were associated with the
same roadway/corridor. The network is intentionally abstracted, so exact locations and source
roadway identifiers are not provided."* Measured against the data, the two types behave very
differently and should not be collapsed into one adjacency matrix.

| | `same_corridor` (55 edges) | `proximity` (131 edges) |
|---|---:|---:|
| residual correlation (median) | **0.844** | 0.617 |
| endpoints share travel axis (N/S vs E/W) | **92%** (46/50) | 56% (64/114) |
| distance band `<=1mi` / `1-5mi` / `5-15mi` | 15 / 30 / 10 | 17 / 90 / 24 |

The 92% axis agreement is strong confirmation that `same_corridor` really does mean "same road":
two stations on one roadway measure the same axis of travel, whereas proximity edges connect
stations on crossing or parallel-but-different roads and agree only slightly better than chance.

**The `same_corridor` subgraph is a set of linear corridors.** 76 stations, 55 edges, 21 connected
components, and **the maximum degree in that subgraph is 2** (42 nodes of degree 1, 34 of degree 2).
Every component is therefore a simple path, not a blob — component sizes 2 to 10 stations. The path
order *is* the spatial sequence of stations along that road. Example chains, with observed mean
hourly volume (`—` = no labels):

```
STN_0117(—) → STN_0118(2065) → STN_0053(2237) → STN_0066(1849) → STN_0065(1401) →
STN_0038(—) → STN_0071(—) → STN_0111(1398) → STN_0062(1126) → STN_0073(1563)

STN_0037(294) → STN_0036(243) → STN_0095(230) → STN_0002(—) → STN_0032(—) → STN_0046(—)

STN_0072(1494) → STN_0064(1864) → STN_0114(1900) → STN_0060(1626) → STN_0057(1689) → STN_0028(—)
```

Volume varies smoothly along a chain, and unlabeled stations frequently sit *between* labeled ones —
which turns many cold-start level priors into a 1-D interpolation along a corridor rather than a
generic attribute lookup. 15 of the 21 corridors contain both labeled and unlabeled stations.

**Correlation decays with graph distance**, which is the property that makes message passing
meaningful:

| hops in full graph | 1 | 2 | 3 | 4 | ≥5 |
|---|---:|---:|---:|---:|---:|
| median residual corr | 0.672 | 0.620 | 0.567 | 0.530 | 0.446 |

| steps along a corridor chain | 1 | 2 | 3 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|
| median residual corr | 0.844 | 0.773 | 0.730 | 0.688 | 0.576 | 0.431 |

**Full-graph topology.** 127 nodes, 186 undirected edges (no reciprocal duplicates — each unordered
pair appears once), **5 connected components** of sizes 52, 43, 26, 3, 3. Degrees 2–6 (mode 3).
No isolated nodes. The two 3-node components are effectively their own islands: no message can
reach them from the main body of the graph.

**No direction information is available on the edges, and it is not recoverable.** What *is*
available:
- The travel **axis** of every station (N/S or E/W) from its two `direction_code` values — 77 N/S
  stations, 41 E/W. This is known for cold stations too, since the target file lists both of their
  direction codes.
- The **ordering** of stations along a corridor, from the path topology. Note this gives sequence,
  not orientation: you know the chain runs A–B–C, not which end is north.
- **[hypothesis]** Orientation could in principle be inferred by cross-correlating directional flows
  between adjacent stations, but not at this resolution: neighbours are 1–5 miles apart, i.e. 1–5
  minutes of travel time, far below the 1-hour sampling interval. Treat the graph as undirected.

### 7.7 Noise is heteroscedastic

Residual std in log space, by decile of predicted volume:

| decile (low→high volume) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| log-residual std | 0.572 | 0.492 | 0.368 | 0.317 | 0.291 | 0.258 | 0.230 | 0.216 | 0.197 | 0.184 |

Overall 0.338. Low-volume cells are **3× noisier in relative terms**. The same effect appears at
station level: per-station sMAPE on the §9.2 holdout ranges 8.0% → 73.2% (median 18.5, p90 36.5), and
the five worst stations are all low-volume (STN_0009 at 235 veh/h → 73.2% sMAPE) while the five best
are all high-volume (STN_0072 at 1,441 veh/h → 8.0%).

This drives two design decisions directly: interval width must be conditioned on the predicted level
(a constant relative width is wrong at both ends), and `reliability_score` should be largely a
function of predicted volume, own-history availability, and horizon.

---

## 8. What the data implies for modelling

Not prescriptions — the measured facts, and what each one constrains.

**Feature space actually available at prediction time.** Calendar (hour, dow, month, day-of-year,
holiday flags, time index for trend), station statics (7 attributes, 2 with missingness), graph
position (degree, edge types, neighbour statistics), and own history (only for 75 of 118 stations,
and never contemporaneous with slices A/B). That is a small feature space — which is why the
in-sample oracle of §5.1 sits at MAE 108: with `pair × hour × dow` alone you are already near the
ceiling of what calendar features can explain, and further gains must come from seasonality, holiday
handling, level drift, and the network factor.

**A trap worth stating explicitly: neighbour readings are not available for 86% of the target rows.**
It is tempting to assume that a station with no history can be predicted from what its neighbours are
doing *at that moment*. That is true only for slice C. Labels stop at 2025-06-30, so for every
Jul–Dec 2025 target — 147,146 rows, 86% of the file — **no station in the network has an observation
at the target timestamp**, neighbour or otherwise. There is nothing to propagate. For those rows the
graph helps by transferring *learned structure* (a neighbour's diurnal shape, its level, its
seasonal response, a corridor-interpolated level prior) fitted on historical data, not by passing
contemporaneous messages. A GNN here is a *station-embedding* device, not a nowcasting device.
Slice C is the exception and the one place where true spatial nowcasting applies.

**Three models, not one.** Slices A, B and C (§5.3) differ in what is knowable:
- A (warm future): own history + seasonal factor + YoY drift + holidays.
- B (cold future): donor shape + attribute/graph level prior + the same seasonal and drift terms;
  widest intervals, lowest reliability.
- C (cold backfill): neighbours are *observed at the same timestamp* — a spatial regression using
  contemporaneous labelled neighbour volumes should beat any calendar model here. Skipping this and
  treating C like B leaves measurable points on the table across 13.9% of rows.

**Validation design.** Temporal only (rules §2.3). The realistic setup is: fit on 2024 + Jan–Apr
2025, score on May–Jun 2025 — that mimics the true task (extrapolate past the end of the labels) and
is what §9.2 uses. For the cold-start slice, add leave-one-station-out on the 75 labelled stations,
which is the only honest simulation of slices B and C. Reporting a single blended CV number hides
exactly the slices that carry 50% of the score.

**Uncertainty.** Coverage must be validated out-of-time (§9.3); in-sample residual quantiles are
optimistic by ~10 points of coverage. Width should scale with predicted level (§7.7) and with slice
(§6.3). Consider asymmetry: the downside tail (holidays, storms) is much fatter than the upside
(§7.5). The constraint `lower_90 >= 0` interacts badly with symmetric intervals at low volumes —
build intervals in log space and transform back.

**Reliability score.** The rules only require [0,1]; the scoring rubric is undisclosed.
**[hypothesis]** it is checked for *discrimination* — whether low scores actually mark the rows that
turned out wrong. A defensible construction is a monotone function of predicted out-of-sample error,
validated by showing that binned reliability tracks realised accuracy on the holdout. The measured
drivers are all available before scoring: own-history availability (2.7× error gap), predicted volume
level (3× noise gap), forecast horizon, station outage history, and holiday/winter dates.

**Scenarios.** The prompts are not in the repo. The correlation structure of §7.6 is the analytical
backbone for whatever they turn out to be: a disruption at one station propagates to same-corridor
neighbours with residual correlation 0.84, to proximity neighbours at 0.62, and the network-mean
residual decays from 0.82 at 1 h to 0.20 at one week — which is a directly defensible basis for
"recovery time" estimates. The observed holiday/storm days are natural empirical analogues for
"what does a large demand shock actually look like on this network" (a −54% to −75% network-wide
day, recovering within 1–3 days).

**Robustness.** 72 of 150 training series have 30+ day outages; two target stations have 6 scored
rows. Any per-series model must degrade gracefully to the pooled/donor path when a series is thin,
and that fallback should be exercised in testing, not just written.

---

## 9. Measured benchmarks

All figures are real measurements on the supplied data, reproducible with Appendix B.

### 9.1 Alignment matters more than the model

`pair × hour × dow` mean profile fit on 2024, scored on all of Jan–Jun 2025:

| variant | coverage of rows | MAE | RMSE | sMAPE |
|---|---:|---:|---:|---:|
| unaligned timestamps | 91.7% | 304.4 | 502.4 | 58.4% |
| **aligned (+2h)** | 91.7% | **129.6** | **254.5** | **24.9%** |
| aligned, median instead of mean | 91.7% | 126.5 | 258.7 | 23.2% |
| aligned, Q4-2024 only | 86.7% | 117.7 | 232.9 | 24.3% |
| aligned, last 8 weeks of 2024 | 86.1% | 128.8 | 247.9 | 25.0% |
| aligned, `pair × hour × dow × month` | 76.7% | 120.9 | 256.6 | 20.3% |
| aligned, `pair × hour × is_weekend` | 91.7% | 142.1 | 269.1 | 27.0% |
| in-sample oracle (fit on validation itself) | 100% | 107.6 | 212.1 | 24.0% |

Note the coverage column: finer groupings cut error but leave more target cells with no lookup
value at all (`pair × hour × dow × month` misses 23% of rows). A production model needs a hierarchical
fallback — cell → series → station → attribute group → global.

Error by validation month for the aligned profile: Jan 158, Feb 158, Mar 124, Apr 108, May 109,
Jun 125. **Error does not grow with horizon** over six months — it tracks *winter*, not distance from
the training data. That is encouraging for the Jul–Dec 2025 task, and it says the H2 forecast's risk
is concentrated in November–December.

### 9.2 Realistic holdout — fit on 2024 + Jan–Apr 2025, score May–Jun 2025

| model | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| `pair × hour × dow` profile | 117.8 | 226.5 | 22.9% |
| × 1.05 flat | 105.8 | 217.6 | 21.4% |
| × 1.08 flat | 105.5 | 220.7 | 21.1% |
| **× seasonal month factor** | **104.9** | 218.7 | **21.1%** |
| 2025-data-only profile | 157.9 | 279.9 | 30.0% |
| blend 50/50 history + 2025-only | 135.1 | 247.0 | 25.6% |

Discarding 2024 to "use only recent data" is a 34% MAE regression — the 2024 file, once aligned, is
the backbone of the model despite being a year old and 9% high in level. Seasonal correction is
worth ~11% MAE. Concentration of error: **the worst 5% of rows carry 32.7% of total absolute
error**, and the worst day is Memorial Day (§7.5).

### 9.3 Interval calibration — nominal 90%

Quantiles fitted on Mar–Apr 2025, applied to the May–Jun holdout:

| interval construction | coverage | mean width | Winkler (90%) |
|---|---:|---:|---:|
| global log-residual quantiles, as fitted | 0.795 | 611 | 943 |
| × 1.3 inflation | 0.860 | 768 | 979 |
| **× 1.6 inflation** | **0.900** | 920 | 1,057 |
| × 2.0 inflation | 0.934 | 1,117 | 1,200 |
| × 2.5 inflation | 0.959 | 1,365 | 1,412 |
| volume-decile-conditional quantiles (no inflation) | 0.727 | 464 | 1,094 |

Two lessons. (1) **Empirical residual quantiles are badly optimistic out-of-time** — ~80% actual
coverage for a nominal 90%, because the residual distribution drifts between the calibration window
and the scored window. Roughly 1.6× inflation restores nominal coverage on this split. (2) Winkler
score keeps *rising* past 90% coverage, so over-widening is penalised too — if the graders score
interval quality with a proper scoring rule, the target is calibration, not safety. Calibrate the
inflation factor on a genuinely out-of-time split, and expect the required factor to be larger for
the H2 2025 targets than for this 2-month holdout.

---

## 10. Open questions and things to confirm

1. **The +2h shift** (§5.1). Strongly evidenced but undocumented. Worth a mentor question phrased as
   file-structure clarification ("is the 2024 file's timestamp convention the same as the validation
   file's?") — that is explicitly in scope for mentors, whereas anything about scoring is not.
2. **Clock convention for the 81 target rows dated 2024** (§5.1). Negligible in volume, but state the
   assumption.
3. **Scenario prompts are not in the repo.** `scenario_template.csv` has 5 empty rows and no
   questions. They must be obtained before that 20% of the score can be addressed.
4. **`reliability_score` semantics are undefined.** Neither the schema nor the rules say what "1"
   means or how it is graded. Design for discrimination, define the semantics explicitly in the
   write-up, and show a calibration plot.
5. **`aadt_band` = `<5k` and `5k-10k` are internally inconsistent** with the observed volumes
   (§6.4). Treated here as unknown buckets.
6. **The 9 network stations that are neither labelled nor scored** contribute graph structure only —
   they can carry paths between a cold station and a labelled one, but never any volume.
7. **Metric unknown.** MAE, RMSE, MAPE and sMAPE rank models differently on this data (median-based
   profiles win on MAE and sMAPE, mean-based on RMSE; §9.1). With a 60× spread in series level, the
   choice between an absolute and a relative metric changes the optimum a lot. Absent disclosure,
   optimise something robust and report several.

---

## 11. Modelling results and what has actually been tried

*Added after building and measuring. Everything below is from the validation protocol
in §12, not from an easier split.*

### 11.1 Where the three slices stand

Nested station CV, fit on 2024 (seen stations only), predict Jan–Jun 2025, 4 folds:

| task | best model | GEH<5 | MAE | notes |
|---|---|---:|---:|---|
| warm-future | GNN (h192+) | **79.0%** | 96.1 | single fold; needs 4-fold confirmation |
| warm-future | GNN (h96, default) | 70.4% | 128.4 | 4-fold mean, ±2.2 |
| warm-future | LightGBM | 70.5% | 173.6 | 4-fold mean, ±0.9 |
| warm-future | backbone | 68.7% | 173.2 | 4-fold mean, ±1.1 |
| cold-interp | kriging GNN (IGNNK-style) | **38.0%** | 322.4 | positive on all 4 folds |
| cold-interp | LightGBM + neighbour feats | 37.1% | 325.6 | ±4.6 across folds |
| cold-future | direct (donor + graph) | 33.9% | 349.0 | ±4.7 |
| cold-future | GNN | 32.8% | 323.8 | ±1.7 |
| cold-future | **chained** (pseudo-history) | 31.6% | 363.0 | **worse than direct** |

Industry calibration target is ≥85% of links below GEH 5. Nothing is close on the cold
slices; warm-future is within reach.

Rows marked *4-fold mean* come from `protocol_results.json`. The two bolded rows are
**single-fold** and have not been reproduced across folds — an earlier version of this
table quoted the warm-future LightGBM fold-1 numbers (71.5% / 154.0) as if they were the
mean; the mean is 70.5% / 173.6. Given the fold spread on the cold slices, treat any
unconfirmed single-fold figure as a hypothesis, not a result.

### 11.2 Things that worked, with the measurement

- **The +2h clock fix.** MAE 304 → 130. Largest single effect in the dataset.
- **Direction-matched graph shape transfer for cold stations.** Transferring a
  neighbour's normalised `hour × dow` profile, *matched on direction_code*, beats the
  axis-level fallback: shape error 0.3445 → 0.2296 (33% better). Direction matching is
  what makes it work — North at station A is the same physical flow as North at station B.
- **hour × month interaction.** The daily *shape* moves with daylight (evening hours run
  16–21% higher in June than December). Backbone 116.4 → 109.9 MAE.
- **Model capacity on warm-future.** hidden 96 → 192 lifted GEH<5 from 68.2% to 78.9%
  and MAE from 129.5 to 97.5. Width helps; depth does not (5 layers < 3 layers).
- **Spatio-temporal kriging for cold-interp.** Making node features time-varying
  (observed residual + observed mask, message-passed over the 5 typed relations) and
  training with whole-station masking beats the static-feature backbone on all 4 folds.
- **Additive split-conformal.** Multiplicative width scaling exploded on collapsed
  intervals (width 16,318 on a 622-vehicle forecast). Additive offsets: Winkler
  16,692 → 1,835.

### 11.3 Things that did not work, and why

- **The chained C→B pipeline.** Reconstructing a cold station's history and feeding it to
  the forecaster is explicitly permitted by DelDOT and is the KITS/MoGERNN pattern, but it
  *hurt*: 31.6% vs 33.9% GEH<5. Two measurements explain it: the reconstruction-derived
  level and shape correlate **0.9994** with the donor prior the forecaster already had, so
  no new information enters; and reconstructions are smooth (no weather, no incidents), so
  training on them understates residual variance.
- **Per-pair drift slopes.** Fitted slopes reached −1.5/year on outage-broken spans and
  extrapolated catastrophically (MAE 173). Shrunk matched-month log ratios: 123.
- **Duan smearing** for the point forecast. +25 MAE — MAE wants the median.
- **Recent-level ratio features.** Hurt (150 vs 118) because an 8-week window confounds
  season with drift. Matched-month comparison is the correct form.
- **The graph, for cold-start *level*.** `aadt_band` alone reaches log-MAE 0.380; the full
  graph blend reaches 0.372. AADT band *is* a coarse volume measurement, so it dominates.
  The graph earns its place on *shape*, not level.
  **Superseded 2026-08-17 -- see §11.4d.** This conclusion was drawn from a
  *station-level* graph blend that averages a station's two directions together. A
  direction-matched *pair-level* transfer reaches 0.338, and the resulting GEH gain on the
  cold slices is the largest single improvement measured since the clock fix. The graph
  does earn its place on level; the old blend was throwing the signal away.

### 11.4 The largest open weaknesses

1. **Cold-slice fold variance is severe.** Backbone cold-interp MAE across four folds:
   610 / 261 / 243 / 219, GEH<5 29.5% / 32.2% / 43.6% / 38.9%. Single-fold cold results
   are meaningless; anything reported must be a 4-fold mean with spread.
2. **Cold-start level is information-bound.** ~50% of the cold penalty is level error and
   it is capped by what `aadt_band` + graph can say. The other ~50% is shape, where the
   graph does help.
3. **The kriging GNN works but is untuned.** `gnn_krige.py` implements IGNNK-style
   time-varying node features (observed residual + mask) with whole-station masking.
   Across 4 folds it beats the backbone on every one: GEH<5 36.1% -> **38.0%**, MAE
   333.3 -> **322.4**. That puts it level with the LightGBM + neighbour-feature path
   (37.1% / 325.6) rather than ahead of it -- the tabular features already captured most
   of the contemporaneous signal. It is a fresh architecture with no capacity sweep
   (hidden=128, 3 layers, early-stopped around epoch 20), and given how badly capacity was
   misjudged on warm-future it probably has headroom.
4. **Early stopping on pinball loss is a poor proxy for GEH.** Validation pinball was flat
   (0.0670 → 0.0692) across a capacity sweep in which GEH<5 rose 68.2% → 79.0%. The
   selection criterion pointed away from a 10-point improvement.
5. **Robustness (15% of the score) is unmeasured.** No outage-stress, graph-stress,
   low-volume or winter slices have been run.
6. **Scenario prompts are still missing** — 20% of the score, nothing buildable.

### 11.4b GEH is an L1 metric in SQUARE-ROOT space (verified 2026-08-17)

    GEH = |M-C| * sqrt(2/(M+C))        2|sqrt(M) - sqrt(C)| = 2|M-C|/(sqrt M + sqrt C)

These agree to within 1.5% over the whole relevant range, so

    GEH < 5   <=>   |sqrt(M) - sqrt(C)| < 2.5

Measured on 500k synthetic (M, C) pairs spanning 1-3000 veh/h and ratios 0.5-1.6: median
relative error 0.0023, max 0.0146, and the **GEH<5 decision agrees 99.96% of the time**.
Mean GEH equals 2x MAE-in-sqrt-space to within 0.5%.

Two consequences, both of which the pipeline currently gets wrong:

1. **The whole pipeline trains in `log1p` space** (`Backbone`, the LightGBM residual
   target, the GNN target `ly`, the pinball loss). Log space is a *relative* error
   geometry. At 2000 veh/h, GEH<5 permits an 11% error; at 50 veh/h it permits 71%. Log
   space weights those equally; GEH weights the high-volume row about 6x more. This is
   the most economical explanation for the measured complementarity in §11.5 -- the GNN
   reaches 97.6% GEH<5 in the lowest volume decile and 40.1% in the highest. **We are
   optimising the wrong end of the volume distribution for 70.5% of the scored rows.**
   The GEH-aligned point loss is L1 on `sqrt(volume)`, not on `log1p(volume)`.
2. **"Fraction below GEH 5" is a hit rate, not an error.** What maximises it is the
   conditional *mode* in sqrt space, not the mean or the median. On right-skewed
   conditional distributions the mode sits below the median, so a median-fitted model is
   biased upward relative to the GEH-optimal point. A per-volume-band multiplicative
   recentring is a cheap test of how much that costs.

External corroboration: Esugo, Haas & Lu, "Hybrid Deep-Learning Approach with a Geoffrey
E. Havers-Based Loss Function", *Transportation Research Record* 2679(1), 2025 -- a
GEH-shaped differentiable loss beat MAE training in 80% of cases and MSE in 94% on
multi-location traffic flow.

`diagnose_geh.py` runs the three cheap consequences of this: best global multiplier,
per-pair level oracle, and GEH<5 by volume decile.

### 11.4c The GNN was never given the backbone (found 2026-08-17, fixed)

LightGBM is fitted on the residual `ly - base` *and* receives `base` as a feature
(`run_pipeline.feature_list`). `gnn_v2.prepare` gave the GNN only statics, a direction
one-hot, Laplacian PE and calendar Fourier terms, with the raw `ly` as target -- so it had
to re-derive every pair's level and its hour x dow x month profile inside a
`hidden`-dimensional node-ID embedding.

That is the most economical explanation for the capacity anomaly in §11.2: width helped
enormously (hidden 96 -> 192 lifted GEH<5 by 10.7 points) while depth did not (5 layers <
3 layers). Width helping and depth not helping means the bottleneck was **memorisation
bandwidth, not message-passing depth**. It also explains why the GNN loses at high volume
-- it had no level anchor, and level error is what GEH punishes at high volume.

`gnn_v2` now predicts the residual when a `base` column is present, exactly like
LightGBM. If this is the right diagnosis, the capacity sweep becomes secondary and the
GNN/LightGBM complementarity should *shrink* (which would also shrink the 84.0% oracle in
§11.5 -- a good outcome, not a loss).

### 11.4d Direction-matched pair-level transfer (measured 2026-08-17)

`DonorModel` computes one level per *station* -- the mean of its pair levels -- and gives
every direction of that station the same number. `GraphShapeTransfer` documents why that is
wrong for *shape* (North at station A is the same physical flow as North at station B) and
fixes it; nothing did the same for *level*.

`PairLevelTransfer` does. Mean |log level| error on held-out pairs, 4 protocol folds:

| level estimator | log-MAE |
|---|---:|
| pipeline path: station donor blend + LightGBM level model | 0.394 |
| raw donor blend, no LightGBM | 0.386 |
| attribute prior (`aadt_band` median) alone | 0.425 |
| **`PairLevelTransfer`** | **0.338** |

End-to-end on the same 4 folds:

| slice | before | after | change | MAE |
|---|---:|---:|---:|---:|
| cold-interp | 37.1% (sd 4.63) | **42.8%** (sd 6.52) | **+5.7** | 325.6 -> 290.3 |
| cold-future | 33.9% (sd 4.70) | **37.1%** (sd 6.15) | **+3.2** | 349.0 -> 332.4 |

Positive on every fold on both slices, and MAE improves alongside GEH -- unlike the
sqrt-weighted warm-future loss (§11.4b), this gain does not depend on which metric the
organisers actually score.

What carries it, and what does not:

- **Direction-matched pair donors** instead of a direction-averaged station level.
- **A corridor:proximity weight ratio of 7.5:1.** `DonorModel` uses 1.5:1, which lets
  distant proximity donors outvote a corridor neighbour worth far more.
- **Not** `distance_band` weighting (0.361 vs 0.338 -- worse), and **not** inverse-variance
  weights fitted from labelled edge disagreement (0.351 -- the per-band samples are too
  small to estimate a variance from). Hand-set weights won.
- **The LightGBM level model actively hurts**: the raw donor blend (0.386) beats the full
  pipeline path (0.394) that wraps it. `predict_levels` blends the two 50/50 and is worse
  than either the donor blend alone or the new transfer.

Two caveats. `prior_k` and the two weights were tuned on these same four folds with 30
held-out pairs each, so the exact values are tuned rather than derived -- set C is untouched
and remains the honest test. And fold variance *rose* (sd 4.6 -> 6.5): the gain is large but
uneven, ranging +1.4 to +9.2 across folds on cold-interp.

Also measured: with the new level, the backbone alone scores 36.2% on cold-future against
37.1% for the full LightGBM residual path. The cold residual learner contributes under a
point; the level was doing nearly all the work.

### 11.5 Ideas not yet tried

- Volume-routed or stacked ensemble. GNN and LightGBM are complementary by volume: the
  GNN wins below ~250 veh/h (97.6% vs 86.7% GEH<5 in the lowest decile), LightGBM above
  (62.4% vs 40.1% in the highest). Per-row oracle selection reaches **84.0%** — near the
  85% target — so the two models already contain most of the needed information.
- GNN node embeddings + Laplacian PE as LightGBM features.
- Direction-share transfer: the per-direction share is stable within (station, direction,
  hour) at std 0.049, and 97% of pairs swing >10 percentage points across the day — an
  unexploited commute signature.
- Routing cold predictions by AADT reliability: the 10 cold stations with an unreliable
  band have 3.4× worse level error (0.904 vs 0.266 log-MAE).
- Two-tower GNN (corridor and proximity as separate towers, fused late) instead of one
  5-relation R-GCN.

## 12. The validation protocol

Nothing can be measured on the real target rows — their labels are withheld. So the 75
labelled stations are split and part of the signal is deliberately hidden.

    HISTORICAL = 2024        FUTURE = Jan–Jun 2025
    set C   15 stations, fixed, opened once at the end
    pool    the remaining 60, 4-fold; each fold hides 15 and fits on 45

| experiment | fit on | predict | mirrors |
|---|---|---|---|
| warm-future | seen, 2024 | seen, 2025 H1 | 120,571 rows (70.5%) |
| cold-interp | seen, 2024 | held out, **2024** | 23,810 rows (13.9%) |
| cold-future | seen, 2024 | held out, **2025 H1** | 26,575 rows (15.5%) |

The period relationship is the point: in deployment the labelled window runs 2024-01 to
2025-06, the backfill targets sit *inside* it and the future targets *after* it. Scoring
cold-interp inside the training window and cold-future after it reproduces that.

Splits are stratified on corridor degree, AADT reliability and `functional_class`
missingness, and frozen in `splits.json`.

**Metric: GEH**, the traffic-engineering standard, `sqrt(2(M-C)^2/(M+C))`; ≥85% of links
below 5 is the industry calibration target. Reported per slice with fold spread, alongside
within-±20%, MAE, sMAPE, R², coverage and Winkler. A single blended number hides exactly
the slices that carry half the score.

### 12.1 Confirmed with DelDOT

- Jul–Dec 2025 is a **batch forecast** using information through 2025-06-30; no new
  observations are revealed during the horizon.
- The objective is **expected hourly volume for planning**, not real-time incident
  prediction — which is why a calibration standard like GEH is the right metric.
- **Model-generated history is permitted** for stations with no observations, but must be
  "treated as an estimate rather than observed truth".
- Limited history is deliberate: "part of the analysis is determining whether an approach
  can generalize beyond locations with long, complete histories."

## Appendix A — quick numeric reference

| Quantity | Value |
|---|---|
| Labelled rows | 1,459,702 (902,902 train + 556,800 validation) |
| Labelled stations / series | 75 / 150 |
| Target rows / stations / series | 170,956 / 118 / 236 |
| Cold-start stations / rows | 43 / 50,385 (29.5%) |
| Train period (as supplied / true clock) | 2024-01-01 00:00–2024-12-31 21:00 / +2h, hours 02–23 |
| Validation period | 2025-01-01 00:00–2025-06-30 23:00, all 24 hours |
| Target period | 2024-01-15 20:00–2025-12-31 23:00 (99.9% in 2025) |
| Overlap of targets with labels | **zero rows** |
| Mean / median hourly volume | 890 / 574 |
| Series level range (mean hourly) | 82 → 4,887 |
| Diurnal peak / trough | 16:00 (1,392) / 02:00 (86) |
| Seasonal range | Jan 0.83 → Aug 1.11 |
| YoY level change (2025 vs 2024, H1) | ×0.907 |
| Train missing days per series (median) | 66.5 of 366 |
| Warm baseline MAE / cold-start MAE | 130 / 357 |
| Interval inflation for true 90% coverage | ×1.6 |

## Appendix B — reproducing these numbers

```python
# pip install pyarrow   (not installed by default in this environment)
import pandas as pd, numpy as np

tr = pd.read_parquet("traffic_train.parquet")
va = pd.read_parquet("traffic_validation.parquet")
tpl = pd.read_csv("submission_template.csv", low_memory=False)
net = pd.read_csv("network.csv")
edg = pd.read_csv("network_edges.csv")

# 1. ALWAYS do this first: align the training clock (§5.1)
tr["timestamp"] = tr["timestamp"] + pd.Timedelta(hours=2)
tr["hour"]        = tr.timestamp.dt.hour        # supplied columns match the WRONG clock
tr["day_of_week"] = tr.timestamp.dt.dayofweek
tr["month"]       = tr.timestamp.dt.month
tr["is_weekend"]  = tr.day_of_week >= 5

# 2. dtype fix before any join (§4.2)
tpl["direction_code"] = tpl["direction_code"].astype(str)
tpl["timestamp"]      = pd.to_datetime(tpl["timestamp"])

# 3. the three target slices (§5.3)
labeled = set(tr.station_key)
tpl["cold"] = ~tpl.station_key.isin(labeled)
tpl["slice"] = np.where(~tpl.cold, "A_warm_future",
                np.where(tpl.timestamp >= "2025-07-01", "B_cold_future", "C_cold_backfill"))
print(tpl["slice"].value_counts())

# 4. the baseline everything else should beat (§9.1)
K = ["station_key", "direction_code", "hour", "day_of_week"]
prof = tr.groupby(K).volume.mean()
pred = prof.reindex(va.set_index(K).index).values      # ~92% coverage; hours 0-1 are empty (§5.1)

# 5. always validate before delivery
#    python validate_submission.py my_submission.csv
```
