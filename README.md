# Delaware Traffic Pattern Analysis

Hourly traffic volume forecasting for 118 anonymous monitoring stations across
Delaware's road network, with calibrated uncertainty, scenario analysis, and a
deployed operational assistant.

**Live demo:** https://d3u4765on3vdko.cloudfront.net/ (API key required — see
[`docs/demo_runbook.md`](docs/demo_runbook.md))

---

## The problem in one paragraph

Traffic counters record how many vehicles pass each hour. Given 2024 plus the
first half of 2025, forecast hourly volume for July–December 2025 — including for
43 stations that have **no recorded history at all**. Every forecast needs a 90%
interval and a reliability score, and five operational scenarios need analysing.
The practical value is planning: knowing when a lane closure disrupts the fewest
drivers turns a judgement call into a number.

## Results

Measured on the validation set (Jan–Jun 2025, warm stations only):

| Metric | Base model | + LightGBM |
|---|---:|---:|
| **WAPE** (primary metric) | 15.28% | **13.43%** |
| MAE (vehicles/hour) | 117.7 | **103.4** |
| R² | — | **0.954** |
| Mean GEH | 4.05 | **3.64** |
| GEH < 5 | 73.2% | **77.2%** |
| GEH < 10 | 91.5% | **93.4%** |
| 90% interval coverage | — | 90.7% |

WAPE is the organisers' primary metric (confirmed 2026-08-18). GEH<5 is retained
because it is the traffic-engineering agreement statistic, but the two disagree:
GEH<5 is a hit rate that low-volume rows pass easily, while WAPE weights every
error by its absolute size.

Cold-slice performance can only be measured by hiding whole stations — see
[`benchmark/LEADERBOARD.md`](benchmark/LEADERBOARD.md).

## How it works

A multiplicative hierarchical model, deliberately chosen so every prediction is
reproducible by hand:

```
forecast = station_baseline
         × hour_of_day        × day_of_week    × month_of_year
         × station_hour_adj   × station_month_adj
         × year_over_year_trend
         + LightGBM residual correction
```

Each factor has a plain meaning, so any figure can be decomposed and challenged.
The `/explain` endpoint returns that decomposition for any prediction.

**Cold-start stations (43 of 118)** have no history. Their hourly, weekly and
seasonal profiles are reconstructed from connected neighbours in the supplied
station graph, weighted by edge type and distance, then scaled to a volume level
predicted from road characteristics (lanes, speed limit, AADT band). These
predictions are labelled `cold_start_enhanced`, carry wider intervals, and are
scored lower on reliability — they are estimates, not observations.

**One finding worth flagging.** The originally distributed `traffic_train.parquet`
was mis-timestamped: its hours 0–21 were really hours 2–23, and midnight and 01:00
were missing entirely. Detected by comparing train/validation hourly profiles
(with the shift, val/train ratios cluster at CV=0.026; without it they scatter to
CV=0.714). Correcting it moved R² from 0.76 to 0.94. The organisers reissued a
corrected file on 2026-08-18, which this repository uses; `HOUR_OFFSET` is now 0.

## Setup

Requires Python 3.12+ (developed on 3.13.5).

```bash
git clone https://github.com/NSF-DARSE/DelDot-Traffic-Team.git
cd DelDot-Traffic-Team

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The supplied data files (`traffic_train.parquet`, `traffic_validation.parquet`,
`network.csv`, `network_edges.csv`, `submission_template.csv`) are committed at
the repository root, so no download step is needed.

## Reproducing the submission

```bash
source .venv/bin/activate

# 1. Reconstruct cold-start profiles from neighbouring stations
python3 src/cold_start_enhance.py         # -> output/cold_start_enhanced.json

# 2. Build the deployable model artifact
python3 src/build_artifact.py 2.2         # -> output/model_artifact_v2.2.json

# 3. Generate the forecast submission
python3 src/run_pipeline.py               # -> output/submission.csv

# 4. Validate against the official checker
python3 validate_submission.py output/submission.csv
```

Expected: `PASS: True`, 170,956 rows, mean forecast 798.21.

Runtime is a few minutes on a laptop. Random seeds are fixed (LightGBM
`seed=42`); all other steps are deterministic aggregations.

## Deliverables

| File | Contents |
|---|---|
| `output/submission.csv` | 170,956 forecasts with `lower_90`, `upper_90`, `reliability_score` |
| `scenario_template_filled.csv` | Five scenario analyses with bounds, recovery times, assumptions |

## Repository layout

```
├── src/                        Forecasting pipeline and AWS handlers
│   ├── forecast_model.py         Layers 1-3: multiplicative model
│   ├── lgb_residual.py           Layer 4: LightGBM residual correction
│   ├── uncertainty.py            90% intervals and reliability scores
│   ├── cold_start_enhance.py     Synthetic history for stations with no data
│   ├── build_artifact.py         Serialise the model for deployment
│   ├── run_pipeline.py           End-to-end submission generation
│   ├── planning.py               Deterministic "best hour / best window" tools
│   ├── narrative.py              Bedrock narration + numeric-containment guard
│   ├── lambda_function.py        On-demand forecast API
│   └── lambda_batch_forecast.py  Daily 30-day rolling forecast
├── benchmark/                  Model-comparison harness (frozen folds, two arms)
├── docs/                       Methodology, security, demo runbook, lessons
├── output/                     Generated artifacts and deliverables
├── web/index.html              Traffic assistant chat page
├── scripts/deploy.sh           One-command AWS deployment
└── validate_submission.py      Official format checker (supplied)
```

## Deployed system

Running in AWS `us-east-1` for roughly **$25/month**, mostly the QuickSight licence.

```
05:00 ET  EventBridge -> batch Lambda -> 257,760 forecasts -> S3
06:00 ET  QuickSight SPICE refresh -> dashboard shows next 30 days

on demand  CloudFront page -> API Gateway (API key) -> Lambda -> forecast
                                                    -> Bedrock (language only)
```

API endpoints, all requiring an `x-api-key` header:

| Endpoint | Purpose |
|---|---|
| `/health` | Service and model version |
| `/forecast` | Single hour, or a full 24-hour profile |
| `/explain` | Factor-by-factor decomposition; `&narrate=true` adds prose |
| `/best-hours` | Quietest hours on a date (deterministic) |
| `/best-window` | Lowest-impact window for planned work (deterministic) |
| `/simulate-closure` | Closure impact on the station and its neighbours (deterministic) |
| `/ask` | Natural-language planning questions |

Redeploy with `bash scripts/deploy.sh 2.2`.

## Responsible AI

The language model never produces a number. Rankings and forecasts are computed
deterministically; Bedrock only converts a question into parameters and a computed
result into prose. A **numeric-containment guardrail** extracts every number from
generated text and rejects the output if any figure is absent from the
computation.

Data is anonymous aggregate hourly counts — no PII, no individual vehicle
tracking, and no attempt to de-anonymise station identifiers. Full detail,
including two security issues found and fixed during review, is in
[`docs/security_privacy_explainability.md`](docs/security_privacy_explainability.md).

## Documentation

| Document | Contents |
|---|---|
| [`docs/technical_documentation.md`](docs/technical_documentation.md) | Method, validation, metrics, assumptions, limitations |
| [`docs/explanation_for_stakeholders.md`](docs/explanation_for_stakeholders.md) | Plain-language explanation |
| [`docs/elevator_pitch.md`](docs/elevator_pitch.md) | 30-second pitch and Q&A |
| [`docs/scenario_analysis.md`](docs/scenario_analysis.md) | Five scenarios, methodology and limits |
| [`docs/security_privacy_explainability.md`](docs/security_privacy_explainability.md) | Privacy, security controls, explainability |
| [`docs/traffic_assistant.md`](docs/traffic_assistant.md) | Conversational planning layer |
| [`docs/closure_simulator.md`](docs/closure_simulator.md) | Closure impact model, capacity calibration, assumptions |
| [`docs/bedrock_narrative_layer.md`](docs/bedrock_narrative_layer.md) | Narration, guardrail, Bedrock model access |
| [`docs/aws_architecture.md`](docs/aws_architecture.md) | Deployed resources and costs |
| [`docs/edge_cases_and_solutions.md`](docs/edge_cases_and_solutions.md) | Twelve edge cases and their handling |
| [`docs/lessons_learned.md`](docs/lessons_learned.md) | What we learned on AWS |
| [`docs/demo_runbook.md`](docs/demo_runbook.md) | Demo run of play and fallbacks |

## Modelling and model comparison — `benchmark/`

[`benchmark/`](benchmark/) adds a comparison harness for the three-slice problem, so two
people can build different models and compare them by reading two numbers rather than
arguing about whose validation was fairer.

- [`benchmark/AGENTS.md`](benchmark/AGENTS.md) — **start here.** The working contract:
  frozen splits, the two evaluation arms, the model interface, the reporting rules, and
  what has already been ruled out.
- [`benchmark/LEADERBOARD.md`](benchmark/LEADERBOARD.md) — current standings, both arms.
- [`benchmark/FINDINGS.md`](benchmark/FINDINGS.md) — every measured result, with the script
  that produced it.

```bash
python3 benchmark/harness.py --model benchmark/models/backbone.py --arm protocol
python3 benchmark/harness.py --leaderboard
```

## Scope and framing

Scenario and closure outputs are **analytical estimates for planning comparison**,
not engineering-grade operational, safety, or capacity recommendations. Cold-start
forecasts have no ground truth available and are provisional until those stations
report real observations.
