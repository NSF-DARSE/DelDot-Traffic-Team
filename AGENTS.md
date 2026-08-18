# AGENTS.md — read this first

Single entry point for this repository, written for LLM agents and for people. It explains
what exists, what was built, in what order, and **why** each decision was taken. Everything
else is linked from here; you should not need to guess where anything lives.

If you read only one section, read [§2 Five things that will silently ruin your
results](#2-five-things-that-will-silently-ruin-your-results).

---

## 1. What this repository is

A DelDOT / NSF-DARSE case study. Forecast **hourly traffic volume** for **170,956 target
cells** across **118 anonymised monitoring stations**, each row needing a point forecast, a
90% prediction interval, and a reliability score. Plus five scenario answers and a technical
write-up.

**Scoring** (from [`README.md`](README.md)):

| area | weight |
|---|---:|
| Forecast performance | 35% |
| Scenario analysis | 20% |
| Uncertainty and reliability | 15% |
| Robustness | 15% |
| Reproducibility and operational readiness | 10% |
| Traffic analysis and technical quality | 5% |

Only 35% rewards point accuracy. Half the score is behaviour on hard slices, honest
intervals, and scenarios.

### What the organisers have confirmed

Answers from Sai Ram Nomula, 2026-08-18. These override anything written before that date.

| question | answer |
|---|---|
| primary automated forecast metric | **WAPE** = `sum\|y - yhat\| / sum\|y\|` — not MAE, not GEH |
| scored as one pool? | **No** — across organiser-defined conditions |
| our A/B/C slice framing | "directionally reasonable" |
| `reliability_score` | a **relative 0–1 confidence indicator**, *not* the probability of any event. Higher = more dependable. |
| interval calibration | evaluated **separately**, via `lower_90` / `upper_90` |
| scenario prompts | **supplied by the organisers, common across teams.** ⚠️ Do **not** substitute self-defined scenarios. |
| robustness definitions | not disclosed — **define and justify your own** stress tests; they assess reproducible stress testing and graceful behaviour under degraded/incomplete inputs |
| operational readiness | **production infra (API/container) is NOT required.** Reproducible code, clear execution instructions, and a credible discussion of monitoring, data quality, retraining and fallback. |

⚠️ **`scenario_template_filled.csv` currently contains five self-defined scenarios.** They
were written before the answer above and must be replaced when the real prompts arrive.

---

## 2. Five things that will silently ruin your results

Each one produces plausible-looking output while being wrong. Four of the five actually
happened to us.

1. **The 2024 training file was reissued on 2026-08-18.** The original had its observations
   offset **+2 hours** and was **missing every 00:00 and 01:00 record** — a TMAS
   source-layout difference between 2024 and 2025. We detected it from the data; the
   organisers confirmed and reissued. `traffic_train.parquet` is now the corrected file
   (984,984 rows, hours 0–23). The old one is kept as `traffic_train_2h_offset.parquet`
   for provenance. **Any hardcoded `+2` offset is now wrong** and will put you two hours
   off in the other direction. `traffic_pipeline.detect_clock_offset()` measures the offset
   on every load and warns if it is not 0.
2. **`direction_code` is `str` in the parquets and `int64` in the CSVs.** A cross-file merge
   returns zero rows and raises nothing.
3. **Missing hours are absent rows, not null rows.** `volume` has no NaNs, so `fillna` does
   nothing. Missingness is whole-day, never partial.
4. **A missing key in the profile base is not a missing correction.** Before the reissue, a
   fit on 2024 alone contained no hours 0–1; `Backbone.predict` mapped them to NaN,
   `nan_to_num` made them 0.0, and the model returned each pair's *whole-day average* for
   01:00 — WAPE **558%** on those rows, +765 veh/h mean bias. Fixed by
   `_complete_hour_dow`. The guard stays even though the data no longer needs it.
5. **Never regenerate the CV splits.** Read [`benchmark/splits_frozen.json`](benchmark/splits_frozen.json).
   The older `splits.json` is a *different* partition that no code ever read — it is kept
   only for provenance.

`traffic_pipeline.load_all()` handles 1–3. **Use it instead of reading files directly.**

---

## 3. The target file is three different problems

Of the 118 target stations, **43 have no labelled history at all**.

| slice | rows | share | own history | network observed at t | what it is |
|---|---:|---:|---|---|---|
| **A warm-future** | 120,571 | 70.5% | 18 months | no (future) | forecast 6 months ahead |
| **B cold-interp** | 23,810 | 13.9% | **none** | **yes** | reconstruct the past from the network |
| **C cold-future** | 26,575 | 15.5% | **none** | no | both problems at once |

They need different things, and **which thing depends on the metric** — a trap worth
understanding before optimising anything.

Oracle substitution on warm-future (protocol arm, corrected data), handing the model one
perfect component:

| perfect component | WAPE | GEH<5 |
|---|---:|---:|
| *nothing* (current backbone) | 13.11% | 79.5% |
| + month | 12.79% | 80.3% |
| + shape | 12.40% | 81.0% |
| + **level** | **12.28%** | 81.4% |
| + level + shape + month | 11.03% | 84.6% |
| full oracle pair×hour×dow×month | **8.48%** | 88.6% |
| + true network shock | 7.43% | 91.9% |

**Under WAPE, level is worth slightly more than shape on warm-future** (−0.83 vs −0.71),
because WAPE weights an error by its absolute size and level error dominates at high volume.
Under GEH<5 the same decomposition says the opposite (+9.3 shape against +1.7 level),
because GEH is a hit rate that low-volume rows pass easily.

On the cold slices the level lever is much larger still — a perfect level was worth +18.5
GEH points on cold-interp against +2.0 for shape. **That cold decomposition has not yet been
re-measured in WAPE on the corrected data**; treat the GEH figures as indicative only.

The practical rule: **cold start is a level problem, warm-future is a mixed problem, and any
lever quoted in GEH points may not survive conversion to WAPE.**

---

## 4. Where everything is

### Start here

| file | what it gives you |
|---|---|
| **[`benchmark/AGENTS.md`](benchmark/AGENTS.md)** | the working contract: splits, evaluation arms, model interface, reporting rules, and every negative result already ruled out. **Read before writing any model.** |
| **[`benchmark/LEADERBOARD.md`](benchmark/LEADERBOARD.md)** | current standings, both arms, generated by the harness |
| **[`benchmark/FINDINGS.md`](benchmark/FINDINGS.md)** | every measured result, with the script that produced it and the reasoning |
| **[`benchmark/PROBLEM_BRIEF.md`](benchmark/PROBLEM_BRIEF.md)** | long-form measured description of the data and every experiment run |

### The delivered pipeline (AWS-shaped)

| path | what it is |
|---|---|
| [`src/forecast_model.py`](src/forecast_model.py) | multiplicative profile model: baseline × hour × dow × month × station adj × trend |
| [`src/lgb_residual.py`](src/lgb_residual.py) | LightGBM residual correction |
| [`src/uncertainty.py`](src/uncertainty.py) | prediction intervals and reliability scores |
| [`src/run_pipeline.py`](src/run_pipeline.py) | end-to-end driver |
| [`src/lambda_function.py`](src/lambda_function.py), [`src/lambda_batch_forecast.py`](src/lambda_batch_forecast.py) | AWS Lambda entry points |
| [`docs/`](docs/) | technical documentation, stakeholder explanation, AWS architecture, edge cases, scenario analysis, security/privacy |
| [`output/submission.csv`](output/) | the generated submission |

⚠️ `src/forecast_model.py` has `HOUR_OFFSET = 2` hardcoded and derives hours 0–1 from the
validation file. **Both are wrong against the reissued data file** (see §2.1).

### The modelling and comparison work

| path | what it is |
|---|---|
| [`benchmark/harness.py`](benchmark/harness.py) | the comparison harness — folds, metrics, results, leaderboard. No model logic. |
| [`benchmark/models/`](benchmark/models/) | model entries implementing the `fit`/`predict` contract |
| [`benchmark/traffic_pipeline.py`](benchmark/traffic_pipeline.py) | canonical loading, the profile `Backbone`, cold-start transfers, conformal calibration, metrics |
| [`benchmark/run_pipeline.py`](benchmark/run_pipeline.py) | slice router, cold-start level model, reliability model, submission writer |
| [`benchmark/validate_protocol.py`](benchmark/validate_protocol.py) | the nested station-CV protocol, three experiments |
| [`benchmark/deploy_arm.py`](benchmark/deploy_arm.py) | the deployment-mimicking arm |
| [`benchmark/gnn_v2.py`](benchmark/gnn_v2.py) | R-GCN over 236 station-direction nodes, 5 typed relations |
| [`benchmark/gnn_krige.py`](benchmark/gnn_krige.py) | IGNNK-style inductive kriging for cold-interp |
| [`benchmark/robustness_slices.py`](benchmark/robustness_slices.py) | the stress slices |
| [`benchmark/warm_decomp.py`](benchmark/warm_decomp.py) | oracle decomposition — where the ceiling is |
| [`benchmark/results/`](benchmark/results/) | harness results; `published/` holds the JSONs behind quoted numbers |
| [`benchmark/slides/`](benchmark/slides/) | decks: the model slice by slice, the validation design, the overview |

---

## 5. How the solution was built, and why

Chronological, because each step was a response to a measurement from the previous one.
Full numbers in [`benchmark/FINDINGS.md`](benchmark/FINDINGS.md).

1. **Establish what the data actually is.** Counting rows revealed the target file is three
   problems, not one, and that 43 stations have zero history. Everything downstream follows
   from that split.
2. **Find the clock offset.** The 2024 and 2025 hour-of-day profiles disagreed by exactly
   two hours. Correcting it moved MAE 304 → 130, the largest single effect in the dataset.
   Later confirmed as a real data-preparation bug and fixed at source by the organisers.
3. **Build a structural backbone before any learning.** `level × shape(hour,dow) × season ×
   drift × holiday` in log space, with hierarchical shrinkage global → axis → station →
   pair. A raw regression would have to rediscover a 60× spread in series levels and a 16×
   diurnal ratio.
4. **Make the profile seasonal, not pooled.** The hour×dow profile is not noisy, it is
   *dated*: the same months a year apart disagree by 0.157 log against a 0.097 within-year
   noise floor. A half-month seasonal kernel beat pooling (profile error 0.170 → 0.142) and
   beat recency weighting, which *hurt*. This was the largest modelling gain.
5. **Fix cold start as a level problem.** Direction-matched *pair* donors along the graph,
   corridor:proximity 6:0.8 — because North at station A is the same physical flow as North
   at station B, and averaging a station's two directions destroys that. Level error
   0.394 → 0.338.
6. **Add learned components only where they pay.** LightGBM residual quantiles for warm
   rows; an R-GCN over station-direction nodes; blended 50/50 because two validation windows
   disagreed about which was better. On cold rows the residual learner does not transfer, so
   it is blended halfway back to the backbone.
7. **Calibrate rather than trust.** Additive split-conformal, Mondrian by slice × volume
   band. Raw quantile heads covered 80/35/26% against a nominal 90%.
8. **Freeze the evaluation.** Nested station CV plus a deployment-mimicking arm, folds
   committed to disk, so results are comparable across people and across time.
9. **Re-check everything against the real metric.** When WAPE was confirmed, re-scoring
   surfaced a 558%-WAPE hole at hours 0–1 that GEH<5 had masked for the whole project.

**Two lessons that kept recurring**, worth internalising before proposing anything:

- **A proxy metric improving is not evidence.** Shape-drift extrapolation improved a
  standalone profile proxy by 24% and made the end-to-end result *worse*.
- **A metric improving is not evidence a feature is sound.** A history-depth feature
  computed from the full panel instead of the fold's own rows raised reliability AUC from
  0.851 to 0.925 — by leaking which rows were held out.

---

## 6. Running things

```bash
P=~/.virtualenvs/data_science/bin/python     # pandas 2.2.3, pyarrow, sklearn 1.6.1,
                                             # lightgbm 4.6, torch 2.6+cu124, torch-geometric

# score a model on the frozen folds
$P benchmark/harness.py --model benchmark/models/backbone.py --arm protocol
$P benchmark/harness.py --model benchmark/models/backbone.py --arm deploy
$P benchmark/harness.py --leaderboard        # regenerate LEADERBOARD.md

# the underlying experiments
$P benchmark/validate_protocol.py --folds 4 --no-gnn
$P benchmark/deploy_arm.py
$P benchmark/warm_decomp.py --folds 4        # oracle ceilings
$P benchmark/robustness_slices.py --folds 4

# gate before delivering anything
$P validate_submission.py output/submission.csv
```

Every script takes `--seed` (default 0). Record it. Data files live at the repository root;
`traffic_pipeline` resolves the path by walking up, and `TRAFFIC_DATA_DIR` overrides it.

---

## 7. Rules that override any modelling idea

From [`rules_and_data_use.md`](rules_and_data_use.md):

- **No future information.** Only data available at or before the predicted timestamp. Build
  features causally; validate with time-based splits, never random k-fold.
- **Never de-anonymise `station_key`.** No external datasets, APIs, or pattern matching to
  recover station identity or withheld observations.
- **Never modify** `target_id`, `station_key`, `direction_code` or `timestamp` in a
  submission; never add, drop, duplicate or reorder rows.
- Scenario results are **analytical estimates**, never engineering-grade operational, safety,
  capacity or design recommendations. Phrase them that way everywhere.
- Document dependencies, preprocessing, procedure and random seeds.

---

## 8. What is open

1. **The five scenario prompts have not arrived.** 20% of the score, nothing buildable until
   they do, and self-defined substitutes are explicitly disallowed.
2. **Cold-future has no measured ceiling.** 15.5% of scored rows, and we cannot say whether
   further work there is capped or open.
3. **Robustness is currently evaluation-slicing only.** The organisers described "graceful
   behavior under degraded/incomplete inputs" — we slice the evaluation set but do not yet
   degrade the *inputs* (drop history, remove edges, blank attributes, inject outages).
4. **Operational-readiness narrative.** No infra needed, but monitoring, data quality,
   retraining and fallback need writing up. The clock-offset detector is a concrete
   data-quality control to point at.
5. **Whether partial observations are scored.** On the first recorded hour after a ≥24h gap
   the true value has a median of 67 veh/h where a model predicts ~613 — the sensor returns
   mid-hour. No forecast can match a partial count. Asked, not yet answered.
