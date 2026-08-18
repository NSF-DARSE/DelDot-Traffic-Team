# AGENTS.md — rules for anyone (human or model) adding a model here

This file is the contract. If you are an LLM agent working in `benchmark/`, read it in
full before writing code, and follow it exactly. The rules exist because each one has
already cost someone a measurement that turned out to mean nothing.

---

## 0. The one-paragraph version

Forecast hourly traffic volume for 170,956 target cells across 118 anonymised stations,
with a 90% interval and a reliability score per row. The target file is **three different
problems**, not one. Every model is scored by `benchmark/harness.py` on **frozen station
folds** so that two models differ only in the model. Do not invent your own split. Do not
compare a number from one arm against a number from the other.

---

## 1. Two things that fail silently — fix both or every number you produce is wrong

1. **The training file's clock runs 2 hours behind** validation and the targets. Add `+2h`
   to `traffic_train.parquet.timestamp` and recompute `hour` / `day_of_week` / `month` /
   `is_weekend` — the shipped columns follow the wrong clock. Measured effect:
   **MAE 304 → 130**, the largest single effect in the dataset. Both teams found this
   independently (see `../docs/edge_cases_and_solutions.md`).
2. **`direction_code` is `str` in the parquets and `int64` in the CSVs.** A cross-file
   merge returns zero rows and raises nothing.

`traffic_pipeline.load_all()` does both. **Use it instead of reading the files directly.**

```python
import traffic_pipeline as T
D = T.load_all()          # -> labeled, network, edges, targets, warm
```

`volume` has no NaNs. Missing hours are **absent rows**, not null rows, so `fillna` does
nothing. Missingness is whole-day, never partial.

---

## 2. The three problems

| slice | rows | share | stations | period | own history | network observed at t |
|---|---:|---:|---:|---|---|---|
| **warm-future** | 120,571 | 70.5% | 75 | Jul–Dec 2025 | 18 months | no (future) |
| **cold-interp** | 23,810 | 13.9% | 43 | Jan 2024 – Jun 2025 | **none** | **yes** |
| **cold-future** | 26,575 | 15.5% | 41 | Jul–Dec 2025 | **none** | no |

No target row overlaps the labelled panel. 43 of the 118 target stations have zero
labelled history — they must be predicted by cross-sectional transfer from
`network.csv` context and `network_edges.csv` neighbours, never from their own past.

---

## 3. Splits — the rule that makes the leaderboard mean anything

**Read `splits_frozen.json`. Never regenerate splits.**

```
report set   15 stations, FIXED. Untouched. Opened exactly once, at the very end.
CV pool      the remaining 60, in 4 folds of 15
per fold     fit on the 45 seen stations, score on the 15 unseen
```

- `validate_protocol.make_splits()` loads this file. If it ever prints a WARNING about
  regenerating from the seed, **stop** — your results are not comparable to anything.
- ⚠️ **`splits.json` (repo root and `benchmark/splits.json`) is a different, older
  partition** (`A_fit` / `B_select` / `C_report`) that no code ever read and that does
  **not** describe these folds. It is kept only for provenance. Ignore it.
- Never tune on the report set. Never look at it. It is the one honest test left.

---

## 4. Two arms, and they are not comparable

| arm | fit on | score | answers |
|---|---|---|---|
| `protocol` | 2024, 45 seen stations | warm-future, cold-interp, cold-future | the only arm that can measure cold start |
| `deploy` | 2024 + Jan–Apr 2025, all stations | May–Jun 2025 | mirrors the real warm task |

The **same backbone** scores **71.7%** GEH<5 on the protocol and **82.7%** on the
deployment arm. The protocol is systematically pessimistic for warm stations because it
throws away the 2025 H1 half-year deployment actually holds.

**Tune warm-slice decisions on `deploy`. Tune cold-slice decisions on `protocol`. Never
put a number from one next to a number from the other.**

---

## 5. How to add a model

Create `benchmark/models/your_model.py` with a `build()` returning an object that has
`name`, `fit(ctx)` and `predict(rows)`:

```python
class MyModel:
    name = "my_model"                       # becomes the leaderboard row + results filename

    def fit(self, ctx):                     # ctx.labeled is ALL you may look at
        ...                                 # ctx also has: network, edges, statics, graph,
        return self                         #   axis, seen_stations, experiment, fold, seed

    def predict(self, rows):
        # -> (len(rows), 3) VOLUMES: lower_90, forecast_volume, upper_90
        # rows carries station_key, direction_code, pair, timestamp + calendar columns
        return arr

def build():
    return MyModel()
```

Then:

```bash
P=~/.virtualenvs/data_science/bin/python
$P benchmark/harness.py --model benchmark/models/your_model.py --arm protocol
$P benchmark/harness.py --model benchmark/models/your_model.py --arm deploy
$P benchmark/harness.py --leaderboard          # regenerate LEADERBOARD.md
```

Use `--max-folds 1` for a smoke test. It runs the first frozen fold only and never changes
how a fold is built.

The harness rejects a prediction that is non-finite, negative, wrongly shaped, or not
ordered `lower <= point <= upper`. `models/_common.py` has a shared interval helper so a
comparison stays a comparison of point forecasts; emit your own intervals if uncertainty
is what you are improving — `cov90` and `winkler` are on the leaderboard for exactly that.

### What `fit` may not do

`ctx.labeled` has already had the held-out stations and out-of-window rows removed.
Reading the parquet files, the submission template, or any cached global inside `fit`
reintroduces the leakage the folds exist to prevent. **Any feature derived from "how much
data do we have about this unit" must be computed from the fold's own rows**, not from the
full panel — this exact mistake once made held-out cold stations look like warm ones and
inflated a reliability AUC from 0.851 to a meaningless 0.925.

---

## 5b. What the organisers confirmed (2026-08-18)

Answers from Sai Ram Nomula. These override every assumption made before that date.

| question | answer | what it changes |
|---|---|---|
| primary forecast metric | **WAPE**, `sum\|y - yhat\| / sum\|y\|` | not GEH, not MAE. Everything in this repo was selected on GEH<5 and needs re-checking. |
| scored as one pool? | **no** — across organiser-defined conditions | the A/B/C split is "directionally reasonable"; report per slice *and* overall |
| `reliability_score` | a **relative 0-1 confidence indicator**, not the probability of any event | higher should mean more dependable. It is a *ranking* problem, not a calibration one. |
| interval calibration | evaluated **separately**, through `lower_90` / `upper_90` | keep the conformal work; it is scored on its own |
| scenario prompts | **supplied by the organisers, common across teams** | ⚠️ **do not substitute self-defined scenarios** |
| robustness definitions | not disclosed; **define and justify your own** | reproducible stress testing + graceful behaviour on degraded/incomplete inputs |

### WAPE is not MAE, even though it looks like it

On a *fixed* row set WAPE = MAE / mean(y), so ranking two models on one slice by WAPE and
by MAE gives the same answer. Across slices it does not: mean volume differs per slice, so
only WAPE is comparable. Aggregate it as a **ratio of sums**, never a mean of per-fold
ratios.

WAPE weights an error by its absolute size. GEH<5 is a hit rate that low-volume rows pass
easily. **They disagree here, sharply** — see `LEADERBOARD.md`. A model that wins one can
lose the other, so state which one you optimised.

Measured starting point for anyone attacking this: on our backbone, warm-future fold 0,
**half the absolute error comes from the bottom five volume deciles, which carry 12.4% of
the volume.** We over-predict low-volume hours. Two fixes were tried and **rejected**: a
global multiplicative recentring (the *oracle* multiplier is worth 0.08 WAPE points) and
re-anchoring the pair level from `median(log1p(y))` to `log1p(mean(y))` (worse: 21.40% ->
23.51%). The open candidate is an L1-in-volume-space objective.

---

## 6. Reporting rules

- **Cold slices: always report the 4-fold mean and the spread.** Backbone cold-interp MAE
  across the four folds is 610 / 261 / 243 / 219. A single-fold cold number is noise.
- Warm-future fold spread is small (sd ≈ 0.6), so warm differences of ~0.5 GEH points are
  readable; cold differences below ~2 points are usually not.
- **Lead with WAPE** — it is the organisers' primary metric. Quote `GEH<5` alongside it;
  they disagree, and the disagreement is informative rather than noise.
- State which arm produced every number.

---

## 7. Competition rules that override any modelling idea

From `../rules_and_data_use.md`:

- **No future information.** Only data available at or before the predicted timestamp.
  Build features causally; validate with time-based splits, never random k-fold.
- **Never de-anonymise `station_key`.** No external traffic datasets, APIs, or pattern
  matching to recover station identity or withheld observations.
- **Never modify** `target_id`, `station_key`, `direction_code`, `timestamp` in a
  submission; never add, drop, duplicate or reorder rows.
- Scenario results are **analytical estimates**, never engineering-grade operational,
  safety, capacity or design recommendations. Phrase them that way everywhere.
- ⚠️ **The five scenario prompts come from the organisers and are common across teams.**
  Do not invent scenarios and do not ship self-defined ones in `scenario_template_filled.csv`
  — the organisers asked explicitly that self-defined scenarios not be substituted for the
  required output. Wait for the prompts.
- Document dependencies, preprocessing, procedure and random seeds.

---

## 8. Where the headroom is — read this before choosing what to build

Oracle substitution (hand the model one perfect component). These bound what work on that
component can be worth:

Warm-future, protocol arm, corrected data. **WAPE is the primary metric**, GEH<5 shown
because the two rank differently:

| perfect component handed to the model | WAPE | GEH<5 |
|---|---:|---:|
| *nothing* (current backbone) | 13.11% | 79.5% |
| + month | 12.79% | 80.3% |
| + shape | 12.40% | 81.0% |
| + **level** | **12.28%** | 81.4% |
| + level + shape + month | 11.03% | 84.6% |
| full oracle pair×hour×dow×month | **8.48%** | 88.6% |
| + true network shock | 7.43% | 91.9% |

**The lever depends on the metric, and this is easy to get wrong.** Under WAPE, level is
worth slightly more than shape on warm-future (−0.83 against −0.71) because WAPE weights an
error by its absolute size and level error dominates at high volume. Under GEH<5 the same
decomposition says the opposite (+9.3 shape against +1.7 level), because GEH is a hit rate
that low-volume rows pass easily. Any lever quoted in GEH points may not survive conversion.

On the cold slices the level lever is much larger — a perfect level was worth +18.5 GEH
points on cold-interp against +2.0 for shape. **That decomposition has not been re-measured
in WAPE on the corrected data**; treat it as indicative. Cold-future has no measured ceiling
at all, in either metric.

---

## 9. Negative results — do not redo these

Every one was measured end to end. Details and numbers in `FINDINGS.md`.

- **Low-rank denoising / empirical-Bayes shrinkage of warm profiles** — nothing, once the
  empty-cell fill is held constant.
- **Recency weighting** of the profile — +2.2% at best, catastrophic when aggressive.
  Seasonal matching dominates recency.
- **Shape-drift extrapolation** — looked excellent on a proxy (−24.1% profile error), was
  **worse** end to end (82.7% → 81.5%).
- **Ridge attribute prior for cold level** — better standalone, no difference inside the
  estimator. Reverted.
- **Random forest / gradient boosting for the cold level prior** — both *worse* than the
  `aadt_band` median. 75 stations is not enough to fit them.
- **CF learned similarity for the contemporaneous shock** — 4.97% of the shock explained
  against 20.24% for the plain network median. The shock is global, not local.
- **GNN capacity** — 12 configs, width 192–768, all between 83.04% and 83.51%. The earlier
  "width is worth 10 points" was a *missing input* (the backbone), not architecture.
- **2-hop shape transfer** — real but ≈ +0.1 GEH end to end.
- **Neighbour lags for the nowcast** — `t-2…t+2` adds +0.003 R² on a 0.49 base, coin-flip
  win rate. Same-hour only is correct.

Two recurring lessons, stated because they keep recurring: **a proxy metric improving is
not evidence**, and **a metric improving is not evidence a feature is sound.**

---

## 10. Environment

```bash
P=~/.virtualenvs/data_science/bin/python     # pandas 2.2.3, pyarrow, sklearn 1.6.1,
                                             # lightgbm 4.6, torch 2.6+cu124, torch-geometric
```

The GNN entry points (`gnn_v2.py`, `gnn_krige.py`) need a GPU to be practical;
`gnn_baseline.pick_device()` probes with a real matmul and falls back to CPU rather than
crashing mid-run. Every script takes `--seed` (default 0). Record it.

Data files live at the **repository root**, not in `benchmark/`. `traffic_pipeline`
resolves this automatically by walking up; `TRAFFIC_DATA_DIR` overrides it.

---

## 11. Orientation

| file | what it is |
|---|---|
| `harness.py` | the comparison harness — folds, metrics, results, leaderboard |
| `LEADERBOARD.md` | generated; every model, both arms |
| `splits_frozen.json` | **the frozen folds. Read, never regenerate.** |
| `FINDINGS.md` | every measured result, with the script that produced it |
| `PROBLEM_BRIEF.md` | the long-form measured description of the data and every experiment |
| `traffic_pipeline.py` | canonical loading, `Backbone`, transfers, conformal, metrics |
| `validate_protocol.py` | the original protocol runner (three experiments, nested station CV) |
| `deploy_arm.py` | the deployment-mimicking arm |
| `gnn_v2.py` / `gnn_krige.py` | R-GCN over 236 station-direction nodes / inductive kriging |
| `results/published/` | the result JSONs behind the numbers quoted in `FINDINGS.md` |
| `slides/` | decks: the model per slice, the validation protocol, the overview |
