# Session findings — 2026-08-17/18

> **⚠️ Data correction, 2026-08-18.** The organisers reissued `traffic_train.parquet`: the
> original 2024 file was offset +2h and was missing every 00:00 and 01:00 record.
> **Every number in this file dated before the "Corrected training data" section below was
> measured on the old file** and is superseded. The reasoning still holds; the magnitudes
> moved. Current standings are always in `LEADERBOARD.md`, regenerated from the harness.
> Also note the primary metric is **WAPE**, confirmed 2026-08-18 — sections written earlier
> lead with GEH<5.

Every number below is measured, with the script that produced it named. Two evaluation
setups are used and they are **not** comparable to each other:

- **protocol** — `validate_protocol.py`, 4-fold nested station CV, fit 2024, score 2025 H1.
  The only way to measure cold-start, because it hides whole stations.
- **deployment arm** — `deploy_arm.py`, fit 2024 + Jan–Apr 2025, score May–Jun 2025.
  Mirrors the real task (we hold 2024 + 2025 H1 and forecast 2025 H2). Warm slices only.

The protocol is systematically pessimistic for warm stations because it throws away the
2025 H1 half-year that deployment actually has: backbone 68.7% GEH<5 on the protocol vs
77.8% on the deployment arm, same code.

## What moved the number

| change | where measured | effect |
|---|---|---|
| Seasonal-kernel profile (`_season_bin`, half-month bins) | deployment arm | **77.8% → 82.2%**, MAE 109.6 → 94.6 |
| `k_season=1.0` (loosen seasonal shrinkage only) | deployment arm | 82.2% → **82.7%**, MAE 94.6 → 93.4 |
| LightGBM residual retuned 350 → 120 trees | deployment arm, 2 windows | mean 82.95% → **83.25%**, MAE 93.6 → 92.4 |
| Blend GNN with LightGBM for the warm slice | deployment arm | 82.9% → **83.5%**, MAE 92.0 → 90.7 |

### Warm-future end to end, deployment arm

| config | GEH<5 | MAE |
|---|---|---|
| pooled backbone + LightGBM n=350 (what we started with) | 79.4% | 102.7 |
| seasonal backbone | 82.7% | 93.4 |
| + LightGBM n=120 | 82.9% | 92.0 |
| GNN (h192, +base) | 83.4% | 90.9 |
| **mean(GNN, LightGBM)** — now deployed | **83.5%** | **90.7** |
| oracle pair × hour × dow × season | 88.6% | 67.7 |

**+4.1 GEH points and 12% lower MAE on 70.5% of the scored rows**, and 1.5 points short of
the 85% industry target with 5.1 points of headroom left.
| `PairLevelTransfer` (direction-matched cold level) | protocol | cold-interp **37.1% → 42.8%**, cold-future **33.9% → 37.1%** |
| GNN gets `base` + GEH selection + sqrt weighting | protocol | warm **70.3% → 72.3%**, fold sd 2.21 → 0.87 |
| `_complete_hour_month` | held-out (hour, month) cells | error 0.0524 → 0.0302 (−42.5%) |

## Ceilings — what is actually attainable

Oracle decompositions, handing the model perfect components (`warm_decomp.py`, and the
deployment-arm version):

| slice | current | perfect profile | + true network shock |
|---|---|---|---|
| warm-future (deployment arm) | 82.2% | **88.6%** | 91.2% |
| warm-future (protocol) | 72.3% | 88.4% | 91.8% |
| cold-interp (protocol) | 42.8% | 80.5% | 85.8% |

**The 85% industry target is reachable on warm-future in deployment** (82.2% now, 88.6%
ceiling) and **not reachable on the cold slices** — a perfect deterministic profile tops
out at 80.5% there, and published work on this problem class reports 52.6% GEH<5 using
probe speeds, weather and coordinates the supplied data does not contain.

## Which lever matters where — the levers are opposite

| slice | perfect level worth | perfect shape worth |
|---|---|---|
| warm-future | +1.7 | **+9.3** |
| cold-interp | **+18.5** | +2.0 |

Cold-slice intuition ("level dominates") was being applied to the slice carrying 70.5% of
the score, where it is false. This is why the cold work and the warm work needed opposite
fixes.

## Why seasonal matching beats recency

The hour × dow profile drifts year over year: the same months a year apart disagree by
0.157 log against a 0.097 within-year noise floor (`shape_drift.py`). So the profile is
not noisy, it is *dated* — which is why smoothing and low-rank denoising did nothing and
seasonal matching did a lot.

On the deployment arm, estimating the May–Jun 2025 profile (`deploy_arm.py`):

| estimator | profile error |
|---|---|
| 2024 only | 0.1412 |
| month kernel w=2 | 0.1423 |
| pooled (what we deployed) | 0.1702 |
| recency, 365d half-life | 0.1741 |
| 2025 only (most recent) | 0.2434 |

Including the recent-but-seasonally-mismatched Jan–Apr 2025 data made the estimate
*worse* than ignoring 2025 entirely. This is the smoothed form of FHWA's month-of-year
seasonal adjustment factors — see sources below.

## Tuning that mattered, and tuning that did not

Once the backbone became seasonal, two inherited constants were wrong:

- **`n_estimators=350` on the residual learner** was tuned against a backbone with no
  seasonal term. With a smaller residual to fit, 350 trees overfit: on the May-Jun window
  the deployed setting added nothing over the backbone (82.2% both). At 120 trees it adds
  +0.7 GEH / -2.2 MAE. Across two windows: 82.15% backbone, 82.95% at n=350, **83.25% at
  n=120**. The margin over n=350 is inside noise; the margin over the backbone alone is not.
- **`k_shape=3` was too much shrinkage for the seasonal term** but is right for the shape
  hierarchy that cold stations depend on, so the two now use separate constants
  (`k_season=1.0`). Sweeping the shared constant: k=1 82.7%, k=3 82.2%, k=6 81.5%, k=12 80.4%.

What did *not* help: shrinking each pair's seasonal pattern toward k-means **factor-group**
centroids, which is NCHRP practice. At 8 groups with alpha=0.25 it reaches 82.4% against
82.7% for simply shrinking less. The per-pair seasonal signal is strong enough that
borrowing from a group dilutes it.

## Negative results (do not redo these)

- **Collaborative filtering for cold-start shape** (`cf_shape.py`): SVD + factor regression
  beats the deployed graph transfer 0.2309 vs 0.2584 (+10.7%) with half the fold variance —
  but an *oracle* shape is worth only ~+2 GEH points on cold-interp, so this converts to
  roughly +0.6. Real, banked, small.
- **CF learned similarity for the contemporaneous shock** (`cf_residual.py`): explains 4.97%
  of the shock against 20.24% for the plain network median and 16.81% for graph neighbours.
  The shock is a *global* phenomenon (weather/season), not a local one. `nbr_net` already
  captures it.
- **Low-rank denoising / empirical-Bayes shrinkage of warm profiles** (`warm_shape.py`):
  nothing, once the empty-cell fill is held constant. The earlier "+13%" was entirely an
  artifact of how unobserved cells were filled.
- **Recency weighting** of the profile: +2.2% at best (365d half-life), catastrophic when
  aggressive. Seasonal match dominates.
- **Shape-drift extrapolation.** The backbone extrapolates *level* drift via matched-month
  ratios, so the profile analogue looked obvious: estimate this year's shape drift on the
  calendar window the two years share, and carry a fraction into the target season. On a
  standalone proxy it looked excellent — profile error 0.1432 → 0.1087 (−24.1%) at
  alpha=0.5. **End to end it was worse**: 82.7% → 81.5% GEH<5. The proxy compared against
  a "matched season, previous year only" baseline, but the seasonal kernel already pulls in
  this year's nearby-season data (April is one bin from May), so the drift was already
  captured and the explicit term double-counted it. Reverted. A cautionary case for
  trusting proxy metrics over end-to-end measurement.

## A calibration bug worth remembering

`run_validation` fits the conformal offsets and the reliability model; `submit` generates
the forecasts. They had drifted apart — validation still used the old level path and
LightGBM alone, while `submit` used `PairLevelTransfer` and a GNN/LightGBM blend. Every
interval was therefore calibrated on residuals from models we no longer ship, which
directly corrupts the 15% uncertainty component. Both now use identical components.

Effect of fixing it, plus the better models underneath: cold-slice conformal offsets
tightened from −1.78/+0.76 to −1.04/+1.00 log at the same 90% coverage, and the reliability
meta-model improved from AUC 0.794 / Brier 0.179 to **AUC 0.810 / Brier 0.172**.

## Structural gaps in the data

- **Hours 0–1 in months 7–12 are never observed anywhere.** 2024 covers hours 2–23 only;
  hours 0–1 exist solely in Jan–Jun 2025. That is 12,295 target rows (7.2% of the
  submission) whose (hour, month) combination has no precedent. `Backbone.predict` mapped
  them to a zero hour × month interaction, discarding a systematic ×1.06 seasonal shift;
  `_complete_hour_month` now borrows the seasonal delta from adjacent hours.
- **Post-outage rows are partial observations.** On the first recorded hour after a ≥24h
  gap, the true value has median 67 veh/h where the model predicts ~613 — the sensor
  returns mid-hour and the record captures a fraction of it. GEH<5 is 0.2% and coverage
  0.6% there. No forecast can match a partial count; this belongs in the reliability score,
  and it is a question for DelDOT (`outputs/deldot_questions.pdf`, Q8).

## Cold level prior — ridge beats an aadt_band median

`PairLevelTransfer` blends graph donors with an attribute prior, and for a pair with no
direction-matched donor that prior *is* the estimate — roughly half the cold set.
Leave-one-station-out over the 75 labelled stations, mean |log level| error:

| prior | all | w/ corridor donor | no corridor donor |
|---|---|---|---|
| aadt_band median (was deployed) | 0.4492 | 0.2647 | 0.5590 |
| **ridge on the static vector** | **0.4316** | 0.2915 | **0.5151** |
| random forest | 0.4569 | 0.2636 | 0.5720 |
| gradient boosting | 0.5083 | 0.2902 | 0.6382 |

Ridge wins by 7.9% exactly where the prior carries the estimate, and both tree models are
worse than the median they were meant to beat — 75 stations is not enough to fit them, the
same small-n lesson as the factor-group attempt.

**But it does not survive contact with the estimator.** Inside `PairLevelTransfer` the prior
is blended with graph donors at `prior_k=1`, so for most pairs it barely carries weight:
0.3367 with the ridge against 0.3378 with the band median across the 4 folds, better on only
2 of them. Reverted. The standalone measurement was answering a different question than the
one that matters — a recurring theme in this session, alongside the shape-drift result.

## GNN capacity is saturated

Twelve configurations on the deployment arm (`sweep_gnn_deploy.py`), width 192–768, head
384–768, node dropout 0.2/0.4: every one lands between 83.04% and 83.51% blended GEH<5,
against 83.46% for the deployed setting, and the wider heads are consistently *worse*. The
best is the **smallest** config (hidden=192, head=384, dropout=0.2) at 83.51% / MAE 90.3 —
a 0.05-point margin that is the max over twelve configs on one split, i.e. selection noise,
not a result. The deployed setting was kept.

The earlier finding that width was worth 10.7 points was measuring the **missing `base`
input**, not the architecture — once the GNN is handed the backbone, capacity stops
mattering. No further tuning warranted here.

## The 50/50 blend weight is right, and the evidence is that the windows disagree

| w_gnn | May–Jun 2025 | Mar–Apr 2025 | mean |
|---|---|---|---|
| 0.00 (LightGBM only) | 82.92% | 83.41% | 83.17% |
| 0.25 | 83.22% | 83.41% | 83.32% |
| **0.50 (deployed)** | 83.46% | 83.32% | **83.39%** |
| 0.75 | 83.51% | 83.13% | 83.32% |
| 1.00 (GNN only) | 83.44% | 82.88% | 83.16% |

Each window has a clear optimum and they point in **opposite directions** — 0.75 on one,
0.0 on the other. The equal blend is best on average and is the only choice that does not
depend on which window you happened to tune against. Left at 0.5.

## Reliability model — AUC 0.794 → 0.851

The reliability model saw only the slice label, the point forecast and the interval width.
The robustness slices say that is the wrong feature set: accuracy varies enormously by hour
(night 60.6% vs 72.3% overall), holiday (66.3%), season (winter 68.0%) and how much history
the pair has (64.9%) — all knowable in advance and none of it visible to those features.
Adding hour, day-of-week, month, weekend/holiday flags, a night flag and per-pair history
depth took it from **AUC 0.794 / Brier 0.179 to AUC 0.851 / Brier 0.153**.

**A train/serve skew this introduced, and how it showed up.** The `pair_n` (history depth)
feature initially collapsed cold-slice reliability to 0.06, against a realistic 0.3–0.5
given cold within-20% runs 32–39%. Cause: in validation folds B and C the "cold" stations
are held-out *warm* stations, so counting history from the full labelled panel gave them
thousands of rows at fit time, while a real cold pair has zero at serve time. The counts are
now taken from each fold's own training rows, which makes them 0 for held-out pairs —
matching deployment. Worth remembering: any feature derived from "how much data do we have
about this unit" has to be computed from what the *fold* could see, not what the panel holds.

The skewed version also *scored better* on the held-out split — AUC 0.925 / Brier 0.102
against 0.851 / 0.153 once fixed — because `pair_n` was partly telling the model "this row
is a held-out warm station" rather than predicting difficulty. A metric improving is not
evidence a feature is sound; the implausible 0.06 cold reliability is what exposed it.

## Robustness profile (4 folds, warm-future GNN)

Deltas are against that model's own 72.3% overall.

| weakest slices | | strongest slices | |
|---|---|---|---|
| post-outage (≥24h gap) | 0.1% (−72.2) | no corridor neighbour | 76.0% (+3.7) |
| night hours | 60.6% (−11.7) | summer | 74.9% (+2.6) |
| limited history pair | 64.9% (−7.4) | unreliable AADT band | 74.5% (+2.2) |
| holiday ±1d | 66.3% (−6.0) | peak hours | 73.1% (+0.8) |
| winter | 68.0% (−4.3) | functional_class missing | 72.2% (−0.1) |

Worth saying explicitly in the write-up: the model is **not** weak on stations with missing
context — no corridor neighbour, unreliable AADT band and missing `functional_class` all
score at or above the overall number. The cold-start weakness is about *zero history*, not
*missing attributes*.

The reliability model was given the features that predict these slices (hour, night flag,
month, holiday, per-pair history depth); it previously saw only the slice label, the
prediction and the interval width, which is why it could not express any of this.

## Graph usage audit, and the seasonal-transfer hole (2026-08-18)

Audit of how much of `network_edges.csv` each consumer actually uses:

| consumer | 1-hop | 2-hop | routes through stations with no labels and no targets |
|---|---|---|---|
| `PairLevelTransfer` (cold level) | yes, direction-matched | fallback only | yes |
| `neighbour_features` (shock) | yes | fallback only | yes |
| `GraphShapeTransfer` (hour x dow) | yes | **no** — falls to the direction-global mean | — |
| `Backbone.transfer_season` | yes | **no** — fell through to a *zero* correction | — |
| `gnn_v2` R-GCN | yes, 5 relations | yes, 3 layers | **no** — not in the node set |

Facts behind it:

- **9 of the 127 stations** in `network.csv` are neither observed nor targets, and **22 of
  the 186 edges** touch one. The level and shock paths hop through them correctly because
  they only test whether the *far* endpoint is labelled. `gnn_v2.build_pair_graph` builds
  its node set from `set(lab.pair) | set(tgt.pair)`, so those 22 edges silently vanish from
  the R-GCN. No cold station depends on such a station for its only donor (0 of 43), and
  the R-GCN currently serves warm rows only, so the cost today is ~0 — but it would matter
  for any graph model taking over a cold slice.
- **`transfer_season` had no fallback at all.** With the full labelled panel, 20 of the 86
  cold pairs — **12,247 rows, 7.2% of the submission** — had no direction-matched 1-hop
  donor and therefore kept `shape_month = 0`, while warm pairs got a term worth about
  +4.9 GEH points and of typical magnitude 0.110 log.

### Neighbour lags do not help the contemporaneous shock

`neighbour_features` reads the exact timestamp only. Ridge on neighbour residuals, 69
stations, temporal 70/30 hold-out, R² of the pair's own shock:

| neighbour features | R² | win rate vs `t` only |
|---|---|---|
| `t` only (deployed) | 0.4882 | — |
| `t-1, t` | 0.4898 | 49% |
| `t-2, t-1, t` | 0.4906 | 54% |
| `t-1, t, t+1` | 0.4903 | 49% |
| `t-2 … t+2` | 0.4912 | 52% |

Coin-flip win rates on a 0.49 base. Hour-to-hour persistence already sits inside the
hour x dow profile, so lagged neighbours are collinear with the backbone. `t+1` would also
be future information relative to the predicted timestamp. Leave it at `t`.

### 2-hop shape transfer: real but too small to ship

4 protocol folds, the 22 held-out pairs that fall through to the direction-global shape and
have a 2-hop direction-matched donor available (mean |log| over hour x dow cells):

| estimator | error | wins |
|---|---|---|
| direction-global fallback (deployed) | 0.3365 | — |
| 2-hop direction-matched | 0.3153 | 50% |
| 50/50 mix with the global fallback | 0.2740 | 68% |

An oracle shape is worth ~+2 GEH on cold-interp and this touches ~18% of cold pairs, so the
end-to-end value is about **+0.1 GEH**. Not implemented.

### The seasonal fallback chain: implemented, and it is GEH-neutral

`transfer_season` now takes four rungs, first non-empty wins, donors always drawn from
pairs whose surface was actually *fitted* (so the result no longer depends on iteration
order): direction-matched 1-hop → **any-direction 1-hop** (weight x0.5) → direction-matched
2-hop (x0.25) → the network-average surface. Coverage of the 86 cold pairs goes 62 / 20 / 2
/ 2, i.e. **0 rows are left with a zero seasonal term**, down from 12,247.

Rung 2 borrows across directions because the seasonal surface barely depends on direction —
mean |log| difference between two seasonal surfaces:

| pair of surfaces | difference |
|---|---|
| corridor neighbour, same direction | 0.0506 |
| corridor neighbour, **opposite** direction | 0.0654 |
| same station, opposite direction | 0.0715 |
| random pair | 0.2808 |
| *magnitude of the term itself* | *0.1097* |

An opposite-direction corridor donor lands closer to the truth (0.0654) than leaving the
term at zero (0.1097) does.

End to end, 4-fold protocol, `--no-gnn`, matched runs before and after
(`protocol_seasonfix.json` vs the same command at the previous commit):

| arm | GEH<5 | MAE | fold sd |
|---|---|---|---|
| cold-interp, deployed config | 43.58 → **43.48** (−0.10) | 290.5 → **289.6** | 5.11 → 5.27 |
| cold-future, deployed config | 37.22 → **37.32** (+0.10) | 332.1 → **331.4** | 5.24 → 5.00 |
| cold-interp backbone | 36.23 → 36.62 (+0.39) | 332.4 → 331.0 | 5.93 → 5.06 |
| cold-future backbone | 33.50 → 34.00 (+0.50) | 350.2 → 349.1 | 5.38 → 4.78 |
| warm-future (both arms) | unchanged to 2 d.p. | unchanged | unchanged |

**Read this honestly: on the deployed cold configuration the GEH effect is inside noise**
(+0.90 / −1.20 / +0.20 / −0.30 per fold on cold-interp). MAE improves on 9 of the 10 cold
arms and fold spread narrows on most, and the effect is clearly positive on the *backbone*
arms where the seasonal term is not competing with the pair-level transfer. Kept, because
it removes a structural discontinuity — a term silently set to zero for 7.2% of the
submission — at no measured cost, not because it moves the score. It is consistent with the
oracle decomposition: a perfect *shape* is worth only ~+2 GEH on cold slices, so a partial
recovery of a shape-like term cannot be worth much there.

## `splits.json` was never the split anything ran on (2026-08-18)

Three documents state that `splits.json` holds the frozen station splits and that
"everyone must use these". Two things are wrong with that:

- **No code ever read the file.** `validate_protocol.make_splits()` regenerated the report
  set and the folds from `numpy.random.default_rng(seed)` at every run.
- **It describes a different partition.** `splits.json` records `A_fit` (45) /
  `B_select` (15) / `C_report` (15); the protocol runs a 15-station report set plus a
  4-fold CV over the remaining 60. The two report sets are not the same stations.

So anyone who followed the documented instruction would have produced numbers comparable
to nobody, and reproducibility rested on `default_rng().permutation` being stable across
numpy versions, which is not a guarantee numpy makes.

Fixed by generating `splits_frozen.json` **from the same function and seed that produced
every number in this file**, committing it, and making `make_splits()` load it and warn
loudly if it ever falls back to regenerating. Verified identical to the runtime splits, so
every result already published here stays valid. `splits.json` is kept for provenance and
marked as not describing these folds.

## The hour 0-1 fallback: a 558% WAPE hole, found by comparing against a simpler model (2026-08-18)

The organisers confirmed the primary automated forecast metric is **WAPE**
(`sum|y - yhat| / sum|y|`), not GEH<5, and that scoring happens across their own conditions
rather than one pool. Re-scoring everything on WAPE surfaced a bug that GEH<5 had hidden
for the whole project.

On the 4-fold protocol, warm-future, a re-implementation of the far simpler multiplicative
model in `src/forecast_model.py` beat the seasonal backbone by **5.19 WAPE points**
(16.21% against 21.41%) while *losing* GEH<5 (68.7% against 71.7%). Decomposing by hour:

| warm-future, 4 folds pooled | backbone (before) | multiplicative | backbone (after fix) |
|---|---:|---:|---:|
| all hours | 21.41% | 16.21% | **13.75%** |
| hours 0-1 only | **558.24%** | 56.93% | **57.43%** |
| hours 2-23 | 13.07% | 15.58% | 13.07% |

Hours 0-1 are 8.3% of rows and 1.53% of the volume, and they accounted for **147.7% of the
gap** -- over 100% because excluding them reverses the ordering by 2.5 points. Mean bias at
hour 0 was **+765 veh/h** (hour 1: +827) against a true mean of 171.

**Cause.** The training clock runs 2h behind, so a fit on 2024 alone contains no hours 0 or
1 at all (`min(hour) == 2`); they exist only in the Jan-Jun 2025 window. `shape_glob` is the
*base* of the shape hierarchy, so a missing key there is not a missing correction but a
missing hour-of-day profile: `predict` mapped it to NaN, `nan_to_num` made it 0.0, and the
model returned the pair's **whole-day average level** for a 01:00 hour.

**Fix.** `_complete_hour_dow` -- circular linear interpolation across the hour axis per
day-of-week, so an unobserved 01:00 is filled from 23:00 and 02:00. It is a no-op when every
hour is observed, and hours 2-23 are unchanged to two decimals, so the fix is surgical.

**Scope.** The deployment arm and the shipped submission both hold the 2025 H1 half-year, so
hours 0-1 are present in the fit there and neither was affected -- which is exactly why the
backbone won the deployment arm (10.87% vs 16.13% WAPE) while losing the protocol arm. The
bug was invisible on the arm that mirrors the real task, and visible only on the arm that
hides half a year.

Two lessons worth keeping:

- **GEH<5 masked it.** GEH is a hit rate and low-volume rows pass it easily, so predicting
  940 where the truth is 171 cost almost nothing on the metric we had been selecting on for
  the entire project. A 5-point WAPE hole sat behind a 3-point GEH advantage.
- **A weaker model is a diagnostic.** Nothing in our own validation flagged this. It
  surfaced only because a much simpler competing model -- five multiplicative factors, no
  shrinkage hierarchy, no seasonal kernel, no conformal layer -- beat us on the real metric,
  and the only honest way to explain that was to find out where.

Also worth stating plainly: this is precisely the failure the robustness component now
targets. The organisers confirmed robustness assesses "graceful behavior under
degraded/incomplete inputs" and that we define the stress tests. A model that returns a
daily average when one hour-of-day is missing from training is the textbook case, and we
would have shipped it.

## Corrected training data, and what it changed (2026-08-18)

The organisers confirmed a data-preparation issue in the originally distributed 2024
training file, caused by a difference in the underlying TMAS source-file layout between 2024
and 2025: the 2024 hourly observations were **offset by two hours**, and the original
**midnight and 01:00 observations were omitted**. A corrected file was issued. Validation,
network files and the submission template are unchanged.

This is the bug this project detected from the data alone (`quickstart_exploration.ipynb`),
independently found by the other half of the team, and raised with the organisers.

### Verification before trusting the reissue

Not assumed — checked (`scratchpad/verify_corrected.py`, three independent tests):

| check | old file | reissued file |
|---|---|---|
| rows | 902,902 | **984,984** (+82,082) |
| distinct hours in 2024 | 0–21 (22) | **0–23, none absent** |
| shipped `hour` column vs timestamp | agrees (on the wrong clock) | **agrees on 100% of rows** |
| best 2024↔2025 profile lag | **+2** (err 0.1775 at +2, 0.6024 at 0) | **0** (err **0.0515**; 0.3259 / 0.3288 at ∓1h) |
| old rows, shifted +2h, vs new | — | **902,902 / 902,902 volumes identical** |
| rows present only in the new file | — | 41,041 at hour 0, 41,041 at hour 1 — *nothing else* |

So the correction is exactly what was described: the same observations, re-aligned, plus the
two restored hours. Our original diagnosis was right in both direction and magnitude.

### The hardcoded offset became the hazard

`CLOCK_SHIFT_H = 2` was correct for the old file and is **wrong** for the new one — it would
put us two hours off in the other direction, and a consistently-wrong clock produces smooth,
plausible profiles. That is precisely how the original bug survived for weeks.

Replaced with `detect_clock_offset()`: centre each pair's log hour-of-day profile in both
files, then pick the circular shift of the training profile minimising mean |difference|
against validation. Measured on every load, warns if not 0. It returns **+2 on the archived
old file and 0 on the reissued one**, so it is verifiably sensitive to the thing it guards.
This is also a concrete data-quality control to point at for the operational-readiness
component, which the organisers confirmed is about monitoring and data quality rather than
deployed infrastructure.

### Structural gap closed

2024 now has **41,610 rows at hours 0–1 in months 7–12**. The 12,297 target rows in that
cell (7.2% of the submission) previously had *no precedent anywhere in the data*; they now
have direct precedent. `_complete_hour_month` is correspondingly less load-bearing.

### Results, corrected data, 4-fold protocol

| model / arm | before (old data) | after |
|---|---|---|
| warm-future backbone, GEH<5 | 71.7% | **79.5%** (sd 0.68) |
| warm-future backbone, MAE | 166.4 | **101.9** |
| warm-future backbone, R² | 0.848 | **0.948** |
| cold-interp `pairlevel_cfshape`, GEH<5 | 43.6% | **45.0%** |
| cold-future `pairlevel_cfshape`, GEH<5 | 37.2% | **43.4%** |
| deployment arm, backbone GEH<5 | 82.7% | **83.3%** |

Harness leaderboard, **WAPE** (primary metric, lower is better):

| protocol arm | warm-future | cold-interp | cold-future |
|---|---:|---:|---:|
| `backbone_seasonal` | **13.11** | 34.29 | **36.34** |
| `slice_routed_lgbm` | 14.35 | **34.03** | 36.65 |
| `team_v1_reimpl` | 15.55 | 44.38 | 47.51 |

| deployment arm | WAPE | GEH<5 | MAE | cov90 | Winkler |
|---|---:|---:|---:|---:|---:|
| `slice_routed_lgbm` | **10.55** | 83.8% | 90.7 | 0.898 | 640 |
| `backbone_seasonal` | 10.78 | 83.3% | 92.6 | 0.871 | 679 |
| `team_v1_reimpl` | 15.09 | 71.5% | 129.7 | 0.868 | 866 |

Warm-future WAPE went 21.41 → **13.11**: the `_complete_hour_dow` fallback fix carried
21.41 → 13.75, the corrected data carried 13.75 → 13.11. The multiplicative model improved
too (16.21 → 15.55) but far less, because it had no hour-0/1 hole to lose. The ordering that
prompted the whole investigation has reversed, and we now lead every slice on both metrics.

### A claim that did not survive the metric change

The oracle decomposition previously supported "the levers point in **opposite** directions:
warm-future is a shape problem (+9.3 shape against +1.7 level), cold start is a level problem
(+18.5 level against +2.0 shape)." Re-run on corrected data and converted to WAPE:

| perfect component handed to the model | WAPE | GEH<5 |
|---|---:|---:|
| *nothing* (current backbone) | 13.11% | 79.5% |
| + month | 12.79% | 80.3% |
| + shape | 12.40% | 81.0% |
| + **level** | **12.28%** | 81.4% |
| + level + shape + month | 11.03% | 84.6% |
| full oracle pair×hour×dow×month | **8.48%** | 88.6% |
| + true network shock | 7.43% | 91.9% |

**Under WAPE, level is worth slightly more than shape on warm-future** (−0.83 against
−0.71) — the reverse of the GEH reading, because WAPE weights an error by its absolute size
and level error dominates at high volume, while GEH is a hit rate that low-volume rows pass
easily. The cold-slice decomposition has **not** been re-measured in WAPE; treat its GEH
figures as indicative. Cold-future still has no measured ceiling in either metric.

The general lesson, which is the third instance of it in this project: **a quantity measured
in the wrong units can invert.** Levers, ceilings and model rankings all need restating in
WAPE before they are trusted.

## Sources

- [FHWA Traffic Monitoring Guide (2016)](https://www.fhwa.dot.gov/policyinformation/tmguide/tmg_fhwa_pl_17_003.pdf) — month-of-year seasonal adjustment factors, factor groups
- [FHWA traffic monitoring methodologies](https://www.fhwa.dot.gov/policyinformation/tmguide/tmg_2013/traffic-monitoring-methodologies.cfm)
- [NCHRP, Guide on Methods for Assigning Counts to Adjustment Factor Groups](https://www.nationalacademies.org/read/27925/chapter/3) — mis-assignment can triple AADT error, hence soft assignment
- [Esugo, Haas & Lu, GEH-based loss function, TRR 2025](https://journals.sagepub.com/doi/10.1177/03611981241274645) — GEH-shaped differentiable loss
- [Shao et al., STID, CIKM 2022](https://arxiv.org/abs/2208.05233) — identity embeddings vs spatio-temporal GNNs
- [Montero-Manso et al., FFORMA, IJF 2020](https://www.sciencedirect.com/science/article/abs/pii/S0169207019300895) — feature-based forecast combination
- [Wickramasuriya et al., MinT, JASA 2019](https://robjhyndman.com/papers/mint.pdf) — hierarchical reconciliation
