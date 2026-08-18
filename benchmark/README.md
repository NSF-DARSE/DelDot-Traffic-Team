# benchmark/ — the three-slice problem, and a way to compare models on it

This folder adds a **model comparison harness** and the modelling work behind it. It does
not change anything in `src/`, `docs/` or `output/` — the delivered pipeline is untouched.

The point is narrow: make it possible for two people to build different models and compare
them by reading two numbers, instead of arguing about whose validation was fairer.

```bash
P=~/.virtualenvs/data_science/bin/python

$P benchmark/harness.py --model benchmark/models/backbone.py --arm protocol
$P benchmark/harness.py --model benchmark/models/backbone.py --arm deploy
$P benchmark/harness.py --leaderboard        # regenerate LEADERBOARD.md
```

**New here? Start with [the repository-root `AGENTS.md`](../AGENTS.md)** — the single entry
point for the whole project. Then read [`AGENTS.md`](AGENTS.md) in this folder — the working
contract, written to be read by an LLM agent as much as by a person. [`LEADERBOARD.md`](LEADERBOARD.md) is the current
standings. [`FINDINGS.md`](FINDINGS.md) is every measured result with the script that
produced it.

## Why the target file is three problems

Of the 118 stations in `submission_template.csv`, **43 have no labelled history at all**.
That splits the 170,956 target rows into three tasks that need opposite things:

| slice | rows | share | own history | network observed at t |
|---|---:|---:|---|---|
| warm-future | 120,571 | 70.5% | 18 months | no (future) |
| cold-interp | 23,810 | 13.9% | **none** | **yes** — reconstruct the past from the network |
| cold-future | 26,575 | 15.5% | **none** | no — both problems at once |

Cold start is dominated by **level** error; warm-future is mixed. Which component matters
most depends on the metric you ask: under WAPE a perfect level is worth slightly more than a
perfect shape on warm-future (−0.83 against −0.71 WAPE points), while under GEH<5 the same
decomposition says the opposite (+9.3 shape against +1.7 level). See `AGENTS.md` §8 — any
lever quoted in GEH may not survive conversion to WAPE.

## What is in here

| | |
|---|---|
| `harness.py` | folds, metrics, results, leaderboard. ~300 lines, no model logic. |
| `models/backbone.py` | the profile backbone alone — the reference row |
| `models/slice_routed_lgbm.py` | backbone + LightGBM quantile residual, routed by slice |
| `models/team_v1.py` | the `src/forecast_model.py` architecture, re-expressed against the harness contract so it can be scored on the same folds (see the caveats in its docstring) |
| `splits_frozen.json` | the frozen folds — **read, never regenerate** |
| `FINDINGS.md`, `PROBLEM_BRIEF.md` | measured results, and the long-form description of the data |
| `results/published/` | the result JSONs behind the numbers quoted in `FINDINGS.md` |
| `slides/` | `slides_models.pdf` (the model, slice by slice), `slides_protocol.pdf` (the validation design), `slides_traffic.pdf` (overview) |

## Two things worth knowing before you touch the data

1. **`traffic_train.parquet` was reissued on 2026-08-18.** The original 2024 file was offset
   +2h and was missing every 00:00 and 01:00 record — a TMAS source-layout difference between
   2024 and 2025, which we detected from the data and the organisers then confirmed and fixed
   at source. **Any hardcoded `+2` offset is now wrong.** `detect_clock_offset()` measures it
   on every load and warns if it is not 0; the old file is archived as
   `traffic_train_2h_offset.parquet`.
2. **`direction_code` is `str` in the parquets and `int64` in the CSVs.** A cross-file
   merge silently returns zero rows.

`traffic_pipeline.load_all()` handles both. Use it rather than reading files directly.

## A correction this work turned up

`splits.json` is described in several places as "the frozen splits — everyone must use
these". **No code ever read it**, and the partition it records (`A_fit` / `B_select` /
`C_report`) is not the one `validate_protocol.py` actually runs. Anyone following that
instruction would have produced numbers comparable to nobody.

`splits_frozen.json` is now the real thing: generated once from the function and seed that
produced every published number, committed, and *loaded* by `make_splits()`, which warns
loudly if it ever falls back to regenerating. `splits.json` is left in place for
provenance only.

## What the organisers confirmed (2026-08-18)

- **The primary automated forecast metric is WAPE**, and scoring happens across
  organiser-defined conditions rather than one undifferentiated pool. The A/B/C framing was
  called "directionally reasonable". Everything in this repo was previously selected on
  GEH<5 — that needs re-checking, and `LEADERBOARD.md` now leads with WAPE.
- **`reliability_score` is a relative 0–1 confidence indicator**, not the probability of a
  specific event. Higher should mean more dependable. Interval calibration is evaluated
  separately through `lower_90` / `upper_90`, so the conformal layer is scored on its own.
- ⚠️ **The five scenario prompts come from the organisers and are common across teams,**
  and they asked explicitly that self-defined scenarios not be substituted. The five
  currently in `scenario_template_filled.csv` are our own constructions and must be
  replaced when the real prompts arrive.
- **Robustness definitions are ours to choose and justify.** The component assesses
  reproducible stress testing and graceful behaviour under degraded or incomplete inputs.

Still open: whether rows that are themselves **partial observations** are scored. On the
first recorded hour after a ≥24h gap the true value has a median of 67 veh/h where a model
predicts ~613 — the sensor returns mid-hour. No forecast can match a partial count.
