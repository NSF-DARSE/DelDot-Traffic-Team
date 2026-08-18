# Case Study: Traffic Pattern Analysis

## Overview
Analyze the supplied network traffic data and develop a method for forecasting
future hourly traffic volume across anonymous monitoring locations.

Evaluation also considers limited-history locations, difficult operating
periods, incomplete observations, uncertainty, and supplied scenario questions.

## Expected Solution Guidelines

## Modelling and model comparison — `benchmark/`

[`benchmark/`](benchmark/) adds a comparison harness for the three-slice problem, so two
people can build different models and compare them by reading two numbers rather than
arguing about whose validation was fairer. It leaves `src/`, `docs/` and `output/`
untouched.

- [`benchmark/AGENTS.md`](benchmark/AGENTS.md) — **start here.** The working contract:
  frozen splits, the two evaluation arms, the model interface, the reporting rules, and
  what has already been ruled out. Written to be read by an LLM agent as much as a person.
- [`benchmark/LEADERBOARD.md`](benchmark/LEADERBOARD.md) — current standings, both arms.
- [`benchmark/FINDINGS.md`](benchmark/FINDINGS.md) — every measured result, with the script
  that produced it.

```bash
P=~/.virtualenvs/data_science/bin/python
$P benchmark/harness.py --model benchmark/models/backbone.py --arm protocol
$P benchmark/harness.py --leaderboard
```
