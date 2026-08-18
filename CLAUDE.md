# CLAUDE.md

**Read [`AGENTS.md`](AGENTS.md) at the repository root first.** It is the single entry point:
what this repo is, what the organisers have confirmed, the five things that will silently
ruin your results, how the solution was built and why, and where every other document lives.

Before writing or changing any model, also read [`benchmark/AGENTS.md`](benchmark/AGENTS.md)
— the working contract (frozen splits, the two evaluation arms, the model interface, the
reporting rules, and the negative results already ruled out).

Three facts that catch people immediately:

- **The primary metric is WAPE**, not MAE and not GEH<5. They rank models differently.
- **`traffic_train.parquet` was reissued on 2026-08-18.** Any hardcoded `+2h` clock offset is
  now wrong. `traffic_pipeline.load_all()` detects the offset and warns.
- **Never regenerate the CV splits** — read `benchmark/splits_frozen.json`.
