# CLAUDE.md

Claude Code and other agents: **read [`AGENTS.md`](AGENTS.md) in this folder first and
follow it exactly.** It is the working contract for `benchmark/` — the frozen splits, the
two evaluation arms, the model interface, the reporting rules, and the list of things
already measured and already ruled out.

Short version, so nothing below surprises you:

- `traffic_pipeline.load_all()` for data. It applies the +2h training clock fix and the
  `direction_code` dtype canonicalisation, both of which fail **silently** otherwise.
- Splits come from `splits_frozen.json`. Never regenerate them. `splits.json` is a stale,
  different partition kept only for provenance.
- `protocol` and `deploy` numbers are never comparable to each other.
- Cold-slice results are reported as a 4-fold mean **with** the spread. A single fold is
  noise there.
- The report set (15 stations) is opened once, at the very end. Do not tune on it.

Add a model with `benchmark/models/<name>.py` defining `build()`, then run
`benchmark/harness.py --model ... --arm protocol|deploy`.
