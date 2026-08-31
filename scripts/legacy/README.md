# scripts/legacy — archived experiment drivers

This directory is for archived experiment code and one-off historical drivers.
It is kept for reproducibility/reference, not as the place to add new experiments.

Superseded by:

- **`experiments/`** — formal experiment entry points, including the config runner
  (`python -m experiments.run configs/exp/<x>.yaml`) and active line-specific
  drivers under `experiments/{distill,fusion,adapt,bigmodel}/`.
- **`eval/`** — `python -m eval '<metrics_glob>'` / `report_contrasts` replaces
  bespoke `analyze_*.py` / `aggregate_*.py` paired-stats code.

Archived contents:

- `bidir/` — bidirectional / CR-AMD / BD-EEG / feature-level mutual drivers,
  kept because the line was closed as null.
- `wrongsample/` — wrong-sample utilization E0-E5, kept for closed/null results.
- `run_*.sh`, `analyze_*`, `aggregate_*` — earlier orchestration and reporting
  helpers. They may encode settled ablation configs, but are not maintained as
  primary entry points.
