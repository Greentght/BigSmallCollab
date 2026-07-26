# scripts/legacy — archived experiment drivers

One-off shell drivers (`*.sh`) and analysis scripts (`analyze_*`, `aggregate_*`)
from earlier experiment rounds, kept for reproducibility/reference. **Superseded by:**

- **`experiments/`** — config-driven runner (`python -m experiments.run configs/exp/<x>.yaml`)
  replaces the ad-hoc `run_*.sh` orchestration.
- **`eval/`** — `python -m eval '<metrics_glob>'` / `report_contrasts` replaces the
  bespoke `analyze_*.py` / `aggregate_*.py` paired-stats code.

These still encode specific settled ablation configs. They are **not maintained**:
paths inside them (e.g. `python scripts/analyze_x.py`) assume the pre-archive layout,
so adjust paths before rerunning. The `.py` entry points they call (`run_distill.py`,
`run_bdeeg_loso.py`, …) remain in `scripts/`.
