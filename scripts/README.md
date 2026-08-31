# scripts

`scripts/` is an operational toolbox, not the formal experiment layer.

- `check/` — smoke tests and contract verification.
- `export/` — model fine-tune/export and artifact generation, usually run in each model's own conda environment.
- `legacy/` — archived or closed experiment drivers kept for reproducibility/reference.

New experiment matrices should go through `configs/exp/*.yaml` +
`python -m experiments.run ...`; active one-off experiment drivers belong under
`experiments/<line>/`, not here.
