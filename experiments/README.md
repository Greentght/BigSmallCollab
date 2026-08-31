# experiments

`experiments/` is the formal experiment layer.

Preferred path for new work:

```bash
python -m experiments.run configs/exp/<name>.yaml --report
```

Directory roles:

- `run.py`, `protocols.py`, `methods.py` — config-driven experiment runner.
- `distill/` — active distillation drivers that have not yet been collapsed into YAML.
- `fusion/` — active ensemble, fusion, and routing drivers.
- `adapt/` — target-support / few-shot adaptation drivers.
- `bigmodel/` — native big-model adaptation and tuning drivers.

Reusable collaboration algorithms belong in `collab/`. Operational tools belong
in `scripts/` (`check/`, `export/`, `legacy/`).
