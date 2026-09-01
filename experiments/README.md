# experiments

`experiments/` is the formal experiment layer.

Preferred path for new work:

```bash
python -m experiments.run configs/exp/<name>.yaml --report
```

Directory roles:

- `run.py` — config-driven experiment runner (cell gen in `data.split`; method registry is `run.py`'s `METHOD_REGISTRY`).
- `distill/` — KD / few-shot Pearson / LOSO distillation drivers (incl. `analyze_fewshot_pearson.py`).
- `bigmodel/` — native big-model adaptation and tuning drivers (CBraMod / LaBraM / MIRepNet).
- `bidir/` — bidirectional / CR-AMD / BD-EEG / feature-level mutual drivers (line closed as null, kept for reproduction).
- `mask/` — wrong-sample utilization driver (`run_wrong_sample.py`).

The D0-onward line (fusion / router / balance-gate / target-support) is archived;
recover it from the `pre-consolidation` git tag.

Reusable collaboration algorithms belong in `collab/`. Operational tools belong
in `scripts/` (`check/`, `export/`).
