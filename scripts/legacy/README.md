# scripts/legacy — canonical launchers for re-runnable experiment lines

This directory keeps the `run_*.sh` launchers that encode the settled command
+ hyperparameters for each experiment line (mask / dkd / eakd / relational /
proto / adaptive / bidir / cramd / bdeeg / loso / cbramod / labram ...). They
are the authoritative record used by REPRO.md — not one-off leftovers.

The python drivers they call now live under `experiments/`:

- `experiments/bidir/` — bidirectional / CR-AMD / BD-EEG / feature-level mutual
  (line closed as null, kept for reproduction)
- `experiments/mask/run_wrong_sample.py` — wrong-sample utilization E0-E5
- `experiments/distill/` — KD / few-shot Pearson / LOSO distill drivers
- `experiments/bigmodel/` — CBraMod / LaBraM / MIRepNet native adaptation + tuning

Superseded reporting helpers (deleted): bespoke `analyze_*.py` /
`aggregate_*.py` paired-stats code is replaced by `python -m eval
'<metrics_glob>'` (`eval/stats.py`).
