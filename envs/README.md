# Conda environments

Each model runs in its **own** conda env (their pinned deps conflict — e.g.
MIRepNet's numpy/mne pins vs LaBraM's timm==0.4.12 + tensorboardX). The framework
never imports two models at once; the cross-env hub talks through cached
artifacts (`results/artifacts/`), so the envs never need to coexist in one process.

| env | covers (adapters) | provenance |
|---|---|---|
| `mirepnet` | ifnet, eegnet, adfcnn, mirepnet | existing MIRepNet env (`conda activate mirepnet`) |
| `cbramod`  | cbramod | existing CBraMod env |
| `labram`   | labram | **cloned from `cbramod`** + `pip install timm==0.4.12` |

Recreate the `labram` env:

```bash
conda create -y -n labram --clone cbramod
conda run -n labram pip install timm==0.4.12
```

Pretrained weights are resolved via `config.weight_path()` under `weights/` (override
with `MIREPNET_WEIGHT` / `CBRAMOD_WEIGHT` / `LABRAM_WEIGHT`).
Vendored deps the framework adds on top of a base EEG env: `pyyaml`, `scipy`,
`scikit-learn` (already present in all three envs).
