# experiments

`experiments/` is the formal experiment layer.

## YAML entrypoints

Distill and fusion both use the shared loader and the same runtime controls:

```bash
python experiments/distill/run_distill.py --config configs/experiments/distill_kd.yaml --gpu 0
python experiments/fusion/run_fusion.py --config configs/experiments/fusion_concat_mlp.yaml --gpu 0
```

There is one YAML per collaboration method:
distill_kd, distill_kd_masked, distill_mmd, distill_kd_mmd, distill_mi,
fusion_avg_prob, fusion_concat_mlp, and fusion_gate_conf_acc.

Each file has a `pairs` list with one entry per selected big-model/small-model
combination. Every pair owns `params` and `grid`; runtime training values can use
those same blocks. A nested `datasets.<dataset>` block can override them.
Distill automatically adds the student scratch baseline unless
include_baseline: false; fusion automatically adds big_only and small_only
unless include_controls: false. Base/control methods therefore do not get their
own YAML files.

`distill_mi.yaml` is a subject-wise few-shot pilot for MIRepNet → IFNet. It
always expands exactly `Base`, `KD_all`, and `CE_MI`; `lam_mi: 0.1` is fixed as
the pilot value and is not selected using evaluation data.

The loader merges model defaults, pair values, dataset overrides, and the current
grid cell, validates fields, expands dataset/protocol/pair/grid combinations, and
writes a sibling <name>.resolved.yaml next to the output CSV. Legacy direct CLI
flags remain available for historical reproduction but are deprecated.

Single-model finetune/export commands for the three big models. Dataset YAML files
only define dataset facts and shared split policy; model YAML files define
``finetune.defaults`` plus per-dataset/per-protocol hyperparameters.

```bash
DATASETS=(BNCI2014001 BNCI2014001-4 BNCI2014004 BNCI2015001 AlexMI)
GPU=7

# MIRepNet: fewshot
for ds in "${DATASETS[@]}"; do
  conda run -n mirepnet python experiments/finetune/finetune.py \
    --model mirepnet --dataset "$ds" --protocol fewshot --gpu "$GPU" \
    --out_csv "results/mirepnet_${ds}_fewshot.csv"
done

# MIRepNet: LOSO
for ds in "${DATASETS[@]}"; do
  conda run -n mirepnet python experiments/finetune/finetune.py \
    --model mirepnet --dataset "$ds" --protocol loso --gpu "$GPU" \
    --out_csv "results/mirepnet_${ds}_loso.csv"
done

# CBraMod: fewshot
for ds in "${DATASETS[@]}"; do
  conda run -n cbramod python experiments/finetune/finetune.py \
    --model cbramod --dataset "$ds" --protocol fewshot --gpu "$GPU" \
    --out_csv "results/cbramod_${ds}_fewshot.csv"
done

# CBraMod: LOSO
for ds in "${DATASETS[@]}"; do
  conda run -n cbramod python experiments/finetune/finetune.py \
    --model cbramod --dataset "$ds" --protocol loso --gpu "$GPU" \
    --out_csv "results/cbramod_${ds}_loso.csv"
done

# LaBraM: fewshot
for ds in "${DATASETS[@]}"; do
  conda run -n labram python experiments/finetune/finetune.py \
    --model labram --dataset "$ds" --protocol fewshot --gpu "$GPU" \
    --out_csv "results/labram_${ds}_fewshot.csv"
done

# LaBraM: LOSO
for ds in "${DATASETS[@]}"; do
  conda run -n labram python experiments/finetune/finetune.py \
    --model labram --dataset "$ds" --protocol loso --gpu "$GPU" \
    --out_csv "results/labram_${ds}_loso.csv"
done
```

`--out_csv` writes one result CSV with `subject,seed,test_acc_pct`. The
`mean_acc_pct` / `std_acc_pct` summary is printed to stdout, not written as a
second file. Existing artifacts are skipped by default and included in the CSV
when readable; append `--force` only when you intentionally want to retrain and
overwrite them. Finetune does not create a log file by default; pass
`--log_file results/logs/<model>_<dataset>_<protocol>.log` to tee stdout/stderr
to a persistent log file.

Directory roles:

- `finetune/` — single-model finetune + artifact export; it provides the shared teacher/student artifacts.
- `distill/run_distill.py` — method-centric YAML distillation runner with automatic Base control.
- `fusion/run_fusion.py` — YAML-driven current artifact fusion runner.
- `bidir/` and `mask/` — historical negative-result reproduction drivers.

The old D0-onward fusion archive remains in the `pre-consolidation` tag; this tree only uses the current unified fusion implementation.

Reusable collaboration algorithms belong in `collab/`. Verification/tooling
belongs in `scripts/`.
