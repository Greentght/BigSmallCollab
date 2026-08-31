#!/usr/bin/env bash
# Exp-B: teacher-correct-only masking with the WEAK tuned teacher (CBraMod native
# CAR-only) -> small students. Teacher export runs in cbramod env; distillation
# reads cached artifacts and trains students in mirepnet env.
set -e
cd /home/lixinli/BigSmallCollab
GPU=5
DATASETS="BNCI2014004 BNCI2014001-4"

echo "===== $(date) export CBraMod-native (CAR-only) teacher ====="
for DS in $DATASETS; do
  conda run -n cbramod python scripts/export/finetune_export.py \
    --model cbramod --dataset $DS --gpu $GPU
done

echo "===== $(date) distill cbramod -> {ifnet, eegnet} ====="
for STU in ifnet eegnet; do
  for DS in $DATASETS; do
    conda run -n mirepnet python experiments/distill/run_distill.py \
      --dataset $DS --teacher cbramod --student $STU \
      --lam_kd 0.5 --lam_feat 0.5 --mask_ablation --gpu $GPU \
      --out_csv results/metrics/${DS}_maskablation_cbramodnative_to_${STU}.csv
  done
done
echo "===== $(date) DONE ====="
