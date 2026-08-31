#!/usr/bin/env bash
# Exp-1: teacher-correct-only masking on KD/Combo distillation.
# Teacher = MIRepNet (big), Student = IFNet (single-band small).
# Per dataset trains: base / KD_all / KD_masked / Combo_all / Combo_masked.
set -e
cd /home/lixinli/BigSmallCollab
GPU=7
ENV=mirepnet
DATASETS="BNCI2014004 BNCI2014001-4"

echo "===== $(date) export MIRepNet teacher artifacts ====="
for DS in $DATASETS; do
  conda run -n $ENV python scripts/export/finetune_export.py \
    --model mirepnet --dataset $DS --gpu $GPU
done

echo "===== $(date) run mask-ablation distillation ====="
for DS in $DATASETS; do
  conda run -n $ENV python experiments/distill/run_distill.py \
    --dataset $DS --teacher mirepnet --student ifnet \
    --lam_kd 0.5 --lam_feat 0.5 --mask_ablation --gpu $GPU \
    --out_csv results/metrics/${DS}_maskablation_mirepnet_to_ifnet.csv
done
echo "===== $(date) DONE ====="
