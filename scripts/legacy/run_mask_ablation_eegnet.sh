#!/usr/bin/env bash
# Exp-1b: teacher-correct-only masking, weaker distill line MIRepNet -> EEGNet.
set -e
cd /home/lixinli/BigSmallCollab
GPU=7; ENV=mirepnet
for DS in BNCI2014004 BNCI2014001-4; do
  echo "===== $(date) $DS ====="
  conda run -n $ENV python scripts/distill/run_distill.py \
    --dataset $DS --teacher mirepnet --student eegnet \
    --lam_kd 0.5 --lam_feat 0.5 --mask_ablation --gpu $GPU \
    --out_csv results/metrics/${DS}_maskablation_mirepnet_to_eegnet.csv
done
echo "===== $(date) DONE ====="
