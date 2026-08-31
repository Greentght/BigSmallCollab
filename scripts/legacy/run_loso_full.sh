#!/usr/bin/env bash
# Phase-2 LOSO: per-fold teacher finetune+export, then student 4-condition distill.
set -e
cd /home/lixinli/BigSmallCollab
GPU=8
echo "===== $(date) LOSO teacher export ====="
for DS in BNCI2014004 BNCI2014001-4; do
  conda run -n mirepnet python scripts/export/export_teacher_loso.py --dataset $DS --gpu $GPU
done
echo "===== $(date) LOSO student distill ====="
for DS in BNCI2014004 BNCI2014001-4; do
  conda run -n mirepnet python experiments/distill/run_loso_distill.py \
    --dataset $DS --student ifnet --lam_kd 0.5 --lam_proto 0.5 --gpu $GPU \
    --out_csv results/metrics/${DS}_loso_mirepnet_to_ifnet.csv
done
echo "===== $(date) DONE ====="
