#!/usr/bin/env bash
# Exp-2: entropy-adaptive KD (MC-dropout uncertainty weight) vs base/all/masked.
# Step1: export teachers WITH MC-dropout uncertainty (each teacher's own env).
# Step2: adaptive distill reads cached logits+MC sidecar, trains students (mirepnet env).
set -e
cd /home/lixinli/BigSmallCollab
GPU=5
DATASETS="BNCI2014004 BNCI2014001-4"

echo "===== $(date) MC export: MIRepNet (mirepnet env) ====="
for DS in $DATASETS; do
  conda run -n mirepnet python scripts/export/export_teacher_mc.py --model mirepnet --dataset $DS --gpu $GPU
done
echo "===== $(date) MC export: CBraMod-native (cbramod env) ====="
for DS in $DATASETS; do
  conda run -n cbramod python scripts/export/export_teacher_mc.py --model cbramod_native --dataset $DS --gpu $GPU
done

echo "===== $(date) adaptive distill (mirepnet env) ====="
for TEA in mirepnet cbramod_native; do
  for STU in ifnet eegnet; do
    for DS in $DATASETS; do
      conda run -n mirepnet python scripts/distill/run_distill.py \
        --dataset $DS --teacher $TEA --student $STU \
        --lam_kd 0.5 --lam_feat 0.5 --adaptive --gpu $GPU \
        --out_csv results/metrics/${DS}_adaptivekd_${TEA/_/}_to_${STU}.csv
    done
  done
done
echo "===== $(date) DONE ====="
