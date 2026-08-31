#!/usr/bin/env bash
# Exp-4 extension: prototype distillation across teacher x student matrix.
# Adds cbramod_native->ifnet, mirepnet->eegnet, cbramod_native->eegnet
# (mirepnet->ifnet already done). Both 2cls(004) and 4cls(001-4).
set -e
cd /home/lixinli/BigSmallCollab
GPU=9
for COMBO in "cbramod_native ifnet" "mirepnet eegnet" "cbramod_native eegnet"; do
  set -- $COMBO; TEA=$1; STU=$2
  for DS in BNCI2014004 BNCI2014001-4; do
    echo "===== $(date) $TEA -> $STU ($DS) ====="
    conda run -n mirepnet python experiments/distill/run_distill.py \
      --dataset $DS --teacher $TEA --student $STU --proto_ablation \
      --lam_kd 0.5 --lam_proto 0.5 --gpu $GPU \
      --out_csv results/metrics/${DS}_proto_${TEA/_/}_to_${STU}.csv
  done
done
echo "===== $(date) DONE ====="
