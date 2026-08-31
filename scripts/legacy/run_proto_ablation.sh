#!/usr/bin/env bash
# Exp-4: class-reliability-aware prototype distillation. Single direction
# MIRepNet -> IFNet. Conditions: base / KD / GlobalFeat / Proto / Proto_w.
# Both 2-class (004) and 4-class (001-4) to test class-count independence.
set -e
cd /home/lixinli/BigSmallCollab
GPU=7
for DS in BNCI2014004 BNCI2014001-4; do
  echo "===== $(date) $DS ====="
  conda run -n mirepnet python experiments/distill/run_distill.py \
    --dataset $DS --teacher mirepnet --student ifnet --proto_ablation \
    --lam_kd 0.5 --lam_proto 0.5 --gpu $GPU \
    --out_csv results/metrics/${DS}_proto_mirepnet_to_ifnet.csv
done
echo "===== $(date) DONE ====="
