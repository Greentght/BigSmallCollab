#!/usr/bin/env bash
# Phase-1 fair check: VanillaKD / ProtoOnly / KDProto under the SAME deterministic
# config as relstats (balanced sampling, seeded). Reuse Base/IntraInter from relstats.
set -e
cd /home/lixinli/BigSmallCollab
GPU=8
for DS in BNCI2014001-4 BNCI2014004; do
  echo "===== $(date) $DS ====="
  conda run -n mirepnet python scripts/run_distill.py \
    --dataset $DS --teacher mirepnet --student ifnet --relational_ablation \
    --rel_conds VanillaKD,ProtoOnly,KDProto \
    --lam_kd 0.5 --lam_proto 0.5 --gpu $GPU \
    --out_csv results/metrics/${DS}_protokd_mirepnet_to_ifnet.csv
done
echo "===== $(date) DONE ====="
