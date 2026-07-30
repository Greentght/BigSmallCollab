#!/usr/bin/env bash
# Exp-5b: add IntraOnly / InterOnly to answer "intra vs inter drives the gain?".
# Re-runs full 7-condition set on both datasets (overwrites CSVs).
set -e
cd /home/lixinli/BigSmallCollab
GPU=8
for DS in BNCI2014001-4 BNCI2014004; do
  echo "===== $(date) $DS ====="
  conda run -n mirepnet python scripts/distill/run_distill.py \
    --dataset $DS --teacher mirepnet --student ifnet --relational_ablation \
    --lam_proto 0.5 --lam_sim 0.5 --lam_intra 0.5 --lam_inter 0.5 --gpu $GPU \
    --out_csv results/metrics/${DS}_relational_mirepnet_to_ifnet.csv
done
echo "===== $(date) DONE ====="
