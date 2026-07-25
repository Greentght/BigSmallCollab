#!/usr/bin/env bash
# Exp-5c evidence hardening: deterministic seeds, 4 core conditions x 3 seeds.
set -e
cd /home/lixinli/BigSmallCollab
GPU=8
for DS in BNCI2014001-4 BNCI2014004; do
  echo "===== $(date) $DS ====="
  conda run -n mirepnet python scripts/run_distill.py \
    --dataset $DS --teacher mirepnet --student ifnet --relational_ablation \
    --rel_conds base,IntraOnly,InterOnly,IntraInter \
    --lam_intra 0.5 --lam_inter 0.5 --gpu $GPU \
    --out_csv results/metrics/${DS}_relstats_mirepnet_to_ifnet.csv
done
echo "===== $(date) DONE ====="
