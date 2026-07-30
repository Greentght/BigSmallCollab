#!/usr/bin/env bash
# Relational (similarity-preserving) KD under LOSO cross-subject. Reuse cached
# mirepnet_loso teachers. Conditions SimFull/IntraInter/IntraOnly/InterOnly.
set -e
cd /home/lixinli/BigSmallCollab
GPU=2
for DS in BNCI2014004 BNCI2014001-4; do
  echo "===== $(date) $DS ====="
  conda run -n mirepnet python scripts/distill/run_loso_distill.py \
    --dataset $DS --student ifnet --conds SimFull,IntraInter,IntraOnly,InterOnly \
    --lam_sim 0.5 --lam_intra 0.5 --lam_inter 0.5 --gpu $GPU \
    --out_csv results/metrics/${DS}_loso_relational_mirepnet_to_ifnet.csv
done
echo "===== $(date) DONE_REL_LOSO ====="
