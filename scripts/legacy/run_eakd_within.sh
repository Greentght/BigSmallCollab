#!/usr/bin/env bash
set -e
cd /home/lixinli/BigSmallCollab
GPU=2
for DS in BNCI2014004 BNCI2014001-4; do
  echo "===== $(date) $DS ====="
  conda run -n mirepnet python experiments/distill/run_distill.py     --dataset $DS --teacher mirepnet --student ifnet --relational_ablation     --rel_conds VanillaKD,EA_KD,CorrectMaskKD,CorrectMaskEA     --lam_kd 0.5 --gpu $GPU     --out_csv results/metrics/${DS}_eakd_within_mirepnet_to_ifnet.csv
done
echo "===== $(date) DONE ====="
