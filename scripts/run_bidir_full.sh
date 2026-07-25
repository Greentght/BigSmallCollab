#!/usr/bin/env bash
# Bidirectional routed distillation LOSO, MIRepNet<->IFNet. Uni vs Bidir.
set -e
cd /home/lixinli/BigSmallCollab
GPU=2
for DS in BNCI2014001-4 BNCI2014004; do
  echo "===== $(date) $DS ====="
  conda run -n mirepnet python scripts/run_bidir_loso.py     --dataset $DS --lam_bs 1.0 --lam_sb 0.1 --gpu $GPU     --out_csv results/metrics/${DS}_loso_bidir_mirepnet_ifnet.csv
done
echo "===== $(date) DONE_BIDIR ====="
