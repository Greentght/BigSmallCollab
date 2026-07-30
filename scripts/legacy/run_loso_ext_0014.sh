#!/usr/bin/env bash
set -e
cd /home/lixinli/BigSmallCollab
GPU=7; DS=BNCI2014001-4
echo "===== $(date) $DS reliability ====="
conda run -n mirepnet python scripts/distill/run_loso_distill.py   --dataset $DS --student ifnet --conds CorrectMaskKD,ConfidenceKD   --lam_kd 0.5 --gpu $GPU   --out_csv results/metrics/${DS}_loso_reliability_mirepnet_to_ifnet.csv
echo "===== $(date) $DS pred-states ====="
conda run -n mirepnet python scripts/bidir/loso_pred_states.py --dataset $DS --gpu $GPU
echo "===== $(date) DONE0014EXT ====="
