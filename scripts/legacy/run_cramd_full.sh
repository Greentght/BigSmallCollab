#!/usr/bin/env bash
set -e
cd /home/lixinli/BigSmallCollab
conda run -n mirepnet python scripts/legacy/bidir/run_cramd_loso.py   --dataset BNCI2014001-4 --warmup 15 --total 50   --lam_bs 0.5 --lam_sb 0.1 --gpu 2 --tag cramd
echo "===== $(date) DONE_CRAMD ====="
