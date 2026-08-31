#!/usr/bin/env bash
# Exp-3: Decoupled-KD (TCKD/NCKD) with target-masking on teacher-wrong samples.
# Meaningful only for >=3 classes (NCKD=0 for 2-class) -> BNCI2014001-4 (4cls).
# Conditions: base / KD_all / KD_masked / DKD_all / DKD_tmask.
set -e
cd /home/lixinli/BigSmallCollab
GPU=4
DS=BNCI2014001-4
for TEA in mirepnet cbramod_native; do
  for STU in ifnet eegnet; do
    echo "===== $(date) $TEA -> $STU ($DS) ====="
    conda run -n mirepnet python experiments/distill/run_distill.py \
      --dataset $DS --teacher $TEA --student $STU \
      --lam_kd 0.5 --dkd_ablation --gpu $GPU \
      --out_csv results/metrics/${DS}_dkd_${TEA/_/}_to_${STU}.csv
  done
done
echo "===== $(date) DONE ====="
