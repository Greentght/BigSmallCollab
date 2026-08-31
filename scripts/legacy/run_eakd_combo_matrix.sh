#!/usr/bin/env bash
# EA-KD on {pure KD, Combo} x {MIRepNet,CBraMod} x {IFNet,EEGNet} x {004,001-4}.
# Conditions per run: base/VanillaKD/EA_KD/Combo/EA_Combo. Within-subject, deterministic.
set -e
cd /home/lixinli/BigSmallCollab
GPU=3
for TEA in mirepnet cbramod; do
  for STU in ifnet eegnet; do
    for DS in BNCI2014004 BNCI2014001-4; do
      echo "===== $(date) $TEA -> $STU ($DS) ====="
      conda run -n mirepnet python experiments/distill/run_distill.py         --dataset $DS --teacher $TEA --student $STU --relational_ablation         --rel_conds base,VanillaKD,EA_KD,Combo,EA_Combo         --lam_kd 0.5 --lam_proto 0.5 --gpu $GPU         --out_csv results/metrics/${DS}_eacombo_${TEA/_/}_to_${STU}.csv
    done
  done
done
echo "===== $(date) DONE_EACOMBO ====="
