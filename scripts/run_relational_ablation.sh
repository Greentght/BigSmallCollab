#!/usr/bin/env bash
# Exp-5: relational (similarity-preserving) KD. MIRepNet -> IFNet.
# base / SampleCos / SimFull / IntraInter / ProtoSim, class-balanced sampling.
set -e
cd /home/lixinli/BigSmallCollab
GPU=8
for DS in BNCI2014004 BNCI2014001-4; do
  echo "===== $(date) $DS ====="
  conda run -n mirepnet python scripts/run_distill.py \
    --dataset $DS --teacher mirepnet --student ifnet --relational_ablation \
    --lam_proto 0.5 --lam_sim 0.5 --lam_intra 0.5 --lam_inter 0.5 --gpu $GPU \
    --out_csv results/metrics/${DS}_relational_mirepnet_to_ifnet.csv
done
echo "===== $(date) DONE ====="
