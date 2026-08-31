#!/usr/bin/env bash
# Launch the FINAL settled CBraMod (45ch channel-template pipeline, PROGRESS
# 2026-07-01 全量终表) on the paper-5 tasks: 80/20 single-session protocol,
# EA + 45ch pad + 250->200Hz / scale=1, official all_patch_reps head,
# equal-lr AdamW 1e-4 wd 5e-2 + cosine + label_smoothing 0.1, bs 64, 50 epochs.
#
# Expected (3seed, paper80): 14001-2 77.78 / 14001-4 62.07 / 004 74.38 /
# AlexMI 66.15 / 15001 71.11.
#
# Usage:
#   bash scripts/legacy/run_cbramod_paper5.sh paper80 "3 5 6 8 2"
set -euo pipefail

PRESET="${1:-paper80}"
GPU_STR="${2:-3 5 6 8 2}"
read -r -a GPUS <<< "$GPU_STR"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs results/cbramod

DATASETS=(
  BNCI2014001_4c
  BNCI2014001_2c
  BNCI2014004
  AlexMI_2c
  BNCI2015001
)

for i in "${!DATASETS[@]}"; do
  DS="${DATASETS[$i]}"
  GPU="${GPUS[$((i % ${#GPUS[@]}))]}"
  LOG="logs/cbramod_${DS}_${PRESET}.log"
  echo "[$(date '+%F %T')] launch ${DS} preset=${PRESET} gpu=${GPU} -> ${LOG}"
  setsid conda run -n cbramod python experiments/bigmodel/cbramod_adapt.py \
      --dataset "${DS}" \
      --preset "${PRESET}" \
      --pipeline template \
      --gpu "${GPU}" \
      --epochs 50 \
      --batch_size 64 \
      --lr 0.0001 \
      --weight_decay 0.05 \
      --dropout 0.1 \
      --label_smoothing 0.1 \
      --warmup_epochs 0 \
      --min_lr 0 \
      --clip_grad_norm 0 \
      --scale_divisor 1 \
      > "${LOG}" 2>&1 < /dev/null &
done

echo "Launched ${#DATASETS[@]} CBraMod template-pipeline jobs. Check logs/cbramod_*_${PRESET}.log"
