#!/usr/bin/env bash
# Launch CBraMod native-preprocessing downstream adaptation on the paper-5 tasks.
#
# Defaults:
#   preset=native70: progress step 2 style, within-subject 70/30, no session filter.
#   hyperparams: adapted from the CBraMod tuning records: full finetune, AdamW,
#                20 epochs, batch 16, lr 1e-3, wd 0.1, native all_patch_reps head.
#
# Usage:
#   bash scripts/run_cbramod_native_paper5.sh native70 "3 5 6 8 2"
#   bash scripts/run_cbramod_native_paper5.sh paper80  "3 5 6 8 2"
set -euo pipefail

PRESET="${1:-native70}"
GPU_STR="${2:-3 5 6 8 2}"
read -r -a GPUS <<< "$GPU_STR"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs results/cbramod_native

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
  LOG="logs/cbramod_native_${DS}_${PRESET}.log"
  echo "[$(date '+%F %T')] launch ${DS} preset=${PRESET} gpu=${GPU} -> ${LOG}"
  setsid conda run -n cbramod python scripts/bigmodel/cbramod_native_adapt.py \
      --dataset "${DS}" \
      --preset "${PRESET}" \
      --gpu "${GPU}" \
      --epochs 20 \
      --batch_size 16 \
      --lr 0.001 \
      --weight_decay 0.1 \
      --dropout 0.5 \
      --label_smoothing 0.0 \
      --warmup_epochs 5 \
      --min_lr 1e-6 \
      --clip_grad_norm 1.0 \
      > "${LOG}" 2>&1 < /dev/null &
done

echo "Launched ${#DATASETS[@]} CBraMod native jobs. Check logs/cbramod_native_*_${PRESET}.log"
