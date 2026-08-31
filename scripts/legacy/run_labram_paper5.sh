#!/usr/bin/env bash
# LaBraM native downstream adaptation over the paper-5 MI tasks (native70 preset,
# validation-selected best epoch — faithful to LaBraM's downstream recipe).
# Usage: bash scripts/legacy/run_labram_paper5.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
PRESET=native70
DATASETS=(BNCI2014001_4c BNCI2014001_2c BNCI2014004 AlexMI_2c BNCI2015001)

source ~/anaconda3/etc/profile.d/conda.sh
conda activate labram

mkdir -p logs results/labram
for ds in "${DATASETS[@]}"; do
  echo "==== $(date '+%F %T') launching $ds ===="
  python experiments/bigmodel/labram_adapt.py \
    --dataset "$ds" --preset "$PRESET" --gpu "$GPU" \
    --epochs 50 --batch_size 32 --seeds 666 667 668 \
    2>&1 | tee "logs/labram_${ds}_${PRESET}.log"
done

echo "==== $(date '+%F %T') all datasets done; aggregating ===="
python experiments/bigmodel/aggregate_labram.py --preset "$PRESET"
