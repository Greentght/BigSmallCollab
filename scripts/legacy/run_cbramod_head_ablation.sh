#!/usr/bin/env bash
# CBraMod head ablation: official 3-layer MLP (all_patch_reps) vs benchmark
# single Linear head. Each dataset uses its CAR-only tp0.3 tuned config
# (scale=1, norm=car), 30% training, seeds 666/667/668. GPUs round-robined.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cbramod

OUT=results/cbramod_native/head_ablation
mkdir -p "$OUT" logs
GPUS=(2 3 4 5 7); gi=0
COMMON="--preset native70 --train_percentage 0.3 --norm_method car --scale_divisor 1 \
        --batch_size 16 --seeds 666 667 668 --dataloader_workers 0"

run() {  # dataset lfreq hfreq notch lr epochs dropout wd
  local ds=$1 lf=$2 hf=$3 notch=$4 lr=$5 ep=$6 do=$7 wd=$8
  local nf=""; [ "$notch" != "none" ] && nf="--notch_freq $notch"
  for head in mlp linear; do
    local gpu=${GPUS[$((gi % ${#GPUS[@]}))]}; gi=$((gi+1))
    echo "[launch] $ds head=$head gpu=$gpu"
    python experiments/bigmodel/cbramod_native_adapt.py --dataset "$ds" $COMMON \
      --l_freq "$lf" --h_freq "$hf" $nf --lr "$lr" --epochs "$ep" \
      --dropout "$do" --weight_decay "$wd" --head "$head" --gpu "$gpu" \
      --out "$OUT/${ds}_${head}.csv" --overwrite \
      > "logs/cbramod_head_${ds}_${head}.log" 2>&1 &
  done
}

# dataset            lf   hf   notch lr     ep  do   wd
run AlexMI_2c        0.3  50   none  0.0005 20  0.1  0.01
run BNCI2014001_2c   0.3  50   none  0.001  50  0.1  0.1
run BNCI2014001_4c   0.3  50   none  0.001  20  0.1  0.1
run BNCI2014004      0.3  75   60    0.001  50  0.5  0.01
run BNCI2015001      0.3  75   60    0.001  50  0.1  0.1

wait
echo "==== all head-ablation runs done ===="