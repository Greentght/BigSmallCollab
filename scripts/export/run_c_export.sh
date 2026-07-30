#!/usr/bin/env bash
# Phase-C export: add BNCI2015001 (12 subj) + AlexMI (8 subj) artifacts so the
# balance-gated selection result gains statistical power (more subjects raise the
# n=9 Wilcoxon floor). Small models + mirepnet in mirepnet env, cbramod_native in
# cbramod env. LOSO first (the protocol the F+T result uses), then within.
# All workers on GPU 2 (avoid GPU 0). Restartable via export_preds.py exists-guard.
set -u
cd "$(dirname "$0")/.."
GPU=2
LOGDIR=logs/c_export
mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

small_worker() {  # model
  local model=$1
  for proto in loso within; do
    for ds in BNCI2015001 AlexMI; do
      echo "=== $model $ds $proto ===" >> "$LOGDIR/${model}.log"
      conda run -n mirepnet python scripts/export/export_preds.py \
        --model "$model" --dataset "$ds" --protocol "$proto" --gpu "$GPU" \
        >> "$LOGDIR/${model}.log" 2>&1
    done
  done
  echo "WORKER DONE: $model" >> "$LOGDIR/${model}.log"
}

cbramod_worker() {
  for proto in loso within; do
    for ds in BNCI2015001 AlexMI; do
      echo "=== cbramod_native $ds $proto ===" >> "$LOGDIR/cbramod_native.log"
      conda run -n cbramod python scripts/export/export_preds.py \
        --model cbramod_native --dataset "$ds" --protocol "$proto" --gpu "$GPU" \
        >> "$LOGDIR/cbramod_native.log" 2>&1
    done
  done
  echo "WORKER DONE: cbramod_native" >> "$LOGDIR/cbramod_native.log"
}

export -f small_worker cbramod_worker
export GPU LOGDIR

for m in mirepnet ifnet eegnet adfcnn; do
  setsid nohup bash -c "small_worker $m" >/dev/null 2>&1 &
  echo "launched small_worker $m pid=$!"
done
setsid nohup bash -c "cbramod_worker" >/dev/null 2>&1 &
echo "launched cbramod_worker pid=$!"
echo "5 workers launched; logs in $LOGDIR/"
