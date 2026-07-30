#!/usr/bin/env bash
# Phase-D0 artifact export queue. Fills the MISSING per-sample artifacts needed
# for the headroom map: small-model within+LOSO (mirepnet env) and cbramod_native
# LOSO (cbramod env). Big-model caches (mirepnet within+loso, cbramod within)
# already exist and are skipped by export_preds.py's `exists` guard.
#
# Concurrency: 4 background workers (3 small models + 1 cbramod), all pinned to
# GPU 2 (user: avoid GPU 0). Each worker runs its jobs SEQUENTIALLY over the two
# datasets / two protocols, so only 4 processes share the GPU + CPU at once.
# CPU threads capped (shared box). Restartable — re-run to refill gaps after a crash.
set -u
cd "$(dirname "$0")/.."
GPU=2
LOGDIR=logs/d0_export
mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

# one background worker = one model, sequential over (dataset x protocol)
small_worker() {  # model
  local model=$1
  for proto in within loso; do
    for ds in BNCI2014001-4 BNCI2014004; do
      echo "=== $model $ds $proto ===" >> "$LOGDIR/${model}.log"
      conda run -n mirepnet python scripts/export/export_preds.py \
        --model "$model" --dataset "$ds" --protocol "$proto" --gpu "$GPU" \
        >> "$LOGDIR/${model}.log" 2>&1
    done
  done
  echo "WORKER DONE: $model" >> "$LOGDIR/${model}.log"
}

cbramod_worker() {
  for ds in BNCI2014001-4 BNCI2014004; do
    echo "=== cbramod_native $ds loso ===" >> "$LOGDIR/cbramod_native_loso.log"
    conda run -n cbramod python scripts/export/export_preds.py \
      --model cbramod_native --dataset "$ds" --protocol loso --gpu "$GPU" \
      >> "$LOGDIR/cbramod_native_loso.log" 2>&1
  done
  echo "WORKER DONE: cbramod_native_loso" >> "$LOGDIR/cbramod_native_loso.log"
}

export -f small_worker cbramod_worker
export GPU LOGDIR

for m in ifnet eegnet adfcnn; do
  setsid nohup bash -c "small_worker $m" >/dev/null 2>&1 &
  echo "launched small_worker $m pid=$!"
done
setsid nohup bash -c "cbramod_worker" >/dev/null 2>&1 &
echo "launched cbramod_worker pid=$!"
echo "4 workers launched; logs in $LOGDIR/"
