#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT/experiments/finetune/run_loso_five_datasets.py"
LOG_DIR="$ROOT/results/reproductions/loso_five_settings_canonical4s_v1/execution_logs"
mkdir -p "$LOG_DIR"

wait_for_current_seed666() {
  local pid="$1"
  local needle="$2"
  while [[ -r "/proc/$pid/cmdline" ]]; do
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$cmdline" == *"$needle"* ]] || break
    sleep 30
  done
}

run_main() {
  local env_name="$1" model="$2" gpu="$3" seed_group="$4"
  shift 4
  local log="$LOG_DIR/${model}_main_${seed_group}.log"
  echo "[suite-start] $model main seeds=$* gpu=$gpu log=$log"
  conda run --no-capture-output -n "$env_name" python -u "$RUNNER" \
    --model "$model" --recipe main --seeds "$@" --gpu "$gpu" > "$log" 2>&1
  echo "[suite-done] $model main seeds=$*"
}

run_cbramod_reference() {
  local seed="$1"
  local log="$LOG_DIR/cbramod_eegfm_full_seed${seed}.log"
  echo "[suite-start] cbramod eegfm_full seed=$seed gpu=3 log=$log"
  conda run --no-capture-output -n cbramod python -u "$RUNNER" \
    --model cbramod --recipe eegfm_full \
    --datasets BNCI2014001 BNCI2014001-4 BNCI2014004 BNCI2015001 \
    --seeds "$seed" --gpu 3 > "$log" 2>&1
  echo "[suite-done] cbramod eegfm_full seed=$seed"
}

# Seed 666 is already running in the interactive sessions. Wait for those
# processes, then rerun seed 666 in resume mode to fill any incomplete cells.
wait_for_current_seed666 34398 \
  'experiments/finetune/run_loso_five_datasets.py --model mirepnet --seeds 666 --gpu 2'
wait_for_current_seed666 43600 \
  'experiments/finetune/run_loso_five_datasets.py --model cbramod --seeds 666 --gpu 3'

run_main mirepnet mirepnet 2 seed666_resume 666
run_main cbramod cbramod 3 seed666_resume 666
run_main mirepnet mirepnet 2 seeds667_668 667 668
run_main cbramod cbramod 3 seeds667_668 667 668

for seed in 666 667 668; do
  run_cbramod_reference "$seed"
done

echo '[suite-complete] five-setting LOSO main and CBraMod reference recipe runs finished'
