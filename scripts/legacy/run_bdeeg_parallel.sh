#!/usr/bin/env bash
# Launch BD-EEG LOSO full run parallelized: one fold per GPU (folds 0-8 -> GPU 0-8),
# each fold runs all 3 seeds x 6 groups. Per-fold CSVs are aggregated afterwards by
# scripts/analyze_bdeeg.py (which globs the fold tag). Detached via setsid so it
# survives terminal close.
cd /home/lixinli/BigSmallCollab || exit 1
DS=BNCI2014001-4
mkdir -p logs/bdeeg
# cap threads per process so 9 parallel jobs don't oversubscribe the shared box
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
for f in $(seq 0 8); do
  gpu=$f
  setsid nice -n 19 conda run -n mirepnet python scripts/run_bdeeg_loso.py \
    --dataset "$DS" --folds "$f" --seeds 666 667 668 \
    --warmup 20 --total 100 --lam_bs 1.0 --lam_sb 0.25 --gamma 1.0 --rho 0.5 \
    --gpu "$gpu" --tag "bdeeg_f${f}" \
    > "logs/bdeeg/f${f}.log" 2>&1 &
  echo "launched fold $f on gpu $gpu (pid $!)"
done
wait
echo "ALL_FOLDS_DONE"
