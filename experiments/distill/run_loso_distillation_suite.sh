#!/usr/bin/env bash
# Run stages in the user's requested order; stop if any cell/group fails.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
gpu="${GPU:-0}"
conda_bin="/home/lixinli/anaconda3/bin/conda"
result_root="${repo_root}/results/distill/loso_five_settings_kd_feature_warmup10_v1"
log_dir="${result_root}/execution_logs"
mkdir -p "${log_dir}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4

# Never start a second coordinator for the same protocol.
exec 9>"${result_root}/suite.lock"
flock -n 9 || { echo 'Another LOSO distillation coordinator is already running.'; exit 1; }
printf '%s\n' "$$" > "${result_root}/suite.pid"
trap 'suite_exit_code=$?; printf "[suite-exit] timestamp=%s code=%s\n" "$(date --iso-8601=seconds)" "${suite_exit_code}"' EXIT

stages=(logits_kd kd_feature warmup10_kd warmup10_kd_feature)
datasets=(BNCI2014001 BNCI2014001-4 BNCI2014004 BNCI2015001 AlexMI)
teachers=(mirepnet cbramod)

printf '[suite-start] timestamp=%s gpu=%s stages=%s\n' "$(date --iso-8601=seconds)" "${gpu}" "${stages[*]}"
for stage in "${stages[@]}"; do
  printf '[stage-start] timestamp=%s stage=%s expected_cells=846\n' "$(date --iso-8601=seconds)" "${stage}"
  for dataset in "${datasets[@]}"; do
    for teacher in "${teachers[@]}"; do
      if [[ "${stage}" == logits_kd ]]; then
        printf '[cache-start] timestamp=%s teacher=%s dataset=%s\n' "$(date --iso-8601=seconds)" "${teacher}" "${dataset}"
        "${conda_bin}" run --no-capture-output -n "${teacher}" python -u \
          experiments/distill/loso_teacher_cache.py \
          --teacher "${teacher}" --datasets "${dataset}" --seeds 666 667 668 --gpu "${gpu}" \
          >> "${log_dir}/teacher_cache_${teacher}_${dataset}.log" 2>&1
        printf '[cache-done] timestamp=%s teacher=%s dataset=%s\n' "$(date --iso-8601=seconds)" "${teacher}" "${dataset}"
      fi
      printf '[group-start] timestamp=%s stage=%s teacher=%s dataset=%s\n' "$(date --iso-8601=seconds)" "${stage}" "${teacher}" "${dataset}"
      "${conda_bin}" run --no-capture-output -n mirepnet python -u \
        experiments/distill/run_loso_distillation.py \
        --stage "${stage}" --teacher "${teacher}" --datasets "${dataset}" \
        --students ifnet eegnet adfcnn --seeds 666 667 668 --gpu "${gpu}" \
        >> "${log_dir}/${stage}_${teacher}_${dataset}.log" 2>&1
      printf '[group-done] timestamp=%s stage=%s teacher=%s dataset=%s\n' "$(date --iso-8601=seconds)" "${stage}" "${teacher}" "${dataset}"
    done
  done
  "${conda_bin}" run --no-capture-output -n mirepnet python -u \
    experiments/distill/run_loso_distillation.py --assert-stage-complete "${stage}" --summarize \
    >> "${log_dir}/stage_barriers.log" 2>&1
  printf '[stage-done] timestamp=%s stage=%s\n' "$(date --iso-8601=seconds)" "${stage}"
done
printf '[suite-done] timestamp=%s\n' "$(date --iso-8601=seconds)"
