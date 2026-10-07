#!/usr/bin/env bash
# Run each protocol stage in order, distributing independent teacher/dataset
# groups across the requested GPUs. A stage barrier prevents mixing stages.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
gpu_csv="${GPUS:-1,3}"
gpu_csv="${gpu_csv//[[:space:]]/}"
IFS=',' read -r -a gpus <<< "${gpu_csv}"
if ((${#gpus[@]} == 0)); then
  echo 'GPUS must contain one or more comma-separated GPU indices.' >&2
  exit 2
fi
for i in "${!gpus[@]}"; do
  [[ "${gpus[i]}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: ${gpus[i]}" >&2; exit 2; }
  for ((j = 0; j < i; j++)); do
    [[ "${gpus[i]}" != "${gpus[j]}" ]] || { echo "Duplicate GPU index: ${gpus[i]}" >&2; exit 2; }
  done
done

conda_bin="/home/lixinli/anaconda3/bin/conda"
result_root="/data1/llx/BigSmallcollab/results/distill/loso_five_settings_kd_feature_warmup10_v1"
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
group_datasets=()
group_teachers=()
for dataset in "${datasets[@]}"; do
  for teacher in "${teachers[@]}"; do
    group_datasets+=("${dataset}")
    group_teachers+=("${teacher}")
  done
done

run_group() {
  local stage="$1" gpu="$2" teacher="$3" dataset="$4"
  if [[ "${stage}" == logits_kd ]]; then
    printf '[cache-start] timestamp=%s teacher=%s dataset=%s gpu=%s\n' \
      "$(date --iso-8601=seconds)" "${teacher}" "${dataset}" "${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${conda_bin}" run --no-capture-output -n "${teacher}" python -u \
      experiments/distill/loso_teacher_cache.py \
      --teacher "${teacher}" --datasets "${dataset}" --seeds 666 667 668 --gpu 0 \
      >> "${log_dir}/teacher_cache_${teacher}_${dataset}.log" 2>&1
    printf '[cache-done] timestamp=%s teacher=%s dataset=%s gpu=%s\n' \
      "$(date --iso-8601=seconds)" "${teacher}" "${dataset}" "${gpu}"
  fi
  printf '[group-start] timestamp=%s stage=%s teacher=%s dataset=%s gpu=%s\n' \
    "$(date --iso-8601=seconds)" "${stage}" "${teacher}" "${dataset}" "${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${conda_bin}" run --no-capture-output -n mirepnet python -u \
    experiments/distill/run_loso_distillation.py \
    --stage "${stage}" --teacher "${teacher}" --datasets "${dataset}" \
    --students ifnet eegnet adfcnn --seeds 666 667 668 --gpu 0 \
    >> "${log_dir}/${stage}_${teacher}_${dataset}.log" 2>&1
  printf '[group-done] timestamp=%s stage=%s teacher=%s dataset=%s gpu=%s\n' \
    "$(date --iso-8601=seconds)" "${stage}" "${teacher}" "${dataset}" "${gpu}"
}

run_worker() {
  local stage="$1" worker_index="$2" worker_count="$3" gpu="$4"
  local group_index
  for ((group_index = worker_index; group_index < ${#group_datasets[@]}; group_index += worker_count)); do
    run_group "${stage}" "${gpu}" "${group_teachers[group_index]}" "${group_datasets[group_index]}"
  done
}

printf '[suite-start] timestamp=%s gpus=%s workers=%s stages=%s\n' \
  "$(date --iso-8601=seconds)" "${gpu_csv}" "${#gpus[@]}" "${stages[*]}"
for stage in "${stages[@]}"; do
  printf '[stage-start] timestamp=%s stage=%s expected_cells=846 workers=%s\n' \
    "$(date --iso-8601=seconds)" "${stage}" "${#gpus[@]}"
  worker_pids=()
  for worker_index in "${!gpus[@]}"; do
    run_worker "${stage}" "${worker_index}" "${#gpus[@]}" "${gpus[worker_index]}" &
    worker_pids+=("$!")
  done
  stage_failed=0
  for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
      stage_failed=1
    fi
  done
  if ((stage_failed)); then
    echo "At least one worker failed during stage ${stage}; stopping before the next stage." >&2
    exit 1
  fi
  "${conda_bin}" run --no-capture-output -n mirepnet python -u \
    experiments/distill/run_loso_distillation.py --assert-stage-complete "${stage}" --summarize \
    >> "${log_dir}/stage_barriers.log" 2>&1
  printf '[stage-done] timestamp=%s stage=%s\n' "$(date --iso-8601=seconds)" "${stage}"
done
printf '[suite-done] timestamp=%s\n' "$(date --iso-8601=seconds)"
