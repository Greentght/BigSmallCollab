#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log_dir="${repo_root}/results/reproductions/loso_five_settings_canonical4s_v1/execution_logs"
conda_bin="/home/lixinli/anaconda3/bin/conda"
gpu="${GPU:-8}"
mkdir -p "${log_dir}"

for model in ifnet eegnet adfcnn; do
  log_path="${log_dir}/${model}_scratch_loso_seeds666_668.log"
  printf '[suite-start] %s gpu=%s seeds=666,667,668 log=%s\n' "${model}" "${gpu}" "${log_path}"
  "${conda_bin}" run --no-capture-output -n mirepnet python -u \
    "${repo_root}/experiments/finetune/run_loso_small_baselines.py" \
    --model "${model}" --seeds 666 667 668 --gpu "${gpu}" \
    > "${log_path}" 2>&1
  printf '[suite-done] %s\n' "${model}"
done
