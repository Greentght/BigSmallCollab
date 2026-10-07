#!/usr/bin/env bash
set -euo pipefail

# Batch export train/test artifacts with sample_uid metadata.
#
# Default mode is audit-only: commands are printed but not executed.
# Use --execute after reviewing the command list.
#
# Examples:
#   bash experiments/finetune/run_all_artifacts.sh --dry-run
#   bash experiments/finetune/run_all_artifacts.sh --execute --gpu 7
#   bash experiments/finetune/run_all_artifacts.sh --execute --protocol loso --gpu 7
#   bash experiments/finetune/run_all_artifacts.sh --dry-run --datasets "BNCI2014004 AlexMI" --seeds "666"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASETS=(BNCI2014001 BNCI2014001-4 BNCI2014004 AlexMI BNCI2015001)
MODELS=(mirepnet cbramod eegnet ifnet adfcnn)
PROTOCOL="fewshot"
GPU="${GPU:-}"
SEEDS=()
KEYS=()
TRAIN_PERCENTAGE=""
VAL_SPLIT=""
FORCE=1
EXPORT_TRAIN=1
EXECUTE=0
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/artifact_export_uid/${RUN_TS}}"
RESULTS_DIR="${RESULTS_DIR:-/data1/llx/BigSmallCollab_results}"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/finetune/run_all_artifacts.sh [options]

Default behavior:
  Prints the command plan only. Nothing is trained until --execute is passed.

Options:
  --execute                  Run the jobs.
  --dry-run                  Print commands only. This is the default.
  --protocol NAME            fewshot or loso. Default: fewshot.
  --gpu ID                   Pass --gpu ID to finetune.py. Omitted by default.
  --datasets "A B ..."       Override dataset list.
  --models "m1 m2 ..."       Override model list.
  --seeds "666 667 ..."      Pass selected seeds to finetune.py.
  --keys "0 1 ..."           Pass selected subjects/folds to finetune.py.
  --train-percentage VALUE   fewshot train fraction, passed as --train_percentage.
  --val-split VALUE          test fraction, passed as --val_split.
  --no-force                 Do not overwrite existing artifacts.
  --no-export-train          Only export test artifacts.
  --log-dir PATH             Directory for per-job logs.
  --results-dir PATH         Directory for per-job summary CSVs.
  -h, --help                 Show this help.

Default datasets:
  BNCI2014001 BNCI2014001-4 BNCI2014004 AlexMI BNCI2015001

Default models:
  mirepnet cbramod eegnet ifnet adfcnn
EOF
}

env_for_model() {
  case "$1" in
    cbramod) echo "cbramod" ;;
    mirepnet|eegnet|ifnet|adfcnn) echo "mirepnet" ;;
    *) echo "ERROR: unknown model '$1'" >&2; return 1 ;;
  esac
}

split_words() {
  local value="$1"
  # shellcheck disable=SC2206
  SPLIT_WORDS_OUT=($value)
}

print_cmd() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --dry-run) EXECUTE=0; shift ;;
    --protocol) PROTOCOL="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --datasets)
      split_words "$2"
      DATASETS=("${SPLIT_WORDS_OUT[@]}")
      shift 2
      ;;
    --models)
      split_words "$2"
      MODELS=("${SPLIT_WORDS_OUT[@]}")
      shift 2
      ;;
    --seeds)
      split_words "$2"
      SEEDS=("${SPLIT_WORDS_OUT[@]}")
      shift 2
      ;;
    --keys)
      split_words "$2"
      KEYS=("${SPLIT_WORDS_OUT[@]}")
      shift 2
      ;;
    --train-percentage) TRAIN_PERCENTAGE="$2"; shift 2 ;;
    --val-split) VAL_SPLIT="$2"; shift 2 ;;
    --no-force) FORCE=0; shift ;;
    --no-export-train) EXPORT_TRAIN=0; shift ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$PROTOCOL" != "fewshot" && "$PROTOCOL" != "loso" ]]; then
  echo "ERROR: --protocol must be fewshot or loso, got '$PROTOCOL'" >&2
  exit 2
fi

if [[ -n "$TRAIN_PERCENTAGE" && -n "$VAL_SPLIT" ]]; then
  echo "ERROR: pass only one of --train-percentage or --val-split" >&2
  exit 2
fi

JOB_COUNT=$((${#DATASETS[@]} * ${#MODELS[@]}))
MODE_LABEL="DRY-RUN"
if [[ "$EXECUTE" -eq 1 ]]; then
  MODE_LABEL="EXECUTE"
  mkdir -p "$LOG_DIR" "$RESULTS_DIR"
fi

echo "[$MODE_LABEL] artifact export jobs=${JOB_COUNT} protocol=${PROTOCOL} force=${FORCE} export_train=${EXPORT_TRAIN}"
echo "[datasets] ${DATASETS[*]}"
echo "[models] ${MODELS[*]}"
[[ -n "$GPU" ]] && echo "[gpu] ${GPU}"
[[ ${#SEEDS[@]} -gt 0 ]] && echo "[seeds] ${SEEDS[*]}"
[[ ${#KEYS[@]} -gt 0 ]] && echo "[keys] ${KEYS[*]}"
echo "[logs] ${LOG_DIR}"
echo "[results] ${RESULTS_DIR}"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    env_name="$(env_for_model "$model")"
    log_file="${LOG_DIR}/${dataset}_${PROTOCOL}_${model}.log"
    out_csv="${RESULTS_DIR}/${dataset}_${PROTOCOL}_${model}.csv"

    cmd=(conda run --no-capture-output -n "$env_name"
         python experiments/finetune/finetune.py
         --model "$model"
         --dataset "$dataset"
         --protocol "$PROTOCOL"
         --log_file "$log_file"
         --out_csv "$out_csv")

    [[ -n "$GPU" ]] && cmd+=(--gpu "$GPU")
    [[ "$FORCE" -eq 1 ]] && cmd+=(--force)
    [[ "$EXPORT_TRAIN" -eq 0 ]] && cmd+=(--no_export_train)
    [[ ${#SEEDS[@]} -gt 0 ]] && cmd+=(--seeds "${SEEDS[@]}")
    [[ ${#KEYS[@]} -gt 0 ]] && cmd+=(--keys "${KEYS[@]}")
    [[ -n "$TRAIN_PERCENTAGE" ]] && cmd+=(--train_percentage "$TRAIN_PERCENTAGE")
    [[ -n "$VAL_SPLIT" ]] && cmd+=(--val_split "$VAL_SPLIT")

    echo
    echo "[job] dataset=${dataset} model=${model} env=${env_name}"
    print_cmd "${cmd[@]}"
    if [[ "$EXECUTE" -eq 1 ]]; then
      "${cmd[@]}"
    fi
  done
done

echo
echo "Done."
