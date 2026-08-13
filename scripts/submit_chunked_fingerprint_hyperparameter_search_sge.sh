#!/usr/bin/env bash
set -euo pipefail

SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
  SUBMIT=true
  shift
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-gridiron}"
OBJECTIVE="${OBJECTIVE:?Set OBJECTIVE to winner, balanced, or margin}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/models/tuning/fingerprint_hyperparameter_search_${OBJECTIVE}.yaml}"
SOURCE_FINGERPRINT_ROOT="${SOURCE_FINGERPRINT_ROOT:-$REPO_ROOT/data/experiments/opponent_adjusted_fingerprints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/data/experiments/fingerprint_hyperparameter_search_chunked/${OBJECTIVE}}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
CHUNK_WORKERS="${CHUNK_WORKERS:-4}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-}"
MIN_FREE_GB="${MIN_FREE_GB:-29}"
export REPO_ROOT CONDA_ENV CONFIG_PATH SOURCE_FINGERPRINT_ROOT OUTPUT_ROOT CHUNK_SIZE CHUNK_WORKERS

cd "$REPO_ROOT"
mkdir -p docs/logs/sge

free_gb="$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if [[ "$free_gb" -lt "$MIN_FREE_GB" ]]; then
  echo "Refusing to prepare chunked HPS: only ${free_gb}G free; require ${MIN_FREE_GB}G." >&2
  exit 1
fi

if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
elif [[ -n "${CONDA_ENV:-}" ]]; then
  echo "Conda environment '$CONDA_ENV' not found; continuing with current environment." >&2
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tdnet-matplotlib-$USER}"

python src/gridiron_ml/cli/run_fingerprint_hyperparameter_search.py build-manifest \
  --project-root "$REPO_ROOT" \
  --config "$CONFIG_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --source-fingerprint-root "$SOURCE_FINGERPRINT_ROOT" \
  "$@"

JOB_COUNT="$(python - <<'PY'
from pathlib import Path
import math
import os
import pandas as pd
manifest = pd.read_csv(Path(os.environ["OUTPUT_ROOT"]) / "job_manifest.csv")
chunk_size = int(os.environ["CHUNK_SIZE"])
print(math.ceil(len(manifest) / chunk_size))
PY
)"

echo "Objective: ${OBJECTIVE}"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_ROOT}"
echo "Chunk size: ${CHUNK_SIZE}"
echo "Chunk workers: ${CHUNK_WORKERS}"
echo "SGE chunk tasks: ${JOB_COUNT}"
echo "SGE task concurrency: ${TASK_CONCURRENCY:-uncapped}"
echo "Free space: ${free_gb}G"

if [[ "$SUBMIT" != true ]]; then
  echo
  echo "Dry run only. To submit, run:"
  echo "  OBJECTIVE=${OBJECTIVE} CHUNK_SIZE=${CHUNK_SIZE} CHUNK_WORKERS=${CHUNK_WORKERS} TASK_CONCURRENCY=${TASK_CONCURRENCY:-25} bash scripts/submit_chunked_fingerprint_hyperparameter_search_sge.sh --submit"
  exit 0
fi

qsub_args=(-V -N "tdnet_hps_${OBJECTIVE}_chunk" -t "1-${JOB_COUNT}")
if [[ -n "$TASK_CONCURRENCY" ]]; then
  qsub_args+=(-tc "$TASK_CONCURRENCY")
fi
qsub_args+=(scripts/sge/fingerprint_hyperparameter_search_chunk_task.sge)

array_submission="$(qsub "${qsub_args[@]}")"
echo "$array_submission"
