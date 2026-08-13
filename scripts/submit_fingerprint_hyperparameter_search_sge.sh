#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-gridiron}"
SOURCE_FINGERPRINT_ROOT="${SOURCE_FINGERPRINT_ROOT:-$REPO_ROOT/data/experiments/opponent_adjusted_fingerprints}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-$REPO_ROOT/data/experiments/fingerprint_hyperparameter_search}"
export REPO_ROOT CONDA_ENV SOURCE_FINGERPRINT_ROOT

cd "$REPO_ROOT"
mkdir -p docs/logs/sge

if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
elif [[ -n "${CONDA_ENV:-}" ]]; then
  echo "Conda environment '$CONDA_ENV' not found; continuing with current environment." >&2
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tdnet-matplotlib-$USER}"

variants=(winner balanced margin)
configs=(
  "$REPO_ROOT/configs/models/tuning/fingerprint_hyperparameter_search_winner.yaml"
  "$REPO_ROOT/configs/models/tuning/fingerprint_hyperparameter_search_balanced.yaml"
  "$REPO_ROOT/configs/models/tuning/fingerprint_hyperparameter_search_margin.yaml"
)

for idx in "${!variants[@]}"; do
  variant="${variants[$idx]}"
  config_path="${configs[$idx]}"
  output_root="$BASE_OUTPUT_ROOT/$variant"
  export CONFIG_PATH="$config_path"
  export OUTPUT_ROOT="$output_root"

  python src/gridiron_ml/cli/run_fingerprint_hyperparameter_search.py build-manifest \
    --project-root "$REPO_ROOT" \
    --config "$CONFIG_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --source-fingerprint-root "$SOURCE_FINGERPRINT_ROOT" \
    "$@"

  job_count="$(python - <<'PY'
from pathlib import Path
import os
import pandas as pd
path = Path(os.environ["OUTPUT_ROOT"]) / "job_manifest.csv"
print(len(pd.read_csv(path)))
PY
)"

  if [[ "$job_count" -lt 1 ]]; then
    echo "No jobs found in manifest for ${variant}." >&2
    exit 1
  fi

  echo "Submitting ${variant} fingerprint search: ${job_count} tasks"
  echo "  config: ${CONFIG_PATH}"
  echo "  output: ${OUTPUT_ROOT}"
  array_submission="$(qsub -V -N "tdnet_hps_${variant}" -t "1-${job_count}" scripts/sge/fingerprint_hyperparameter_search_task.sge)"
  echo "$array_submission"

  array_job_id="$(printf '%s\n' "$array_submission" | sed -n 's/.*job-array \([0-9][0-9]*\).*/\1/p; s/.*job \([0-9][0-9]*\).*/\1/p' | head -1)"
  if [[ -z "$array_job_id" ]]; then
    echo "Could not parse SGE array job id for ${variant}; submit merge manually with scripts/sge/fingerprint_hyperparameter_search_merge.sge." >&2
    exit 1
  fi

  echo "Submitting ${variant} merge job after ${array_job_id}"
  qsub -V -N "tdnet_hps_${variant}_merge" -hold_jid "$array_job_id" scripts/sge/fingerprint_hyperparameter_search_merge.sge
done
