#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-gridiron}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/data/experiments/opponent_adjusted_ablation_shap}"
SOURCE_FINGERPRINT_ROOT="${SOURCE_FINGERPRINT_ROOT:-$REPO_ROOT/data/experiments/opponent_adjusted_fingerprints}"
export REPO_ROOT CONDA_ENV OUTPUT_ROOT SOURCE_FINGERPRINT_ROOT

cd "$REPO_ROOT"
mkdir -p docs/logs/sge

if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
elif [[ -n "${CONDA_ENV:-}" ]]; then
  echo "Conda environment '$CONDA_ENV' not found; continuing with current environment." >&2
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python src/gridiron_ml/cli/run_opponent_ablation_shap.py build-manifest \
  --project-root "$REPO_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --source-fingerprint-root "$SOURCE_FINGERPRINT_ROOT" \
  "$@"

JOB_COUNT="$(python - <<'PY'
from pathlib import Path
import os
import pandas as pd
path = Path(os.environ["OUTPUT_ROOT"]) / "job_manifest.csv"
print(len(pd.read_csv(path)))
PY
)"

if [[ "$JOB_COUNT" -lt 1 ]]; then
  echo "No jobs found in manifest." >&2
  exit 1
fi

echo "Submitting ${JOB_COUNT} SGE tasks from ${OUTPUT_ROOT}/job_manifest.csv"
ARRAY_SUBMISSION="$(qsub -V -t "1-${JOB_COUNT}" scripts/sge/opponent_ablation_shap_task.sge)"
echo "$ARRAY_SUBMISSION"

ARRAY_JOB_ID="$(printf '%s\n' "$ARRAY_SUBMISSION" | sed -n 's/.*job-array \([0-9][0-9]*\).*/\1/p; s/.*job \([0-9][0-9]*\).*/\1/p' | head -1)"
if [[ -z "$ARRAY_JOB_ID" ]]; then
  echo "Could not parse SGE array job id from qsub output; submit merge manually with scripts/sge/opponent_ablation_merge.sge." >&2
  exit 1
fi

echo "Submitting held merge/analysis job after ${ARRAY_JOB_ID}"
qsub -V -hold_jid "$ARRAY_JOB_ID" scripts/sge/opponent_ablation_merge.sge
