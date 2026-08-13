#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:?REPO_ROOT is required}"
cd "$ROOT"
if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tdnet-mpl-${JOB_ID:-local}-${SGE_TASK_ID:-0}"
mkdir -p "$MPLCONFIGDIR"
python -m gridiron_ml.cli.publication.run_scientific_cross_fitted_oof \
  --manifest "${MANIFEST:?MANIFEST is required}" \
  --frozen-inventory "${FROZEN_INVENTORY:?FROZEN_INVENTORY is required}" \
  --task-id "${SGE_TASK_ID:?SGE_TASK_ID is required}" \
  --output-root "${OOF_ROOT:?OOF_ROOT is required}" \
  --workers "${NSLOTS:-1}"
