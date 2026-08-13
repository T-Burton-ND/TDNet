#!/usr/bin/env bash
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/corrected_f6_wide_margin_hps}"
RUN_ROOT="${OUTPUT_ROOT}/experiments/publication_hps_corrected_f6_wide_margin_gap_v1"
JOB_MANIFEST="${JOB_MANIFEST:-${RUN_ROOT}/job_manifest.parquet}"
CHUNK_MANIFEST="${CHUNK_MANIFEST:-${RUN_ROOT}/chunk_manifest.parquet}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/job_logs}"
CONDA_ENV="${CONDA_ENV:-gridiron}"
TC="${TC:-50}"
HVMEM="${HVMEM:-16G}"

[[ -f "${JOB_MANIFEST}" ]] || { echo "Missing ${JOB_MANIFEST}" >&2; exit 2; }
[[ -f "${CHUNK_MANIFEST}" ]] || { echo "Missing ${CHUNK_MANIFEST}" >&2; exit 2; }
[[ -f "${RUN_ROOT}/smoke_test.ok" ]] || {
  echo "Refusing submission without ${RUN_ROOT}/smoke_test.ok" >&2
  exit 2
}
mkdir -p "${LOG_ROOT}"
TASK_COUNT="$(conda run -n "${CONDA_ENV}" python -c 'import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))' "${CHUNK_MANIFEST}")"

array_submission="$(qsub -terse -clear -cwd -V -j y -q long \
  -N tdnet_f6wide_hps -t "1-${TASK_COUNT}" -tc "${TC}" \
  -pe smp 1 -l h_rt=48:00:00 -l h_vmem="${HVMEM}" \
  -o "${LOG_ROOT}/" \
  -v "REPO_ROOT=${ROOT},JOB_MANIFEST=${JOB_MANIFEST},CONDA_ENV=${CONDA_ENV}" \
  "${ROOT}/scripts/sge/experiment_chunk_task.sge")"
array_job_id="$(printf '%s' "${array_submission}" | grep -Eo '[0-9]+' | head -1)"
[[ -n "${array_job_id}" ]] || { echo "Could not parse array job ID: ${array_submission}" >&2; exit 2; }

finalize_submission="$(qsub -terse -clear -cwd -V -j y -q long \
  -N tdnet_f6wide_finalize -hold_jid "${array_job_id}" \
  -pe smp 1 -l h_rt=08:00:00 -l h_vmem=16G \
  -o "${LOG_ROOT}/" \
  -v "REPO_ROOT=${ROOT},JOB_MANIFEST=${JOB_MANIFEST},OUTPUT_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV}" \
  "${ROOT}/scripts/sge/finalize_new_fingerprint_hps.sge")"

echo "array=${array_submission}"
echo "finalize=${finalize_submission}"
echo "tasks=${TASK_COUNT}"
echo "tc=${TC}"
echo "h_vmem=${HVMEM}"
