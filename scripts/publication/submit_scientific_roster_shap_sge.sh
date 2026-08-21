#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHAP_CONFIG="${SHAP_CONFIG:-${REPO_ROOT}/configs/publication/scientific_roster_shap_study.yaml}"
SHAP_OUTPUT_ROOT="${SHAP_OUTPUT_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_shap/corrected_f6_final_v1}"
CONDA_ENV="${CONDA_ENV:-gridiron}"
TC="${TC:-25}"
HVMEM="${HVMEM:-16G}"
SLOTS="${SLOTS:-1}"
HRT="${HRT:-24:00:00}"
SHAP_MODEL_N_JOBS="${SHAP_MODEL_N_JOBS:-1}"
SHAP_PREDICTION_BATCH_SIZE="${SHAP_PREDICTION_BATCH_SIZE:-512}"
mkdir -p "${SHAP_OUTPUT_ROOT}/job_logs"

task_count="$(conda run -n "${CONDA_ENV}" python -c 'import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))' "${SHAP_OUTPUT_ROOT}/job_manifest.parquet")"
TASK_SPEC="${TASK_SPEC:-1-${task_count}}"
FINAL_HOLD_JID="${FINAL_HOLD_JID:-}"
array_submission="$(qsub -terse -clear -cwd -V -j y -q long \
  -N tdnet_f6_shap -t "${TASK_SPEC}" -tc "${TC}" -pe smp "${SLOTS}" \
  -l h_rt="${HRT}" -l h_vmem="${HVMEM}" -o "${SHAP_OUTPUT_ROOT}/job_logs/" \
  -v "REPO_ROOT=${REPO_ROOT},SHAP_CONFIG=${SHAP_CONFIG},SHAP_OUTPUT_ROOT=${SHAP_OUTPUT_ROOT},CONDA_ENV=${CONDA_ENV},FORCE_RERUN=${FORCE_RERUN:-0},SHAP_MODEL_N_JOBS=${SHAP_MODEL_N_JOBS},SHAP_PREDICTION_BATCH_SIZE=${SHAP_PREDICTION_BATCH_SIZE}" \
  "${REPO_ROOT}/scripts/sge/scientific_roster_shap_task.sge")"
array_job_id="$(printf '%s' "${array_submission}" | grep -Eo '[0-9]+' | head -1)"
[[ -n "${array_job_id}" ]] || { echo "Could not parse array job ID: ${array_submission}" >&2; exit 2; }

hold_jobs="${array_job_id}${FINAL_HOLD_JID:+,${FINAL_HOLD_JID}}"
final_submission="$(qsub -terse -clear -cwd -V -j y -q long \
  -N tdnet_f6_shap_finalize -hold_jid "${hold_jobs}" -pe smp 1 \
  -l h_rt=02:00:00 -l h_vmem=8G -o "${SHAP_OUTPUT_ROOT}/job_logs/" \
  -v "REPO_ROOT=${REPO_ROOT},SHAP_CONFIG=${SHAP_CONFIG},SHAP_OUTPUT_ROOT=${SHAP_OUTPUT_ROOT},CONDA_ENV=${CONDA_ENV}" \
  "${REPO_ROOT}/scripts/sge/scientific_roster_shap_finalize.sge")"

echo "array=${array_submission}"
echo "finalize=${final_submission}"
echo "tasks=${task_count}"
echo "task_spec=${TASK_SPEC}"
echo "final_hold_jid=${FINAL_HOLD_JID}"
echo "slots=${SLOTS} model_n_jobs=${SHAP_MODEL_N_JOBS} walltime=${HRT}"
echo "shap_prediction_batch_size=${SHAP_PREDICTION_BATCH_SIZE}"
