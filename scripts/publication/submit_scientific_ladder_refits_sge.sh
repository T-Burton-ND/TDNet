#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELECTION="${SELECTION:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_heatmaps/corrected_full_ladder_v1/holdout_selection.csv}"
FINGERPRINT_DATA="${FINGERPRINT_DATA:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/fingerprint_ladder_v3/canonical_fingerprint.parquet}"
TRAINING_END_SEASON="${TRAINING_END_SEASON:-2025}"
HOLDOUT_SEASON="${HOLDOUT_SEASON:-}"
ROSTER_OUTPUT_ROOT="${ROSTER_OUTPUT_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_refits/f0_f8_margin_through_${TRAINING_END_SEASON}_v1}"
CONDA_ENV="${CONDA_ENV:-gridiron}"
TC="${TC:-24}"
HVMEM="${HVMEM:-16G}"
HOLD_ARGS=()
if [[ -n "${HPS_HOLD_JID:-}" ]]; then
  HOLD_ARGS=(-hold_jid "${HPS_HOLD_JID}")
fi
mkdir -p "${ROSTER_OUTPUT_ROOT}/job_logs"

array_submission="$(qsub -terse -clear -cwd -V -j y -q long \
  -N tdnet_scientific_refit -t 1-54 -tc "${TC}" "${HOLD_ARGS[@]}" \
  -pe smp 1 -l h_rt=24:00:00 -l h_vmem="${HVMEM}" \
  -o "${ROSTER_OUTPUT_ROOT}/job_logs/" \
  -v "REPO_ROOT=${REPO_ROOT},SELECTION=${SELECTION},FINGERPRINT_DATA=${FINGERPRINT_DATA},ROSTER_OUTPUT_ROOT=${ROSTER_OUTPUT_ROOT},TRAINING_END_SEASON=${TRAINING_END_SEASON},HOLDOUT_SEASON=${HOLDOUT_SEASON},CONDA_ENV=${CONDA_ENV}" \
  "${REPO_ROOT}/scripts/sge/refit_scientific_ladder_roster_task.sge")"
array_job_id="$(printf '%s' "${array_submission}" | grep -Eo '[0-9]+' | head -1)"
[[ -n "${array_job_id}" ]] || { echo "Could not parse array job ID: ${array_submission}" >&2; exit 2; }
final_submission="$(qsub -terse -clear -cwd -V -j y -q long \
  -N tdnet_scientific_refit_finalize -hold_jid "${array_job_id}" \
  -pe smp 1 -l h_rt=02:00:00 -l h_vmem=8G \
  -o "${ROSTER_OUTPUT_ROOT}/job_logs/" \
  -v "REPO_ROOT=${REPO_ROOT},SELECTION=${SELECTION},FINGERPRINT_DATA=${FINGERPRINT_DATA},ROSTER_OUTPUT_ROOT=${ROSTER_OUTPUT_ROOT},TRAINING_END_SEASON=${TRAINING_END_SEASON},HOLDOUT_SEASON=${HOLDOUT_SEASON},CONDA_ENV=${CONDA_ENV}" \
  "${REPO_ROOT}/scripts/sge/refit_scientific_ladder_roster_finalize.sge")"
echo "array=${array_submission}"
echo "finalize=${final_submission}"
