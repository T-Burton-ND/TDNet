#!/usr/bin/env bash
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GAP_SELECTION="${GAP_SELECTION:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/corrected_f6_wide_margin_hps/experiments/publication_hps_corrected_f6_wide_margin_gap_v1/summary/tables/best_configuration_by_cell.parquet}"
SCIENTIFIC_SELECTION="${SCIENTIFIC_SELECTION:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/new_fingerprint_hps/experiments/publication_hps_fingerprint_contract_corrected_scientific_v3/summary/tables/best_configuration_by_cell.parquet}"
FINGERPRINT_DATA="${FINGERPRINT_DATA:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/fingerprint_ladder_v3/canonical_fingerprint.parquet}"
SELECTION_ROOT="${SELECTION_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/corrected_f6_wide_margin_roster/selection_v1}"
PROSPECTIVE_ROOT="${PROSPECTIVE_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/corrected_f6_wide_margin_roster/through_2025_v1}"
HOLDOUT_ROOT="${HOLDOUT_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/corrected_f6_wide_margin_roster/holdout_2025_v1}"
CONDA_ENV="${CONDA_ENV:-gridiron}"
TC="${TC:-12}"
HVMEM="${HVMEM:-24G}"
HOLD_ARGS=()
if [[ -n "${HPS_HOLD_JID:-}" ]]; then HOLD_ARGS=(-hold_jid "${HPS_HOLD_JID}"); fi
mkdir -p "${SELECTION_ROOT}/job_logs" "${PROSPECTIVE_ROOT}/job_logs" "${HOLDOUT_ROOT}/job_logs"

prep="$(qsub -terse -clear -cwd -V -j y -q long -N tdnet_f6wide_select "${HOLD_ARGS[@]}" \
  -pe smp 1 -l h_rt=02:00:00 -l h_vmem=8G -o "${SELECTION_ROOT}/job_logs/" \
  -v "REPO_ROOT=${ROOT},GAP_SELECTION=${GAP_SELECTION},SCIENTIFIC_SELECTION=${SCIENTIFIC_SELECTION},FINGERPRINT_DATA=${FINGERPRINT_DATA},SELECTION_ROOT=${SELECTION_ROOT},CONDA_ENV=${CONDA_ENV}" \
  "${ROOT}/scripts/sge/corrected_f6_wide_roster_prep.sge")"
prep_id="$(printf '%s' "${prep}" | grep -Eo '[0-9]+' | head -1)"
[[ -n "${prep_id}" ]] || { echo "Could not parse prep job: ${prep}" >&2; exit 2; }
manifest="${SELECTION_ROOT}/refit_manifest.csv"

submit_refit() {
  local label="$1" output="$2" train_end="$3"
  local array final array_id
  array="$(qsub -terse -clear -cwd -V -j y -q long -N "tdnet_f6w_${label}" -hold_jid "${prep_id}" \
    -t 1-34 -tc "${TC}" -pe smp 1 -l h_rt=24:00:00 -l h_vmem="${HVMEM}" \
    -o "${output}/job_logs/" \
    -v "REPO_ROOT=${ROOT},MANIFEST=${manifest},FINGERPRINT_DATA=${FINGERPRINT_DATA},ROSTER_OUTPUT_ROOT=${output},TRAIN_END_SEASON=${train_end},CONDA_ENV=${CONDA_ENV}" \
    "${ROOT}/scripts/sge/corrected_f6_wide_roster_task.sge")"
  array_id="$(printf '%s' "${array}" | grep -Eo '[0-9]+' | head -1)"
  [[ -n "${array_id}" ]] || { echo "Could not parse ${label} array: ${array}" >&2; exit 2; }
  final="$(qsub -terse -clear -cwd -V -j y -q long -N "tdnet_f6w_${label}_final" -hold_jid "${array_id}" \
    -pe smp 1 -l h_rt=08:00:00 -l h_vmem=24G -o "${output}/job_logs/" \
    -v "REPO_ROOT=${ROOT},MANIFEST=${manifest},FINGERPRINT_DATA=${FINGERPRINT_DATA},ROSTER_OUTPUT_ROOT=${output},TRAIN_END_SEASON=${train_end},CONDA_ENV=${CONDA_ENV}" \
    "${ROOT}/scripts/sge/corrected_f6_wide_roster_finalize.sge")"
  echo "${label}_array=${array}"
  echo "${label}_finalize=${final}"
}

echo "selection=${prep}"
submit_refit prospective "${PROSPECTIVE_ROOT}" 2025
submit_refit holdout "${HOLDOUT_ROOT}" 2024
