#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TDNET_HOLDOUT_ROOT:-holdout_2025_refits}}"
JOB_MANIFEST="${JOB_MANIFEST:-${OUTPUT_ROOT}/job_manifest.csv}"
HOLD_JID=""
SUBMIT=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hold-jid) HOLD_JID="$2"; shift 2 ;;
    --submit) SUBMIT=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -f "${JOB_MANIFEST}" ]] || { echo "Missing ${JOB_MANIFEST}" >&2; exit 2; }
echo "36 one-model holdout refits; train=2010-2023; val=2024; eval=2025; tc=10"
if [[ "${SUBMIT}" != true ]]; then
  echo "Dry run only. Add --submit to submit refit -> poll finalize -> publication chain."
  exit 0
fi
args=(-V -N tdnet_2025_holdout -t 1-36 -tc 10 -v "REPO_ROOT=${REPO_ROOT},JOB_MANIFEST=${JOB_MANIFEST}" scripts/sge/holdout_2025_refit_task.sge)
if [[ -n "${HOLD_JID}" ]]; then args=(-hold_jid "${HOLD_JID}" "${args[@]}"); fi
array_output="$(qsub "${args[@]}")"
echo "${array_output}"
array_id="$(printf '%s\n' "${array_output}" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')"
[[ -n "${array_id}" ]] || { echo "Could not parse refit array ID." >&2; exit 2; }
finalize_output="$(qsub -V -N tdnet_2025_holdout_finalize -hold_jid "${array_id}" -v "REPO_ROOT=${REPO_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}" scripts/sge/holdout_2025_finalize.sge)"
echo "${finalize_output}"
finalize_id="$(printf '%s\n' "${finalize_output}" | sed -nE 's/.*job ([0-9]+).*/\1/p')"
[[ -n "${finalize_id}" ]] || { echo "Could not parse finalize job ID." >&2; exit 2; }
qsub -V -N tdnet_2025_holdout_publish -hold_jid "${finalize_id}" \
  -v "REPO_ROOT=${REPO_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}" scripts/sge/holdout_2025_publish.sge
