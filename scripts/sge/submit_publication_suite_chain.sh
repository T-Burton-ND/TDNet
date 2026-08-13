#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CATALOG="${CATALOG:-${TDNET_ARTIFACT_ROOT:-publication_artifacts}/suite_manifests/array_catalog.csv}"
ARRAYS=(publication_matrix hps_spline hps_hist_gradient_boosted hps_mlp hps_structured_mlp hps_kernel hps_temporal legacy_balanced_recovery legacy_margin_recovery legacy_ablation_recovery final_model_diagnostics preseason_saved_states)
PREVIOUS=""

cd "${REPO_ROOT}"
for array_id in "${ARRAYS[@]}"; do
  command=(scripts/sge/submit_publication_suite_array.sh --catalog "${CATALOG}" --array-id "${array_id}" --validated-suite --submit)
  if [[ -n "${PREVIOUS}" ]]; then
    command+=(--hold-jid "${PREVIOUS}")
  fi
  output="$("${command[@]}")"
  printf '%s\n' "${output}"
  PREVIOUS="$(printf '%s\n' "${output}" | sed -nE 's/.*job-array ([0-9]+).*/\1/p' | tail -n 1)"
  [[ -n "${PREVIOUS}" ]] || { echo "Could not parse qsub job id for ${array_id}; stopping chain." >&2; exit 2; }
done
printf 'final_job_id=%s\n' "${PREVIOUS}"
