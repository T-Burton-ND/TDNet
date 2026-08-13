#!/usr/bin/env bash
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/2025_roster_regenerations}"
RANKING="${RANKING:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/corrected_f6_wide_margin_roster/holdout_2025_v1/preseason_model_rankings.csv}"
CONDA_ENV="${CONDA_ENV:-gridiron}"
HOLDOUT_SCIENTIFIC_JOB="${HOLDOUT_SCIENTIFIC_JOB:-}"

submit_build() {
  local name="$1" inventory="$2" staged="$3" allow="$4" hold="${5:-}"
  local hold_args=()
  if [[ -n "$hold" ]]; then hold_args=(-hold_jid "$hold"); fi
  qsub -terse -clear -cwd -V -j y -q long -N "$name" "${hold_args[@]}" \
    -pe smp 1 -l h_rt=24:00:00 -l h_vmem=24G \
    -o "$ARTIFACT_ROOT/job_logs/" \
    -v "REPO_ROOT=$ROOT,INVENTORY=$inventory,RANKING=$RANKING,STAGED_ROOT=$staged,ALLOW_TRAINING_THROUGH_HOLDOUT=$allow,CONDA_ENV=$CONDA_ENV" \
    "$ROOT/scripts/sge/build_2025_holdout_examples.sge"
}

mkdir -p "$ARTIFACT_ROOT/job_logs"
scientific_through_2025="${SCIENTIFIC_THROUGH_2025:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_refits/f0_f8_margin_through_2025_v1/final_model_inventory.csv}"
wide_through_2025="${WIDE_THROUGH_2025:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/corrected_f6_wide_margin_roster/through_2025_v1/final_model_inventory.csv}"
scientific_holdout="${SCIENTIFIC_HOLDOUT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_refits/f0_f8_margin_holdout_2025_v1/final_model_inventory.csv}"
wide_holdout="${WIDE_HOLDOUT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/corrected_f6_wide_margin_roster/holdout_2025_v1/final_model_inventory.csv}"

echo "scientific_through_2025=$(submit_build tdnet25_sci_dry "$scientific_through_2025" "$ARTIFACT_ROOT/through_2025/scientific" 1)"
echo "wide_through_2025=$(submit_build tdnet25_wide_dry "$wide_through_2025" "$ARTIFACT_ROOT/through_2025/wide_f6" 1)"
echo "wide_holdout_2025=$(submit_build tdnet25_wide_hold "$wide_holdout" "$ARTIFACT_ROOT/holdout_2025/wide_f6" 0)"
echo "scientific_holdout_2025=$(submit_build tdnet25_sci_hold "$scientific_holdout" "$ARTIFACT_ROOT/holdout_2025/scientific" 0 "$HOLDOUT_SCIENTIFIC_JOB")"
