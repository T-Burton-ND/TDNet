#!/usr/bin/env bash
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARTIFACT_ROOT="${TDNET_ARTIFACT_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts}"
BUNDLE="${BUNDLE:-$ARTIFACT_ROOT/scientific_roster_refits/f0_f8_margin_through_2025_v1}"
MANIFEST="${MANIFEST:-$BUNDLE/refit_manifest.csv}"
FROZEN_INVENTORY="${FROZEN_INVENTORY:-$BUNDLE/final_model_inventory.csv}"
OOF_ROOT="${OOF_ROOT:-$ARTIFACT_ROOT/scientific_calibration_oof/f0_f8_through_2025_v1}"
CONDA_ENV="${CONDA_ENV:-gridiron}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
HVMEM="${HVMEM:-32G}"

cd "$ROOT"
mkdir -p "$BUNDLE/job_logs/calibration" "$OOF_ROOT"
submission="$(qsub -terse -clear -cwd -V -j y -q long -N tdnet_scientific_oof -t 1-54 -tc "$MAX_CONCURRENT" -pe smp 1 -l h_rt=48:00:00 -l h_vmem="$HVMEM" -o "$BUNDLE/job_logs/calibration/" -v "REPO_ROOT=$ROOT,MANIFEST=$MANIFEST,FROZEN_INVENTORY=$FROZEN_INVENTORY,OOF_ROOT=$OOF_ROOT,CONDA_ENV=$CONDA_ENV" "$ROOT/scripts/publication/run_scientific_cross_fitted_oof_sge_task.sh")"
array_job_id="$(printf '%s' "$submission" | grep -Eo '[0-9]+' | head -1)"
[[ -n "$array_job_id" ]] || { echo "Could not parse OOF array job ID: $submission" >&2; exit 2; }
final_submission="$(qsub -terse -clear -cwd -V -j y -q long -N tdnet_scientific_calibrators -hold_jid "$array_job_id" -pe smp 1 -l h_rt=04:00:00 -l h_vmem=16G -o "$BUNDLE/job_logs/calibration/" -v "REPO_ROOT=$ROOT,OOF_ROOT=$OOF_ROOT,BUNDLE=$BUNDLE,CONDA_ENV=$CONDA_ENV" "$ROOT/scripts/publication/fit_scientific_frozen_calibrators_sge_task.sh")"
echo "OOF_ARRAY=$submission"
echo "CALIBRATOR_FINALIZE=$final_submission"
