#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-gridiron}"
TC="${TC:-50}"
HVMEM="${HVMEM:-16G}"
HEATMAP_ROOT="${HEATMAP_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_heatmaps/corrected_full_ladder_v1}"
COMPRESSION_ROOT="${COMPRESSION_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/f6_compression/sequential_nested_v1}"
DATA="${DATA:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/fingerprint_ladder_v3/canonical_fingerprint.parquet}"
LEGACY_ROSTER="${LEGACY_ROSTER:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_refits/f0_f8_margin_through_2025_v1/final_model_inventory.csv}"
LEGACY_F7="${LEGACY_F7:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/new_fingerprint_hps/experiments/publication_hps_new_fingerprints_scientific_v1/summary/tables/best_configuration_by_cell.parquet}"
CORRECTED="${CORRECTED:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/new_fingerprint_hps/experiments/publication_hps_fingerprint_contract_corrected_scientific_v3/summary/tables/best_configuration_by_cell.parquet}"
SHAP_IMPORTANCE="${SHAP_IMPORTANCE:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_shap/provisional_corrected_f6_v1/summary/source_importance.parquet}"

mkdir -p "${HEATMAP_ROOT}/job_logs" "${COMPRESSION_ROOT}/job_logs"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
conda run -n "${CONDA_ENV}" python -m gridiron_ml.cli.publication.build_scientific_heatmap_eval_manifest \
  --legacy-roster "${LEGACY_ROSTER}" --legacy-f7 "${LEGACY_F7}" --corrected "${CORRECTED}" \
  --data "${DATA}" --output-root "${HEATMAP_ROOT}"
conda run -n "${CONDA_ENV}" python -m gridiron_ml.cli.publication.build_f6_compression_manifest \
  --config "${REPO_ROOT}/configs/publication/f6_compression_study.yaml" \
  --shap-importance "${SHAP_IMPORTANCE}" --corrected-selection "${CORRECTED}" \
  --data "${DATA}" --output-root "${COMPRESSION_ROOT}"

heatmap_array="$(qsub -terse -clear -cwd -V -j y -q long -N tdnet_heatmap_eval \
  -t 1-1080 -tc "${TC}" -pe smp 1 -l h_rt=24:00:00 -l h_vmem="${HVMEM}" \
  -o "${HEATMAP_ROOT}/job_logs/" \
  -v "REPO_ROOT=${REPO_ROOT},HEATMAP_ROOT=${HEATMAP_ROOT},CONDA_ENV=${CONDA_ENV}" \
  "${REPO_ROOT}/scripts/sge/scientific_heatmap_eval_task.sge")"
heatmap_id="$(printf '%s' "${heatmap_array}" | grep -Eo '[0-9]+' | head -1)"

compression_array="$(qsub -terse -clear -cwd -V -j y -q long -N tdnet_f6c_eval \
  -t 1-1080 -tc "${TC}" -pe smp 1 -l h_rt=24:00:00 -l h_vmem="${HVMEM}" \
  -o "${COMPRESSION_ROOT}/job_logs/" \
  -v "REPO_ROOT=${REPO_ROOT},COMPRESSION_ROOT=${COMPRESSION_ROOT},CONDA_ENV=${CONDA_ENV}" \
  "${REPO_ROOT}/scripts/sge/f6_compression_task.sge")"
compression_id="$(printf '%s' "${compression_array}" | grep -Eo '[0-9]+' | head -1)"

compression_finalize="$(qsub -terse -clear -cwd -V -j y -q long -N tdnet_f6c_finalize \
  -hold_jid "${compression_id}" -pe smp 1 -l h_rt=08:00:00 -l h_vmem=8G \
  -o "${COMPRESSION_ROOT}/job_logs/" \
  -v "REPO_ROOT=${REPO_ROOT},COMPRESSION_ROOT=${COMPRESSION_ROOT},CONDA_ENV=${CONDA_ENV}" \
  "${REPO_ROOT}/scripts/sge/f6_compression_finalize.sge")"
compression_finalize_id="$(printf '%s' "${compression_finalize}" | grep -Eo '[0-9]+' | head -1)"

heatmap_finalize="$(qsub -terse -clear -cwd -V -j y -q long -N tdnet_heatmap_finalize \
  -hold_jid "${heatmap_id},${compression_finalize_id}" -pe smp 1 -l h_rt=08:00:00 -l h_vmem=12G \
  -o "${HEATMAP_ROOT}/job_logs/" \
  -v "REPO_ROOT=${REPO_ROOT},HEATMAP_ROOT=${HEATMAP_ROOT},COMPRESSION_ROOT=${COMPRESSION_ROOT},CONDA_ENV=${CONDA_ENV}" \
  "${REPO_ROOT}/scripts/sge/scientific_heatmap_eval_finalize.sge")"

echo "heatmap_array=${heatmap_array}"
echo "compression_array=${compression_array}"
echo "compression_finalize=${compression_finalize}"
echo "heatmap_finalize=${heatmap_finalize}"
