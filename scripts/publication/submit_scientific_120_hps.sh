#!/usr/bin/env bash
# Submit only after the input and CPU smoke gates have passed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT=/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/data
RESULT_ROOT=/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/results
SNAPSHOT_ROOT=/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/code_snapshot
PLAN="${SNAPSHOT_ROOT}/publication/2026/research/scientific_120_hps_v1/plan.json"
if [[ "${1:-}" != "--submit" ]]; then
  echo "Dry run only. Use --submit after all 20 source receipts and smoke tests pass."
  exit 0
fi
if [[ ! -f "${PLAN}" ]]; then
  echo "Missing frozen HPS plan" >&2; exit 2
fi
if [[ ! -f "${SNAPSHOT_ROOT}/source_commit.txt" ]]; then
  echo "Missing frozen source snapshot" >&2; exit 2
fi
commit="$(cat "${SNAPSHOT_ROOT}/source_commit.txt")"
if [[ ! -f "${RESULT_ROOT}/smoke_passed.json" ]]; then
  echo "Missing CPU smoke acceptance receipt" >&2; exit 2
fi
if [[ ! -f "${RESULT_ROOT%/results}/input_preflight.json" ]]; then
  echo "Missing 20-tier source preflight receipt" >&2; exit 2
fi
for tier in {0..19}; do
  if [[ ! -f "${DATA_ROOT}/F${tier}/receipt.json" ]]; then
    echo "Missing F${tier} historical matrix receipt" >&2; exit 2
  fi
done
active="$(qstat -u "${USER}" | awk 'NR>2 && $3 ~ /^td_hps_/ && ($5 == "r" || $5 == "t") {count++} END {print count+0}')"
if [[ "${active}" -gt 15 ]]; then
  echo "Refusing submission: ${active} HPS tasks are already running; ten more would exceed 25" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}/logs"
qsub -N td_hps_120 -t 1-12000 -tc 10 -pe smp 1 \
  -o "${RESULT_ROOT}/logs" -e "${RESULT_ROOT}/logs" \
  -v "REPO_ROOT=${SNAPSHOT_ROOT},HPS_DATA_ROOT=${DATA_ROOT},HPS_RESULT_ROOT=${RESULT_ROOT},HPS_EXPECTED_COMMIT=${commit}" \
  "${SNAPSHOT_ROOT}/scripts/sge/scientific_120_hps_task.sge"
