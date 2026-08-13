#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${1:-}"
shift || true

usage() {
  cat <<'EOF'
Usage: scripts/tdnet-run.sh <run> [tdnet-run args...]

Runs:
  data       configs/td_run/data_and_train.yaml
  eval       configs/td_run/evaluate_latest.yaml
  blog       configs/td_run/weekly_blog.yaml
  sim        configs/sim/tdsim_config.yaml through td-sim

Examples:
  scripts/tdnet-run.sh data
  scripts/tdnet-run.sh eval --workflow evaluation
  scripts/tdnet-run.sh blog
  scripts/tdnet-run.sh sim --season 2026 --n-sims 100
EOF
}

if [[ -z "${RUN_NAME}" || "${RUN_NAME}" == "-h" || "${RUN_NAME}" == "--help" ]]; then
  usage
  exit 0
fi

case "${RUN_NAME}" in
  data)
    CONFIG="${ROOT_DIR}/configs/td_run/data_and_train.yaml"
    COMMAND=(python -m gridiron_ml.td_run.cli "${CONFIG}" "$@")
    ;;
  eval)
    CONFIG="${ROOT_DIR}/configs/td_run/evaluate_latest.yaml"
    COMMAND=(python -m gridiron_ml.td_run.cli "${CONFIG}" "$@")
    ;;
  blog)
    CONFIG="${ROOT_DIR}/configs/td_run/weekly_blog.yaml"
    COMMAND=(python -m gridiron_ml.td_run.cli "${CONFIG}" "$@")
    ;;
  sim)
    CONFIG="${ROOT_DIR}/configs/sim/tdsim_config.yaml"
    COMMAND=(python -m gridiron_ml.td_sim.cli --config "${CONFIG}" "$@")
    ;;
  *)
    usage
    exit 2
    ;;
esac

mkdir -p "${ROOT_DIR}/logs"
STAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${ROOT_DIR}/docs/logs/tdnet_${RUN_NAME}_${STAMP}.out"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

{
  echo "TDNet ${RUN_NAME} run"
  echo "Started: $(date)"
  echo "Config : ${CONFIG}"
  echo "Log    : ${LOG_FILE}"
  echo "Command: ${COMMAND[*]}"
  echo "========================================================================"
  "${COMMAND[@]}"
  echo "========================================================================"
  echo "Finished: $(date)"
} 2>&1 | tee "${LOG_FILE}"
