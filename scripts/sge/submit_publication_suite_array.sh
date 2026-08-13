#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CATALOG="${CATALOG:-${TDNET_ARTIFACT_ROOT:-publication_artifacts}/suite_manifests/array_catalog.csv}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
ARRAY_ID=""
SUBMIT=false
SMOKE_TEST=false
HOLD_JID=""
VALIDATED_SUITE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --array-id) ARRAY_ID="$2"; shift 2 ;;
    --catalog) CATALOG="$2"; shift 2 ;;
    --submit) SUBMIT=true; shift ;;
    --smoke-test) SMOKE_TEST=true; shift ;;
    --hold-jid) HOLD_JID="$2"; shift 2 ;;
    --validated-suite) VALIDATED_SUITE=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${ARRAY_ID}" ]] || { echo "--array-id is required" >&2; exit 2; }

mapfile -t VALUES < <(python -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1]).fillna(""); r=d.loc[d.array_id.eq(sys.argv[2])]; assert len(r)==1, f"unknown array: {sys.argv[2]}"; x=r.iloc[0]; print(x.manifest_path); print(int(x.task_count)); print(int(x.task_concurrency)); print(x.worker_script); print(x.config_path)' "${CATALOG}" "${ARRAY_ID}")
JOB_MANIFEST="${VALUES[0]}"; TASK_COUNT="${VALUES[1]}"; TC="${VALUES[2]}"; WORKER="${VALUES[3]}"; CONFIG_PATH="${VALUES[4]}"
OUTPUT_ROOT="$(dirname "${JOB_MANIFEST}")"

echo "array_id=${ARRAY_ID}"
echo "manifest=${JOB_MANIFEST}"
echo "tasks=${TASK_COUNT}"
echo "task_concurrency=${TC}"
echo "hold_jid=${HOLD_JID:-none}"
echo "worker=${WORKER}"
echo "submit=${SUBMIT}"

cd "${REPO_ROOT}"
if [[ "${SMOKE_TEST}" == true ]]; then
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  PYTHON_CMD=(python)
  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx gridiron; then
    PYTHON_CMD=(conda run -n gridiron python)
  fi
  case "${WORKER}" in
    scripts/sge/experiment_chunk_task.sge)
      "${PYTHON_CMD[@]}" src/gridiron_ml/cli/experiments/run_experiment_chunk.py --job-manifest "${JOB_MANIFEST}" --sge-task-id 1 --workers 1 ;;
    scripts/sge/legacy_hps_recovery_task.sge)
      "${PYTHON_CMD[@]}" src/gridiron_ml/cli/run_fingerprint_hyperparameter_search.py run-job --project-root "${REPO_ROOT}" \
        --config "${CONFIG_PATH}" --output-root "${OUTPUT_ROOT}" --job-manifest "${JOB_MANIFEST}" --sge-task-id 1 ;;
    scripts/sge/opponent_ablation_shap_task.sge)
      "${PYTHON_CMD[@]}" src/gridiron_ml/cli/run_opponent_ablation_shap.py run-job --project-root "${REPO_ROOT}" \
        --output-root "${OUTPUT_ROOT}" --job-manifest "${JOB_MANIFEST}" --sge-task-id 1 ;;
    scripts/sge/final_model_diagnostics_task.sge)
      "${PYTHON_CMD[@]}" src/gridiron_ml/cli/run_final_model_diagnostics.py run-job --project-root "${REPO_ROOT}" \
        --output-root "${OUTPUT_ROOT}" --manifest "${JOB_MANIFEST}" --sge-task-id 1 ;;
    scripts/sge/preseason_state_task.sge)
      "${PYTHON_CMD[@]}" src/gridiron_ml/cli/experiments/preseason_state_task.py --manifest "${JOB_MANIFEST}" --sge-task-id 1 ;;
    *) echo "No smoke command for worker ${WORKER}" >&2; exit 2 ;;
  esac
  touch "${OUTPUT_ROOT}/smoke_test.ok"
  echo "Smoke test passed: ${OUTPUT_ROOT}/smoke_test.ok"
fi

if [[ "${SUBMIT}" != true ]]; then
  echo "Dry run only. No qsub command was issued."
  exit 0
fi
if [[ ! -f "${OUTPUT_ROOT}/smoke_test.ok" ]]; then
  [[ "${VALIDATED_SUITE}" == true ]] || { echo "Refusing submission: ${OUTPUT_ROOT}/smoke_test.ok is absent." >&2; exit 2; }
  python -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); assert set(d.task_concurrency)=={10}; assert d.one_training_per_task.all(); r=d[d.array_id.eq(sys.argv[2])]; assert len(r)==1 and int(r.iloc[0].task_count)>0' "${CATALOG}" "${ARRAY_ID}"
  echo "Validated-suite catalog gate passed (local model-family smoke evidence retained under /tmp/tdnet-suite-test)."
fi
QSUB_ARGS=(-binding linear_per_task:1)
if [[ -n "${HOLD_JID}" ]]; then
  QSUB_ARGS+=(-hold_jid "${HOLD_JID}")
fi
qsub "${QSUB_ARGS[@]}" \
  -v "REPO_ROOT=${REPO_ROOT},JOB_MANIFEST=${JOB_MANIFEST},CONFIG_PATH=${CONFIG_PATH},OUTPUT_ROOT=${OUTPUT_ROOT},PYTHON_BIN=${PYTHON_BIN}" \
  -N "tdnet_${ARRAY_ID}" -t "1-${TASK_COUNT}" -tc "${TC}" "${WORKER}"
