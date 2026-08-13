#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_MANIFEST=""
CHUNK_MANIFEST=""
MAX_CHUNKS=""
TASK_CONCURRENCY=10
JOB_NAME=tdnet_publication_matrix
SUBMIT=false
SMOKE_TEST=false
CONDA_ENV=gridiron

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-manifest) JOB_MANIFEST="$2"; shift 2 ;;
    --chunk-manifest) CHUNK_MANIFEST="$2"; shift 2 ;;
    --max-chunks) MAX_CHUNKS="$2"; shift 2 ;;
    --task-concurrency) TASK_CONCURRENCY="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --smoke-test) SMOKE_TEST=true; shift ;;
    --submit) SUBMIT=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${JOB_MANIFEST}" ]] || { echo "--job-manifest is required" >&2; exit 2; }
[[ -n "${CHUNK_MANIFEST}" ]] || CHUNK_MANIFEST="$(dirname "${JOB_MANIFEST}")/chunk_manifest.parquet"

PYTHON_CMD=(python)
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
fi

CHUNK_COUNT="$("${PYTHON_CMD[@]}" -c 'import pandas as pd,sys; p=sys.argv[1]; d=pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p); print(len(d))' "${CHUNK_MANIFEST}")"
if [[ -n "${MAX_CHUNKS}" && "${MAX_CHUNKS}" -lt "${CHUNK_COUNT}" ]]; then
  CHUNK_COUNT="${MAX_CHUNKS}"
fi
if [[ "${SMOKE_TEST}" == true && "${CHUNK_COUNT}" -gt 5 ]]; then
  CHUNK_COUNT=5
fi

echo "job_manifest=${JOB_MANIFEST}"
echo "chunk_manifest=${CHUNK_MANIFEST}"
echo "array=1-${CHUNK_COUNT}"
echo "task_concurrency=${TASK_CONCURRENCY}"
echo "submit=${SUBMIT}"

if [[ "${SMOKE_TEST}" == true ]]; then
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_CMD[@]}" src/gridiron_ml/cli/experiments/run_experiment_chunk.py \
    --job-manifest "${JOB_MANIFEST}" --chunk-id 0 --workers 1
  touch "$(dirname "${JOB_MANIFEST}")/smoke_test.ok"
fi

if [[ "${SUBMIT}" == true ]]; then
  [[ -f "$(dirname "${JOB_MANIFEST}")/smoke_test.ok" ]] || {
    echo "Refusing submission: run this wrapper with --smoke-test first." >&2
    exit 2
  }
  qsub -terse -cwd -V \
    -N "${JOB_NAME}" -t "1-${CHUNK_COUNT}" -tc "${TASK_CONCURRENCY}" \
    -pe smp 1 -l h_rt=08:00:00 -l h_vmem=4G \
    -v "REPO_ROOT=${REPO_ROOT},JOB_MANIFEST=${JOB_MANIFEST},CONDA_ENV=${CONDA_ENV}" \
    "${REPO_ROOT}/scripts/sge/experiment_chunk_task.sge"
else
  echo "Dry run only. Add --submit after smoke-test validation."
fi
