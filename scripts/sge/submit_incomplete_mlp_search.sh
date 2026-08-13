#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"
JOB_MANIFEST="$ARTIFACT_ROOT/experiments/publication_hps_mlp_v2/job_manifest.parquet"
SPEC="$(PYTHONPATH="$REPO_ROOT/src" python "$REPO_ROOT/src/gridiron_ml/cli/sge/build_incomplete_mlp_task_spec.py" "$JOB_MANIFEST")"
if [[ -z "$SPEC" ]]; then
  echo "No incomplete MLP chunks found."
  exit 0
fi

IFS=',' read -r -a TASKS <<< "$SPEC"
echo "incomplete_chunk_count=${#TASKS[@]}"
submit_range() {
  local task_range="$1"
  local job_id
  job_id="$(qsub -terse -cwd -V -N tdnet_publication_mlp_retry \
    -t "$task_range" -tc 1 -pe smp 16 -l h_rt=48:00:00 -l h_vmem=8G \
    -v "REPO_ROOT=$REPO_ROOT,CONDA_ENV=gridiron,JOB_MANIFEST=$JOB_MANIFEST,RETRY_INCOMPLETE=true" \
    "$REPO_ROOT/scripts/sge/publication_mlp_task.sge")"
  echo "submitted_range=$job_id tasks=$task_range"
}

run_start="${TASKS[0]}"
previous="${TASKS[0]}"
for ((index=1; index<${#TASKS[@]}; index++)); do
  current="${TASKS[index]}"
  if (( current == previous + 1 )); then
    previous="$current"
    continue
  fi
  if [[ "$run_start" == "$previous" ]]; then
    submit_range "$run_start"
  else
    submit_range "$run_start-$previous"
  fi
  run_start="$current"
  previous="$current"
done
if [[ "$run_start" == "$previous" ]]; then
  submit_range "$run_start"
else
  submit_range "$run_start-$previous"
fi
