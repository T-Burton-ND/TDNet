#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$REPO_ROOT/configs/publication/hps_mlp.yaml"
DATA_PATH="$REPO_ROOT/data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${TDNET_ARTIFACT_ROOT:-publication_artifacts}}"
JOB_ROOT="$ARTIFACT_ROOT/experiments/publication_hps_mlp_v2"
CONDA_ENV="gridiron"
SMOKE=false
SUBMIT=false
RESUME=false
MAX_TRIALS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; JOB_ROOT="$ARTIFACT_ROOT/experiments/publication_hps_mlp_v2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --max-trials) MAX_TRIALS="$2"; shift 2 ;;
    --smoke-test) SMOKE=true; shift ;;
    --resume) RESUME=true; shift ;;
    --submit) SUBMIT=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REPO_ROOT/docs/logs/sge"

if [[ ! -f "$JOB_ROOT/job_manifest.parquet" ]]; then
  build_args=(
    python "$REPO_ROOT/src/gridiron_ml/cli/experiments/build_experiment_manifest.py"
    --project-root "$REPO_ROOT"
    --config "$CONFIG_PATH"
    --data-path "$DATA_PATH"
    --output-root "$ARTIFACT_ROOT"
  )
  if [[ -n "$MAX_TRIALS" ]]; then
    build_args+=(--max-trials "$MAX_TRIALS")
  fi
  "${build_args[@]}"
fi

CHUNK_COUNT="$(python -c 'import pandas as pd,sys; p=sys.argv[1]; d=pd.read_parquet(p); print(len(d))' "$JOB_ROOT/chunk_manifest.parquet")"
echo "repo_root=$REPO_ROOT"
echo "config=$CONFIG_PATH"
echo "data_path=$DATA_PATH"
echo "job_manifest=$JOB_ROOT/job_manifest.parquet"
echo "chunk_count=$CHUNK_COUNT"
echo "sge_layout=one array task at a time; 16 slots; 16 single-threaded workers"
echo "thread_controls=OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 TORCH_INTEROP_THREADS=1"
echo "resume_incomplete=$RESUME"

if [[ "$SMOKE" == true ]]; then
  python "$REPO_ROOT/src/gridiron_ml/cli/experiments/run_experiment_chunk.py" \
    --job-manifest "$JOB_ROOT/job_manifest.parquet" --chunk-id 0 --workers 1
  touch "$JOB_ROOT/smoke_test.ok"
  echo "smoke_test=PASS"
fi

if [[ "$SUBMIT" == true ]]; then
  [[ -f "$JOB_ROOT/smoke_test.ok" ]] || { echo "Refusing submission: run --smoke-test first." >&2; exit 2; }
  command=(qsub -terse -cwd -V -N tdnet_publication_mlp -t "1-${CHUNK_COUNT}" -tc 1 \
    -pe smp 16 -l h_rt=48:00:00 -l h_vmem=4G \
    -v "REPO_ROOT=$REPO_ROOT,CONDA_ENV=$CONDA_ENV,JOB_MANIFEST=$JOB_ROOT/job_manifest.parquet,RETRY_INCOMPLETE=$RESUME" \
    "$REPO_ROOT/scripts/sge/publication_mlp_task.sge")
  echo "submitted_command=${command[*]}"
  submission_log="$REPO_ROOT/docs/logs/sge/publication_mlp_submission.stdout"
  "${command[@]}" | tee "$submission_log"
else
  echo "submit=DRY_RUN"
fi
