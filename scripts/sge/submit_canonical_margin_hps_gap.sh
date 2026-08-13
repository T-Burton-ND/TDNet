#!/usr/bin/env bash
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST="${JOB_MANIFEST:-${ROOT}/outputs/publication/canonical_margin_hps_gap_v1/job_manifest.parquet}"
CONFIG="${CONFIG_PATH:-${ROOT}/configs/models/tuning/fingerprint_hyperparameter_search_margin.yaml}"
TASK_COUNT="${TASK_COUNT:-2544}"
TC="${TASK_CONCURRENCY:-50}"
SUBMIT=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=true; shift ;;
    --dry-run) SUBMIT=false; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
python - "$MANIFEST" "$TASK_COUNT" <<'PY'
import json, pandas as pd, sys
d=pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith('.parquet') else pd.read_csv(sys.argv[1])
assert len(d)==int(sys.argv[2]), (len(d),sys.argv[2])
assert set(d.objective.astype(str))=={'margin'}
assert set(d.canonical_feature_config.astype(str))=={'F2','F3','F5'}
assert all(d.train_years_json.astype(str).map(lambda x: max(json.loads(x)) < 2025))
assert all(d.val_years_json.astype(str).map(lambda x: max(json.loads(x)) < 2025))
assert all(d.test_years_json.astype(str).map(lambda x: max(json.loads(x)) < 2025))
print(f"manifest_rows={len(d)} margin_only=true tiers=F2,F3,F5")
PY
echo "manifest=${MANIFEST}"
echo "tasks=${TASK_COUNT}"
echo "task_concurrency=${TC}"
echo "slots_per_task=10"
echo "artifact_root=${TDNET_CANONICAL_HPS_ARTIFACT_ROOT:-${ROOT}/outputs/publication/canonical_margin_hps_gap_v1/artifacts}"
echo "submit=${SUBMIT}"
if [[ "$SUBMIT" != true ]]; then
  echo "Dry run only. Add --submit to issue qsub."
  exit 0
fi
qsub -cwd -V -j y -N tdnet_canonical_margin_hps_gap \
  -pe smp 10 -l h_rt=48:00:00 -l h_vmem=32G -tc "$TC" -t "1-${TASK_COUNT}" \
  -v "REPO_ROOT=${ROOT},JOB_MANIFEST=${MANIFEST},CONFIG_PATH=${CONFIG}" \
  "${ROOT}/scripts/sge/canonical_margin_hps_task.sge"
