#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEARCH_MANIFEST="${1:?Usage: submit_constraint_free_extra_sge.sh MANIFEST [TC]}"
TC="${2:-10}"
if ! [[ "$TC" =~ ^[0-9]+$ ]] || (( TC < 1 || TC > 10 )); then
  echo "TC must be an integer from 1 through 10" >&2
  exit 2
fi
if [[ ! -f "$SEARCH_MANIFEST" ]]; then
  echo "Missing manifest: $SEARCH_MANIFEST" >&2
  exit 2
fi
readarray -t DETAILS < <(/users/tburton2/.conda/envs/gridiron/bin/python - "$SEARCH_MANIFEST" <<'PY'
import json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text())
if manifest['task_concurrency_max']>10 or manifest['stage'] not in (1,2):
    raise SystemExit('Invalid reducer manifest or TC cap')
print(len(manifest['tasks']))
print(manifest['stage'])
PY
)
TASK_COUNT="${DETAILS[0]}"
STAGE="${DETAILS[1]}"
OUT_DIR="$(dirname "$SEARCH_MANIFEST")"
QSTAT="$(qstat -u "$USER")"
if printf '%s\n' "$QSTAT" | awk -v name="td_cf_x${STAGE}" '$3 == name {found=1} END {exit !found}'; then
  echo "An array for this reducer stage is already active" >&2
  exit 2
fi
ACTIVE="$(printf '%s\n' "$QSTAT" | awk '$3 ~ /^td_cf_/ && $5 == "r" {n++} END {print n+0}')"
if (( ACTIVE + TC > 10 )); then
  echo "Requested TC would exceed 10 active constraint-free tasks: $ACTIVE + $TC" >&2
  exit 2
fi
mkdir -p "$OUT_DIR/scheduler_logs"
JOB_ID="$(qsub -terse -clear -cwd -j y -q long \
  -N "td_cf_x${STAGE}" -t "1-${TASK_COUNT}" -tc "$TC" \
  -pe smp 1 -l h_rt=48:00:00 -l h_vmem=16G \
  -o "$OUT_DIR/scheduler_logs" \
  -v "REPO_ROOT=$REPO_ROOT,SEARCH_MANIFEST=$SEARCH_MANIFEST" \
  "$REPO_ROOT/scripts/sge/constraint_free_extra_task.sge")"
/users/tburton2/.conda/envs/gridiron/bin/python - "$OUT_DIR/submission.json" "$JOB_ID" "$TC" "$TASK_COUNT" <<'PY'
from datetime import datetime,timezone
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    'submitted_at_utc':datetime.now(timezone.utc).isoformat(),
    'job_id':sys.argv[2],'tc':int(sys.argv[3]),'tasks':int(sys.argv[4]),
},sort_keys=True,indent=2)+'\n')
PY
echo "$JOB_ID"
