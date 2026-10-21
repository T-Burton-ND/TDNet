#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/publication/2026/week_08/pre_game/scientific/full_f0_f19/f09_f17"
mkdir -p "$OUT_DIR/scheduler_logs"
if [[ ! -f "$REPO_ROOT/data/what_if_2026_fingerprints/week08_expanded_states/manifest.json" ]]; then
  echo "Week 8 expanded source states are missing" >&2
  exit 2
fi
if [[ ! -f "$REPO_ROOT/publication/2026/week_08/pre_game/scientific/full_f0_f19/f17_f19_market_snapshot.parquet" ]]; then
  echo "Week 8 F17/F19 market snapshot is missing" >&2
  exit 2
fi
JOB_ID="$(qsub -terse -clear -cwd -j y -q long \
  -N td_w08_exp -t 1-54 -tc 8 -pe smp 1 \
  -l h_rt=06:00:00 -l h_vmem=16G \
  -o "$OUT_DIR/scheduler_logs" \
  -v "REPO_ROOT=$REPO_ROOT" \
  "$REPO_ROOT/scripts/sge/expanded_week08_forecast_task.sge")"
export JOB_ID OUT_DIR
/users/tburton2/.conda/envs/gridiron/bin/python - <<'PY'
import json,os
from datetime import datetime,timezone
from pathlib import Path
path=Path(os.environ['OUT_DIR'])/'submission.json'
path.write_text(json.dumps({
    'submitted_at_utc':datetime.now(timezone.utc).isoformat(),
    'job_id':os.environ['JOB_ID'],'tasks':54,'task_concurrency':8,
    'deadline':'2026-10-22T00:00:00Z',
},indent=2)+'\n')
PY
echo "$JOB_ID"
