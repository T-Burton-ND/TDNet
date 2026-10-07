#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
list="$repo_root/scripts/cold_training_sources_2026.txt"
run_root="${TDNET_COLD_ARCHIVE_ROOT:-/groups/bsavoie2/tburton2/TDNet/training_archives/2026-10-07}"
log_root="$repo_root/.archive_jobs/2026-10-07"
count="$(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "$list" | wc -l)"
if (( count < 1 )); then
  echo "No sources configured" >&2
  exit 2
fi
mkdir -p "$log_root"
submission="$(qsub -terse -clear -cwd -j y -q long -N td_cold_zst \
  -t "1-$count" -tc 2 -pe smp 2 -l h_rt=48:00:00 -l h_vmem=8G \
  -o "$log_root" -v "TDNET_ARCHIVE_LIST=$list,TDNET_ARCHIVE_ROOT=$run_root" \
  "$repo_root/scripts/sge/compress_cold_training_task.sge")"
if [[ "$submission" =~ ([0-9]+)\. ]]; then
  job_id="${BASH_REMATCH[1]}"
else
  echo "Could not parse job ID from qsub response: $submission" >&2
  exit 1
fi
python3 - "$log_root/submission.json" "$job_id" "$count" "$list" "$run_root" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
receipt, job_id, count, source_list, archive_root = sys.argv[1:]
Path(receipt).write_text(json.dumps({
    'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
    'job_id': job_id,
    'tasks': int(count),
    'source_list': source_list,
    'archive_root': archive_root,
    'source_retained': True,
}, indent=2, sort_keys=True) + '\n')
PY
printf '%s\n' "$job_id"
