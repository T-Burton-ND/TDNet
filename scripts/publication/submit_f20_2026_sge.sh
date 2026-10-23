#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG_DIR=/groups/bsavoie2/tburton2/TDNet/f20_2026/prospective/scheduler_logs
mkdir -p "$LOG_DIR"
LEDGER="$LOG_DIR/submissions.tsv"
RUNNER="$REPO_ROOT/scripts/publication/run_f20_sge.sh"

# Local campus scheduler time; all jobs are away from the DST transition hour.
# The runner calculates each game's official cutoff with America/New_York.
while read -r week monday thursday; do
  for phase_hour in monday12 tuesday12 wednesday12 thursday21 thursday2330; do
    case "$phase_hour" in
      monday12) date="${monday}1200"; phase=early ;;
      tuesday12) date="$(date -d "${monday:0:4}-${monday:4:2}-${monday:6:2} +1 day" +%Y%m%d)1200"; phase=early ;;
      wednesday12) date="$(date -d "${monday:0:4}-${monday:4:2}-${monday:6:2} +2 days" +%Y%m%d)1200"; phase=early ;;
      thursday21) date="${thursday}2100"; phase=official ;;
      thursday2330) date="${thursday}2330"; phase=official ;;
    esac
    name="f20_w${week}_${phase_hour}"
    if [[ -f "$LEDGER" ]] && awk -F '\t' -v wanted="$name" '$1 == wanted {found=1} END {exit !found}' "$LEDGER"; then
      continue
    fi
    job_id=$(qsub -terse -a "$date" -N "$name" -cwd -j y -o "$LOG_DIR" \
      -l h_rt=00:20:00 -l h_vmem=4G \
      "$RUNNER" "$week" "$phase" "$phase_hour")
    printf '%s\t%s\t%s\n' "$name" "$date" "$job_id" >> "$LEDGER"
    printf '%s %s %s\n' "$name" "$date" "$job_id"
  done
done <<'SCHEDULE'
9 20261026 20261029
10 20261102 20261105
11 20261109 20261112
12 20261116 20261119
13 20261123 20261126
15 20261207 20261210
SCHEDULE
