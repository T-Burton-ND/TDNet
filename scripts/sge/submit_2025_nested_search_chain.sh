#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${TDNET_NESTED_SEARCH_ROOT:-nested_search_2025}}"
CATALOG="${ARTIFACT_ROOT}/array_catalog.csv"
SUBMIT=false
HOLD_JID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=true; shift ;;
    --hold-jid) HOLD_JID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
python -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); assert len(d)==4 and d.task_count.sum()==165200 and d.task_count.max()<=75000 and set(d.task_concurrency)=={10} and d.one_training_per_task.all() and set(d.cv_folds)=={5}' "${CATALOG}"
echo "4 sequential SGE-safe shards; 165,200 tasks; tc=10; five rolling-origin CV folds ending 2020-2024"
if [[ "${SUBMIT}" != true ]]; then echo "Dry run only."; exit 0; fi
previous="${HOLD_JID}"
mapfile -t array_ids < <(python -c 'import pandas as pd,sys; print("\n".join(pd.read_csv(sys.argv[1]).array_id))' "${CATALOG}")
for array_id in "${array_ids[@]}"; do
  mapfile -t values < <(python -c 'import pandas as pd,sys; r=pd.read_csv(sys.argv[1]).query("array_id == @sys.argv[2]").iloc[0]; print(r.manifest_path); print(r.config_path); print(r.output_root); print(int(r.task_count))' "${CATALOG}" "${array_id}")
  args=(-V -N "tdnet_${array_id}" -t "1-${values[3]}" -tc 10 -v "REPO_ROOT=${REPO_ROOT},JOB_MANIFEST=${values[0]},CONFIG_PATH=${values[1]},OUTPUT_ROOT=${values[2]}" scripts/sge/nested_2025_hps_task.sge)
  if [[ -n "${previous}" ]]; then args=(-hold_jid "${previous}" "${args[@]}"); fi
  output="$(qsub "${args[@]}")"; echo "${output}"
  previous="$(printf '%s\n' "${output}" | sed -nE 's/.*job-array ([0-9]+).*/\1/p')"
  [[ -n "${previous}" ]] || exit 2
done
echo "final_job_id=${previous}"
