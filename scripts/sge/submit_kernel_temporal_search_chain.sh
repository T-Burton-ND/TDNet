#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${TDNET_ARTIFACT_ROOT:-publication_artifacts}}"
HOLD_JID="${HOLD_JID:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
SUBMIT=false
[[ "${1:-}" == "--submit" ]] && SUBMIT=true

DATA="${REPO_ROOT}/data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"
for spec in "hps_kernel:configs/publication/hps_kernel.yaml" "hps_temporal:configs/publication/hps_temporal.yaml"; do
  name="${spec%%:*}"; config="${spec#*:}"
  PYTHONPATH="${REPO_ROOT}/src" python "${REPO_ROOT}/src/gridiron_ml/cli/experiments/build_experiment_manifest.py" \
    --project-root "${REPO_ROOT}" --config "${config}" --data-path "${DATA}" \
    --output-root "${ARTIFACT_ROOT}" --minimum-free-gb 30
  experiment="publication_${name}_v1"
  manifest="${ARTIFACT_ROOT}/experiments/${experiment}/job_manifest.parquet"
  chunks="${ARTIFACT_ROOT}/experiments/${experiment}/chunk_manifest.parquet"
  count="$(python -c 'import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))' "${chunks}")"
  echo "${name}: ${count} one-run tasks, tc=10, hold_jid=${HOLD_JID:-none}"
  if [[ "${SUBMIT}" == true ]]; then
    [[ -f "$(dirname "${manifest}")/smoke_test.ok" ]] || { echo "Missing smoke_test.ok for ${name}" >&2; exit 2; }
    hold=(); [[ -n "${HOLD_JID}" ]] && hold=(-hold_jid "${HOLD_JID}")
    output="$(qsub -binding linear_per_task:1 "${hold[@]}" \
      -v "REPO_ROOT=${REPO_ROOT},JOB_MANIFEST=${manifest},CONDA_ENV=gridiron,PYTHON_BIN=${PYTHON_BIN}" \
      -N "tdnet_${name}" -t "1-${count}" -tc 10 "${REPO_ROOT}/scripts/sge/experiment_chunk_task.sge")"
    echo "${output}"
    HOLD_JID="$(sed -nE 's/.*job-array ([0-9]+).*/\1/p' <<<"${output}" | tail -1)"
    [[ -n "${HOLD_JID}" ]] || { echo "Could not parse qsub job id" >&2; exit 2; }
  fi
done
