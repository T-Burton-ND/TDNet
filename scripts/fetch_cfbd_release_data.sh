#!/usr/bin/env bash
set -euo pipefail

# Fetch the raw CFBD inputs needed to reproduce TDNet fingerprints.
# Credentials are never written to disk; CFBD_API_KEY must already be in the
# caller's environment.  The fetcher is cache-aware, so rerunning this script
# only requests missing files unless --refresh is supplied.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_YEAR="${START_YEAR:-2010}"
END_YEAR="${END_YEAR:-2025}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/data/raw/cfbd/v2}"
CONFIG="$ROOT/configs/fetch/cfbd_single_year.yaml"

if [[ -z "${CFBD_API_KEY:-}" ]]; then
  echo "Set CFBD_API_KEY in the environment before running this script." >&2
  exit 2
fi

for ((year=START_YEAR; year<=END_YEAR; year++)); do
  args=(--config "$CONFIG" --year "$year" --output-root "$OUTPUT_ROOT")
  if [[ "${REFRESH:-0}" == "1" ]]; then
    args+=(--refresh)
  fi
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    python -m gridiron_ml.pipeline.fetch.cfbd_fetch_v2 "${args[@]}"
done
