#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/submit_experiment_chunks.sh" --job-name tdnet_feature_model "$@"

