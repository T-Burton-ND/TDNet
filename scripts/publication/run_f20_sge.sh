#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${SGE_O_WORKDIR:-/users/tburton2/Repos/TDNet}
F20_ROOT=/groups/bsavoie2/tburton2/TDNet/f20_2026
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /users/tburton2/.conda/envs/gridiron/bin/python \
  "$REPO_ROOT/scripts/publication/run_f20_weekly.py" \
  --week "$1" --phase "$2" --run-label "$3" \
  --archive "$F20_ROOT/prospective" \
  --schedule "$REPO_ROOT/data/raw/cfbd/v2/games/2026.parquet" \
  --models "$F20_ROOT/models/baseline_2017_2024_v1"
