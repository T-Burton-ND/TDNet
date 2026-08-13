#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m pytest -q \
  tests/test_eval_training.py \
  tests/test_model_catalog_contract.py \
  tests/test_matchup_unit.py \
  tests/test_model_feature_adapter.py \
  tests/test_season_vs_vegas_eval.py \
  tests/test_td_eval_artifacts.py \
  tests/test_tdlinear_loss.py \
  tests/test_tdsim_checkpoints.py \
  tests/test_shap_analysis.py \
  tests/test_data_points.py \
  "$@"
