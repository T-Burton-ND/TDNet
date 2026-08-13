import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).parents[1] / "src/gridiron_ml/cli/publication/run_canonical_margin_hps_task.py"
_SPEC = importlib.util.spec_from_file_location("canonical_margin_hps_worker", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
audit_row = _MODULE.audit_row


def test_canonical_margin_hps_worker_rejects_holdout_rows(tmp_path):
    source = tmp_path / "fingerprint.parquet"
    pd.DataFrame({"season": [2023, 2024, 2025]}).to_parquet(source)
    row = {
        "objective": "margin",
        "canonical_feature_config": "F2",
        "train_years_json": json.dumps([2010, 2023, 2025]),
        "val_years_json": json.dumps([2024]),
        "test_years_json": json.dumps([]),
        "fingerprint_path": str(source),
        "model_config_path": "configs/models/linear/config_ridge.yaml",
        "feature_registry": "configs/features/feature_registry.yaml",
        "feature_ladders": "configs/features/feature_ladders.yaml",
        "output_dir": str(tmp_path / "out"),
    }
    with pytest.raises(ValueError, match="holdout/prospective"):
        audit_row(row)


def test_canonical_margin_hps_worker_audits_legal_source(tmp_path):
    source = tmp_path / "fingerprint.parquet"
    pd.DataFrame({"season": [2023, 2024]}).to_parquet(source)
    row = {
        "objective": "margin",
        "canonical_feature_config": "F5",
        "train_years_json": json.dumps([2010, 2023]),
        "val_years_json": json.dumps([2024]),
        "test_years_json": json.dumps([]),
        "fingerprint_path": str(source),
        "model_config_path": "configs/models/linear/config_ridge.yaml",
        "feature_registry": "configs/features/feature_registry.yaml",
        "feature_ladders": "configs/features/feature_ladders.yaml",
        "output_dir": str(tmp_path / "out"),
    }
    audit = audit_row(row)
    assert audit["status"] == "pass"
    assert audit["holdout_2025_excluded"] is True
