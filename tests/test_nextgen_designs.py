import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.experiments.nextgen_designs import (
    Formula, baseline_formulas, feature_record, reciprocal_counterparts,
)
from gridiron_ml.experiments.nextgen_contract import validate_feature_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_full_designs_preserve_canonical_a_and_c_input_limit():
    names = json.loads((ROOT / "docs/publication_2026/feature_manifests/F6.json").read_text())["feature_names"]
    schema = json.loads((ROOT / "configs/experiments/nextgen_feature_manifest_schema_v1.json").read_text())
    a = baseline_formulas(names, "a")
    assert [x.name for x in a] == names and all(x.operation == "identity" for x in a)
    for design in "abc":
        formulas = baseline_formulas(names, design)
        cp = reciprocal_counterparts([x.name for x in formulas])
        records = [feature_record(x, "F06", design, cp[x.name], endpoints=["canonical_f6_source"]) for x in formulas]
        validate_feature_manifest(records, schema)
        assert all(set(x.inputs) <= set(names) and len(x.inputs) <= 5 for x in formulas)
    assert len(baseline_formulas(names, "b")) > len(a)
    assert 60 <= len(baseline_formulas(names, "c")) < len(a)


def test_composites_do_not_change_definition_when_data_are_missing():
    frame = pd.DataFrame({"x": [1., 1., np.nan], "y": [3., np.nan, 2.]})
    mean = Formula("m", ("x", "y"), "mean", "consensus", "yards")
    assert mean.evaluate(frame).iloc[0] == 2
    assert mean.evaluate(frame).iloc[1:].isna().all()
    ratio = Formula("r", ("x", "y"), "ratio", "rate", "fraction")
    assert ratio.evaluate(pd.DataFrame({"x": [1., 1.], "y": [0., -1.]})).isna().all()
    assert "COUNT" in mean.excel and "NA()" in ratio.excel
