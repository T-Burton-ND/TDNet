import json
from pathlib import Path
import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_source_policy import baseline_features, assert_excluded_absent, EXCLUDED_COACH_SP
from gridiron_ml.experiments.nextgen_designs import baseline_formulas


def test_exclusion_applies_to_every_design_and_cannot_return_as_metadata_or_formula():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/experiments/nextgen_fingerprints_v1.json').read_text())
    original=json.loads((root/config['source_f6_manifest']).read_text())['feature_names']
    retained=baseline_features(original,config)
    assert len(retained)==225 and set(original)-set(retained)==EXCLUDED_COACH_SP
    for design in 'abc':
        formulas=baseline_formulas(retained,design)
        assert not {i for f in formulas for i in f.inputs}&EXCLUDED_COACH_SP
    with pytest.raises(ValueError,match='remain'):
        assert_excluded_absent(pd.DataFrame(columns=list(EXCLUDED_COACH_SP)))
    with pytest.raises(ValueError,match='remain'):
        assert_excluded_absent(pd.DataFrame(),[{'name':'renamed','source_inputs':list(EXCLUDED_COACH_SP)}])
    with pytest.raises(ValueError,match='policy required'):
        baseline_features(original,{})
