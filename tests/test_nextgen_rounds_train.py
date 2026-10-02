import numpy as np

from gridiron_ml.experiments.nextgen_screening_reduced_parallel import source_to_matchup
from scripts.nextgen_rounds_train import (
    SOURCE_F13, SOURCE_F14, SOURCE_F15, SOURCE_F16,
    _record, stage_sources,
)


def test_rounds_add_one_family_and_keep_market_separate():
    assert stage_sources("F12_original") == ()
    assert stage_sources("F12_corrected") == ()
    assert stage_sources("F13") == SOURCE_F13
    assert stage_sources("F14") == SOURCE_F13 + SOURCE_F14
    assert stage_sources("F15") == SOURCE_F13 + SOURCE_F14 + SOURCE_F15
    assert stage_sources("F16") == SOURCE_F13 + SOURCE_F14 + SOURCE_F15 + SOURCE_F16
    assert stage_sources("F17_market") == stage_sources("F16")


def test_offense_defense_and_situational_role_matchups():
    records = [_record("offense_resid_yards"), _record("defense_resid_yards"),
               _record("early_rush_hhi")]
    raw = np.array([[2.0, 3.0, 0.7, 4.0, 5.0, 0.2]])
    result = source_to_matchup(raw, records)
    np.testing.assert_allclose(result, [[7.0, 7.0, 0.5]])
