import json

import pandas as pd

from gridiron_ml.publication.imputation import TemporalDonorImputer


def test_temporal_donors_exclude_future_rows_and_report_identity():
    donors = pd.DataFrame(
        {
            "row_id": ["prior", "future", "same_week"],
            "season": [2025, 2026, 2026],
            "week": [12, 2, 2],
            "distance": [0.0, 0.0, 0.0],
            "value": [4.0, 99.0, 88.0],
        }
    )
    target = pd.DataFrame({"row_id": ["target"], "season": [2026], "week": [2], "distance": [0.0], "value": [None]})
    result, audit = TemporalDonorImputer(("distance",), ("value",), k=3).fit(donors).transform(target)
    assert result.loc[0, "value"] == 4.0
    assert json.loads(audit.loc[0, "donor_ids_json"]) == ["prior"]
    assert bool(audit.loc[0, "fallback_used"]) is False


def test_temporal_donor_fallback_is_training_only_and_finite():
    donors = pd.DataFrame({"row_id": ["d"], "season": [2025], "week": [1], "distance": [1.0], "value": [7.0]})
    target = pd.DataFrame({"row_id": ["target"], "season": [2025], "week": [1], "distance": [1.0], "value": [None]})
    result, audit = TemporalDonorImputer(("distance",), ("value",), k=1).fit(donors).transform(target)
    assert result.loc[0, "value"] == 7.0
    assert bool(audit.loc[0, "fallback_used"]) is True
    assert audit.loc[0, "donor_count"] == 0


def test_donor_distance_order_is_numeric_before_id_tie_break():
    donors = pd.DataFrame(
        {
            "row_id": ["ten", "two"],
            "season": [2024, 2024],
            "week": [1, 1],
            "distance": [10.0, 2.0],
            "value": [10.0, 2.0],
        }
    )
    target = pd.DataFrame({"row_id": ["target"], "season": [2025], "week": [1], "distance": [0.0], "value": [None]})
    _, audit = TemporalDonorImputer(("distance",), ("value",), k=1).fit(donors).transform(target)
    assert json.loads(audit.loc[0, "donor_ids_json"]) == ["two"]
