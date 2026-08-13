import pandas as pd

from gridiron_ml.publication.tables import build_publication_tables


def test_publication_tables_materialize_json_test_seasons_and_market_rows(tmp_path):
    results = pd.DataFrame([
        {
            "status": "success", "objective": "winner", "feature_config": "F6",
            "model_level": "M1", "test_seasons_json": "[2020]",
            "winner_accuracy": 0.6, "brier_score": 0.2, "runtime_seconds": 1.0,
        },
        {
            "status": "success", "objective": "margin", "feature_config": "F7",
            "model_level": "M1", "test_seasons_json": "[2021]",
            "mae": 10.0, "rmse": 12.0, "runtime_seconds": 2.0,
        },
    ])
    tables = build_publication_tables(
        output_root=tmp_path,
        experiment_results=results,
        feature_tiers=pd.DataFrame(),
        model_families=pd.DataFrame(),
    )
    assert tables["table_01_data_summary"].loc[0, "seasons"] == 2
    assert set(tables["table_04_historical_performance"]["test_season"]) == {2020, 2021}
    assert len(tables["table_09_market_incremental_value"]) == 2
