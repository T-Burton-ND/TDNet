import json

import pandas as pd

from gridiron_ml.pipeline.contracts.artifacts import (
    CANONICAL_FINGERPRINT_FILENAME,
    cleanup_fingerprint_artifacts,
    legacy_fingerprints_path,
    legacy_labels_path,
    metadata_path,
    team_week_fingerprints_path,
    team_week_labels_path,
)
from gridiron_ml.pipeline.contracts.columns import (
    REQUIRED_FINGERPRINT_COLUMNS,
    REQUIRED_LABEL_COLUMNS,
    REQUIRED_PREDICTION_ROW_COLUMNS,
    REQUIRED_TEAM_GAME_COLUMNS,
)
from gridiron_ml.pipeline.contracts.features import (
    DEFAULT_TRAINING_TARGET,
    SAME_WEEK_TARGET,
    blocked_coach_columns,
    blocked_training_columns,
    is_market_column,
)
from gridiron_ml.pipeline.contracts.metadata import (
    ARTIFACT_KIND_HISTORICAL_FINGERPRINT,
    MARKET_POLICY_EVALUATION_ONLY,
    METADATA_DEFAULT_TARGET,
    METADATA_MARKET_POLICY,
    METADATA_ROW_SEMANTICS,
    ROW_SEMANTICS_STATE_AFTER_WEEK,
)
from gridiron_ml.fingerprints.builders.v0 import V0FingerprintBuilder
from gridiron_ml.pipeline.schemas import validate_fingerprint_frame, validate_label_frame, validate_prediction_rows, validate_team_game_table


def test_feature_contract_constants_match_current_values():
    assert DEFAULT_TRAINING_TARGET == "y_next_margin"
    assert SAME_WEEK_TARGET == "y_margin_this_week"
    assert is_market_column("market_spread_close")
    assert is_market_column("diff_market_spread_close")
    assert not is_market_column("offense_ppa")
    assert blocked_coach_columns(
        [
            "coach_career_seasons",
            "coach_season_wins",
            "coach_career_mean_postseason_rank_points",
        ]
    ) == ["coach_season_wins", "coach_career_mean_postseason_rank_points"]
    assert blocked_training_columns(["offense_ppa", "market_spread_close", "coach_season_games"]) == [
        "market_spread_close",
        "coach_season_games",
    ]


def test_required_schema_contracts_are_validator_inputs():
    team_game = pd.DataFrame(
        {
            col: [1 if col != "keys_team" and col != "keys_opponent" else "Team"]
            for col in REQUIRED_TEAM_GAME_COLUMNS
        }
    )
    team_game["game_is_home"] = [True]
    validate_team_game_table(team_game)

    labels = pd.DataFrame(
        {
            "keys_season": [2024],
            "keys_team": ["Team"],
            "keys_week": [0],
            "y_margin_this_week": [pd.NA],
            "y_next_margin": [7.0],
            "y_has_next_game": [True],
        }
    )
    validate_label_frame(labels.loc[:, REQUIRED_LABEL_COLUMNS])
    validate_fingerprint_frame(labels.loc[:, REQUIRED_FINGERPRINT_COLUMNS])

    prediction = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_id": [10],
            "home_team": ["Home"],
            "away_team": ["Away"],
        }
    )
    validate_prediction_rows(prediction.loc[:, REQUIRED_PREDICTION_ROW_COLUMNS])


def test_v0_metadata_sidecar_uses_central_keys_and_values(tmp_path):
    builder = V0FingerprintBuilder(version=0, root=tmp_path, team_game_tables_dir=tmp_path)
    path = tmp_path / "metadata.json"

    builder._write_metadata(path, artifact_kind=ARTIFACT_KIND_HISTORICAL_FINGERPRINT, season=2024)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[METADATA_ROW_SEMANTICS] == ROW_SEMANTICS_STATE_AFTER_WEEK
    assert payload[METADATA_DEFAULT_TARGET] == DEFAULT_TRAINING_TARGET
    assert payload[METADATA_MARKET_POLICY] == MARKET_POLICY_EVALUATION_ONLY
    assert payload["season"] == 2024


def test_fingerprint_rebuild_cleanup_removes_generated_artifacts_not_debug_files(tmp_path):
    fp_dir = tmp_path / "data" / "fingerprints" / "v0"
    debug_dir = fp_dir / "debug"
    debug_dir.mkdir(parents=True)
    season_dir = fp_dir / "2024"
    season_dir.mkdir()

    generated = [
        fp_dir / CANONICAL_FINGERPRINT_FILENAME,
        legacy_fingerprints_path(fp_dir, 0),
        legacy_labels_path(fp_dir, 0),
        metadata_path(fp_dir),
        team_week_fingerprints_path(fp_dir, 2024),
        team_week_labels_path(fp_dir, 2024),
        metadata_path(season_dir),
    ]
    for path in generated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")
    debug_file = debug_dir / "2024_Notre_Dame.csv"
    debug_file.write_text("keep", encoding="utf-8")

    removed = cleanup_fingerprint_artifacts(fp_dir, version=0)

    assert set(removed) == set(generated)
    assert all(not path.exists() for path in generated)
    assert debug_file.exists()
