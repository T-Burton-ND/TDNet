from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.fingerprints.ladder import (
    build_schedule_graph_features,
    build_temporal_state_features,
)
from gridiron_ml.publication.protocol import validate_feature_frame


ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> pd.DataFrame:
    rows = []
    game_specs = {
        1: ("g1", "A", "B", 7.0, 1.0),
        2: ("g2", "B", "A", -3.0, 3.0),
        3: ("g3", "A", "B", 10.0, 5.0),
    }
    histories = {"A": [], "B": []}
    for week, (game_id, first, second, first_margin, first_value) in game_specs.items():
        values = {first: first_value, second: first_value + 1.0}
        margins = {first: first_margin, second: -first_margin}
        for team, opponent in ((first, second), (second, first)):
            histories[team].append(values[team])
            rows.append(
                {
                    "keys_season": 2024,
                    "keys_week": week,
                    "keys_game_id": game_id,
                    "keys_team": team,
                    "keys_opponent": opponent,
                    "next_opponent": opponent,
                    "games_played": len(histories[team]),
                    "offense_ppa": np.mean(histories[team]),
                    "y_margin_this_week": margins[team],
                    "market_spread_close": -2.5,
                }
            )
    return pd.DataFrame(rows)


def test_temporal_features_recover_game_contributions_exactly():
    enriched = build_temporal_state_features(_fixture(), columns=["offense_ppa"])
    team_a = enriched.loc[enriched["keys_team"].eq("A")].sort_values("keys_week")
    assert team_a["time_adj_last1__offense_ppa"].tolist() == [1.0, 4.0, 5.0]
    assert team_a["time_adj_last3__offense_ppa"].tolist() == [1.0, 2.5, 10.0 / 3.0]


def test_future_source_change_does_not_change_earlier_temporal_features():
    original = _fixture()
    changed = original.copy()
    future = changed["keys_week"].eq(3) & changed["keys_team"].eq("A")
    changed.loc[future, "offense_ppa"] = 1000.0
    columns = [column for column in build_temporal_state_features(original).columns if column.startswith("time_adj_")]
    left = build_temporal_state_features(original).loc[lambda x: x["keys_week"] < 3, columns]
    right = build_temporal_state_features(changed).loc[lambda x: x["keys_week"] < 3, columns]
    pd.testing.assert_frame_equal(left, right)


def test_future_game_change_does_not_change_earlier_graph_features():
    original = _fixture()
    changed = original.copy()
    changed.loc[changed["keys_week"].eq(3), "y_margin_this_week"] *= -1
    columns = [column for column in build_schedule_graph_features(original).columns if column.startswith("graph_")]
    left = build_schedule_graph_features(original).loc[lambda x: x["keys_week"] < 3, columns]
    right = build_schedule_graph_features(changed).loc[lambda x: x["keys_week"] < 3, columns]
    pd.testing.assert_frame_equal(left, right)


def test_realized_ladder_is_distinct_and_market_contract_is_explicit():
    enriched = build_schedule_graph_features(build_temporal_state_features(_fixture()))
    manifests = validate_feature_frame(
        enriched,
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladder_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    assert manifests["F4"]["feature_count"] < manifests["F5"]["feature_count"]
    assert manifests["F5"]["feature_count"] < manifests["F6"]["feature_count"]
    assert manifests["F7"]["feature_names"] == ["market_spread_close"]
    assert set(manifests["F8"]["feature_names"]) == (
        set(manifests["F6"]["feature_names"]) | {"market_spread_close"}
    )
