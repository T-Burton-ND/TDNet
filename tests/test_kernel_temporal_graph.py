import networkx as nx
import numpy as np
import pandas as pd
import pytest

from gridiron_ml.fingerprints.temporal import build_temporal_fingerprints
from gridiron_ml.graph import build_season_graph, export_season_graph
from gridiron_ml.graph.season import NON_CONFERENCE_EDGE_COLOR, _season_edge_groups, conference_color_map
from gridiron_ml.models import build_model_from_config, validate_model_contract
from gridiron_ml.publication.weekly import select_closest_games


@pytest.mark.parametrize("model_type", ["rbf_kernel_ridge", "rbf_svr", "gaussian_process", "nystroem_ridge"])
def test_kernel_models_follow_contract(model_type, tmp_path):
    x = np.linspace(-2, 2, 36)
    X = pd.DataFrame({"a": x, "b": np.sin(x)})
    y = 4*x + np.sin(3*x)
    params = {"n_components": 12, "gamma": .2, "alpha": 1.0} if model_type == "nystroem_ridge" else {}
    model = build_model_from_config({"family": "kernel", "model_type": model_type,
                                     "max_train_samples": 30, "params": params})
    validate_model_contract(model); model.fit(X, y)
    assert np.isfinite(model.predict_margin(X)).all()
    loaded = type(model).load(model.save(tmp_path / f"{model_type}.pkl"))
    np.testing.assert_allclose(model.predict_margin(X), loaded.predict_margin(X))


def test_temporal_fingerprints_are_shifted_before_smoothing():
    frame = pd.DataFrame({"keys_team": ["A"]*4, "keys_season": [2025]*4,
                          "keys_week": [0,1,2,3], "keys_game_id": [0,1,2,3],
                          "offense_ppa": [10.,20.,30.,999.]})
    out = build_temporal_fingerprints(frame, columns=["offense_ppa"], half_life=1, trend_lag=2)
    assert pd.isna(out.loc[0, "time_adj_lag1__offense_ppa"])
    assert out.loc[3, "time_adj_lag1__offense_ppa"] == 30
    changed = frame.copy(); changed.loc[3, "offense_ppa"] = -999
    out2 = build_temporal_fingerprints(changed, columns=["offense_ppa"], half_life=1, trend_lag=2)
    assert out.loc[3, "time_adj_ewm__offense_ppa"] == out2.loc[3, "time_adj_ewm__offense_ppa"]


@pytest.mark.parametrize("model_type", ["decay_ridge", "trend_elastic_net", "temporal_random_forest", "temporal_hist_gradient_boosted"])
def test_temporal_model_contract(model_type):
    X = pd.DataFrame({"time_adj_ewm__a": np.arange(30.), "time_adj_trend3__a": np.sin(np.arange(30.))})
    y = X.iloc[:, 0] / 3
    params = {"alpha": 1.0} if model_type == "decay_ridge" else (
        {"alpha": .01, "l1_ratio": .25, "max_iter": 1000} if model_type == "trend_elastic_net" else
        {"n_estimators": 8, "n_jobs": 1} if model_type == "temporal_random_forest" else
        {"max_iter": 8, "min_samples_leaf": 3})
    model = build_model_from_config({"family": "temporal", "model_type": model_type, "params": params})
    validate_model_contract(model); model.fit(X, y)
    assert np.isfinite(model.predict_margin(X)).all()


def test_temporal_columns_pass_through_unit_matchups():
    from gridiron_ml.td_run.matchups import MatchupBuilder
    home = pd.DataFrame({"time_adj_ewm__offense_ppa": [1.0]})
    away = pd.DataFrame({"time_adj_ewm__offense_ppa": [0.4]})
    result = MatchupBuilder(representation="unit_matchup").build_many(home, away)
    assert "net_time_adj_ewm__offense_ppa" in result.columns


def test_tdgraph_exports_nodes_edges_and_graphml(tmp_path):
    games = pd.DataFrame([{"id": 1, "season": 2025, "week": 1, "start_date": "2025-08-01",
        "completed": True, "neutral_site": False, "conference_game": True, "venue": "X",
        "home_id": 1, "home_team": "A", "home_classification": "fbs", "home_conference": "C1", "home_points": 21,
        "away_id": 2, "away_team": "B", "away_classification": "fbs", "away_conference": "C2", "away_points": 14}])
    graph = build_season_graph(games, season=2025)
    assert isinstance(graph, nx.MultiDiGraph) and graph.nodes["A"]["wins"] == 1
    meta = export_season_graph(graph, tmp_path)
    assert meta["nodes"] == 2 and (tmp_path / "season.graphml").exists()


def test_tdgraph_separates_conference_and_nonconference_edges():
    games = pd.DataFrame([
        {"id": 1, "season": 2025, "week": 1, "start_date": "2025-08-01",
         "completed": True, "neutral_site": False, "conference_game": True, "venue": "X",
         "home_id": 1, "home_team": "A", "home_classification": "fbs", "home_conference": "C1", "home_points": 21,
         "away_id": 2, "away_team": "B", "away_classification": "fbs", "away_conference": "C1", "away_points": 14},
        {"id": 2, "season": 2025, "week": 1, "start_date": "2025-08-01",
         "completed": True, "neutral_site": False, "conference_game": False, "venue": "Y",
         "home_id": 3, "home_team": "C", "home_classification": "fbs", "home_conference": "C2", "home_points": 17,
         "away_id": 4, "away_team": "D", "away_classification": "fbs", "away_conference": "C3", "away_points": 10},
    ])
    graph = build_season_graph(games, season=2025)
    conference_edges, non_conference_edges = _season_edge_groups(graph, graph)
    assert conference_edges == {"C1": [("B", "A")]}
    assert non_conference_edges == [("D", "C")]
    assert conference_color_map(["SEC"])["SEC"] != NON_CONFERENCE_EDGE_COLOR


def test_select_closest_games_uses_absolute_predicted_margin():
    games = pd.DataFrame({
        "game_id": [1, 2, 3],
        "game_start_time_utc": ["2025-09-01T00:00:00Z"] * 3,
        "away_team": ["A", "C", "E"],
        "home_team": ["B", "D", "F"],
        "predicted_margin": [17.0, 1.5, 3.0],
        "pred_winner": ["B", "C", "F"],
        "model_agreement": [1.0, 0.6, 0.8],
        "pred_home_win_probability": [0.9, 0.49, 0.55],
    })
    closest = select_closest_games(games, count=2)
    assert closest["game_id"].tolist() == [2, 3]
