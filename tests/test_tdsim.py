import numpy as np
import pandas as pd

from gridiron_ml.td_sim import TDSim
from gridiron_ml.td_sim.probability import clip_probabilities, sigmoid_margin_to_prob


class DummyRecursiveModel:
    model_name = "dummy"
    model_family = "dummy"

    def predict(self, X, meta_df=None, market_df=None):
        margin = pd.to_numeric(X.iloc[:, 0], errors="coerce").fillna(0.0).to_numpy(dtype=float) * 10.0
        out = pd.DataFrame(
            {
                "pred_margin": margin,
                "pred_proba_home_win": 1.0 / (1.0 + np.exp(-margin / 10.0)),
                "pred_pick_home": (margin > 0).astype(int),
            }
        )
        if meta_df is not None:
            out = pd.concat([meta_df.reset_index(drop=True), out], axis=1)
        return out

    def total_rank(self, X, meta_df=None):
        score = pd.to_numeric(X.iloc[:, 0], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out = pd.DataFrame({"score": score})
        if meta_df is not None:
            out = pd.concat([meta_df.reset_index(drop=True), out], axis=1)
        return out.sort_values("score", ascending=False).reset_index(drop=True)


def test_probability_helpers_clip_and_flatten_with_random_scaling():
    sharp = sigmoid_margin_to_prob([14.0], scale=7.0, random_scaling_factor=0.5).iloc[0]
    flat = sigmoid_margin_to_prob([14.0], scale=7.0, random_scaling_factor=2.0).iloc[0]
    assert sharp > flat > 0.5
    assert clip_probabilities([-1.0, 0.5, 2.0], 0.05, 0.95).tolist() == [0.05, 0.5, 0.95]


def test_recursive_td_sim_writes_final_outputs(tmp_path):
    sim = TDSim(
        config={
            "simulation": {"n_sims": 3, "random_seed": 3, "model_residual_std": 1.0},
            "recursive": {"performance_sampler": "hybrid", "knn_neighbors": 2, "knn_max_candidates": 10},
            "runtime": {"show_progress": False},
            "outputs": {"top_n_teams": 2},
        },
        schedule=_schedule(),
        fingerprint_frame=_fingerprints(),
        model_specs=[{"name": "dummy", "path": "dummy.pkl", "model": DummyRecursiveModel(), "family": "dummy"}],
    )

    result = sim.run(season=2026, N=3, workflow="single_model", output_dir=tmp_path)

    final_records = result["final_records"]
    average_poll = result["average_poll"]
    assert {"final_records.csv", "average_poll.csv", "top25_projected_records.png"}.issubset(
        {path.name for path in result["saved"]}
    )
    assert set(final_records.loc[final_records["summary_level"] == "aggregate", "team"]) == {"A", "B", "C"}
    assert not average_poll.empty
    assert result["output_dir"].is_relative_to(tmp_path)


def test_recursive_td_sim_multi_model_mode_aggregates_models(tmp_path):
    specs = [
        {"name": "model_a", "path": "a.pkl", "model": DummyRecursiveModel(), "family": "dummy"},
        {"name": "model_b", "path": "b.pkl", "model": DummyRecursiveModel(), "family": "dummy"},
    ]
    sim = TDSim(
        config={
            "simulation": {"n_sims": 2, "random_seed": 5, "model_residual_std": 1.0},
            "recursive": {"performance_sampler": "historical"},
            "runtime": {"show_progress": False},
            "outputs": {"top_n_teams": 2},
        },
        schedule=_schedule(),
        fingerprint_frame=_fingerprints(),
        model_specs=specs,
    )

    result = sim.run(season=2026, N=2, workflow="multi_model", output_dir=tmp_path)
    aggregate = result["final_records"].loc[result["final_records"]["summary_level"] == "aggregate"]

    assert result["models"]["name"].tolist() == ["model_a", "model_b"]
    assert aggregate["models_used"].eq(2).all()


def _schedule():
    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 1, 2],
            "game_id": [1, 2, 3],
            "season_type": ["regular", "regular", "regular"],
            "home_team": ["A", "C", "B"],
            "away_team": ["B", "A", "C"],
            "neutral_site": [False, False, False],
            "conference_game": [False, False, False],
            "venue": [None, None, None],
            "start_date": [None, None, None],
            "home_points": [np.nan, np.nan, np.nan],
            "away_points": [np.nan, np.nan, np.nan],
        }
    )


def _fingerprints():
    rows = []
    for season in [2024, 2025]:
        rows.extend(_season_rows(season=season, teams=["A", "B", "C"], margins=[7.0, -4.0, 10.0]))
    rows.extend(_week0_rows(season=2026, teams=["A", "B", "C"]))
    return pd.DataFrame(rows)


def _week0_rows(season, teams):
    base = {"A": 0.4, "B": 0.1, "C": -0.2}
    rows = []
    for team in teams:
        rows.append(
            {
                "keys_season": season,
                "keys_team": team,
                "keys_week": 0,
                "keys_game_id": pd.NA,
                "keys_opponent": pd.NA,
                "game_is_home": pd.NA,
                "games_played": 0,
                "offense_ppa": base[team],
                "defense_ppa": -base[team],
            }
        )
    return rows


def _season_rows(season, teams, margins):
    rows = _week0_rows(season, teams)
    games = [
        (1, 1, "A", "B", margins[0]),
        (1, 2, "C", "A", margins[1]),
        (2, 3, "B", "C", margins[2]),
    ]
    cumulative = {team: 0.0 for team in teams}
    games_played = {team: 0 for team in teams}
    for week, game_id, home, away, home_margin in games:
        for team, opponent, is_home, team_margin in [
            (home, away, True, home_margin),
            (away, home, False, -home_margin),
        ]:
            games_played[team] += 1
            cumulative[team] += team_margin / 10.0
            avg = cumulative[team] / games_played[team]
            rows.append(
                {
                    "keys_season": season,
                    "keys_team": team,
                    "keys_week": week,
                    "keys_game_id": game_id,
                    "keys_opponent": opponent,
                    "game_is_home": is_home,
                    "games_played": games_played[team],
                    "offense_ppa": avg,
                    "defense_ppa": -avg,
                    "y_margin_this_week": team_margin,
                }
            )
    return rows
