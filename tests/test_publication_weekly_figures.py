import pandas as pd
from gridiron_ml.publication.polls import load_ap_top25
from gridiron_ml.cli.publication.build_2025_roster_outputs import _write_poll_support

from gridiron_ml.publication import PublicationFigureBuilder
from gridiron_ml.publication.preseason_states import build_preseason_state_frame
from gridiron_ml.publication.poll_recaps import (
    plot_season_podium_gaps,
    plot_season_poll_race,
    season_poll_podium_summary,
)
from gridiron_ml.publication.weekly import (
    _build_schedule_driven_matchups,
    frozen_model_set_sha256,
    format_eastern_kickoffs,
    format_vegas_spread,
    plot_all_games_table,
    plot_top25_matchups,
    summarize_weekly_predictions,
)
from gridiron_ml.publication.recaps import plot_sunday_recap_table
from gridiron_ml.publication.polls import add_team_records
from gridiron_ml.publication.team_labels import format_team_with_ap_rank
from gridiron_ml.td_run.matchups import MatchupBuilder


def test_poll_support_drops_ensemble_exposed_disabled_ballot(tmp_path, monkeypatch):
    ballots = pd.DataFrame(
        [
            {
                "ballot_model": model,
                "keys_team": team,
                "ballot_rank": rank,
                "poll_points": 26 - rank if rank <= 25 else 0,
                "top25_vote": rank <= 25,
                "first_place_vote": rank == 1,
            }
            for model, teams in {
                "enabled": ["Ohio State", "Air Force"],
                "disabled_internal": ["Air Force", "Ohio State"],
            }.items()
            for rank, team in enumerate(teams, start=1)
        ]
    )
    monkeypatch.setattr(
        "gridiron_ml.cli.publication.build_2025_roster_outputs.plot_consensus_poll_table",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "gridiron_ml.cli.publication.build_2025_roster_outputs.plot_ballot_logo_grid",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "gridiron_ml.cli.publication.build_2025_roster_outputs.plot_model_disagreement",
        lambda *args, **kwargs: None,
    )
    poll = _write_poll_support(
        {"poll": pd.DataFrame(), "ballots": ballots, "failures": pd.DataFrame()},
        tmp_path,
        season=2025,
        week=15,
        objective="margin",
        ap=pd.DataFrame(),
        logo_dir=tmp_path / "logos",
        model_order=["ENABLED"],
    )
    assert poll.iloc[0]["keys_team"] == "Ohio State"
    assert poll.iloc[0]["season"] == 2025
    assert poll.iloc[0]["week"] == 15
    assert set(pd.read_csv(tmp_path / "tdnet_model_ballots.csv")["ballot_model"]) == {"enabled"}


def test_weekly_figures_render_with_missing_logos(tmp_path):
    games = pd.DataFrame([{
        "game_id": 1, "game_start_time_utc": "2026-08-29T16:00:00Z", "away_team": "Away",
        "home_team": "Home", "pred_winner": "Home", "predicted_margin": 4.2,
        "pred_home_win_probability": 0.64, "model_agreement": 0.8, "neutral_site": False,
        "away_rank": 10, "home_rank": 2,
    }])
    model_set = frozen_model_set_sha256(["a" * 64, "b" * 64])
    top25 = plot_top25_matchups(games, tmp_path / "top25.png", logo_dir=tmp_path / "logos", title="Test", model_set_sha256=model_set, checkpoint_count=2)
    all_games = plot_all_games_table(games, tmp_path / "all.png", title="Test", model_set_sha256=model_set, checkpoint_count=2)
    assert top25.exists() and not top25.with_suffix(".svg").exists()
    assert all_games.exists() and not all_games.with_suffix(".svg").exists()


def test_public_pick_kickoffs_are_rendered_in_eastern_time():
    games = pd.DataFrame([{
        "game_start_time_utc": "2026-08-29T16:00:00Z", "away_team": "Away",
        "home_team": "Home", "pred_winner": "Home", "predicted_margin": 4.0,
        "pred_home_win_probability": .6, "model_agreement": .8,
    }])
    assert format_eastern_kickoffs(games["game_start_time_utc"]).iloc[0] == "Sat 12:00 PM"


def test_public_pick_spread_names_favorite_at_publication():
    assert format_vegas_spread({
        "home_team": "Home", "away_team": "Away", "market_spread_close": -3.5,
    }) == "Home −3.5"
    assert format_vegas_spread({
        "home_team": "Home", "away_team": "Away", "market_spread_close": 2.5,
    }) == "Away −2.5"
    assert format_vegas_spread({
        "home_team": "Home", "away_team": "Away", "market_spread_close": 0,
    }) == "Pick'em"
    assert format_vegas_spread({
        "home_team": "Home", "away_team": "Away", "market_spread_close": None,
    }) == "Not available"
    assert format_vegas_spread({
        "home_team": "Home", "away_team": "Away", "market_spread_close": -3.25,
    }) == "Home −3.25"


def test_team_label_prefers_canonical_ap_rank_columns():
    game = pd.Series({
        "away_team": "Away", "home_team": "Home",
        "ap_rank_away": 7, "ap_rank_home": pd.NA,
        "away_rank": 12, "home_rank": pd.NA,
    })
    assert format_team_with_ap_rank(game, "away") == "#7 Away"
    assert format_team_with_ap_rank(game, "home") == "Home"


def test_add_team_records_uses_results_through_completed_week():
    poll = pd.DataFrame({"keys_team": ["A", "B", "C"]})
    games = pd.DataFrame([
        {"season_type": "regular", "week": 1, "home_team": "A", "away_team": "B", "home_points": 24, "away_points": 17},
        {"season_type": "regular", "week": 2, "home_team": "C", "away_team": "A", "home_points": 10, "away_points": 10},
        {"season_type": "regular", "week": 3, "home_team": "B", "away_team": "C", "home_points": 21, "away_points": 7},
        {"season_type": "postseason", "week": 1, "home_team": "B", "away_team": "A", "home_points": 35, "away_points": 7},
    ])
    records = add_team_records(poll, games, completed_week=2).set_index("keys_team")
    assert records.loc["A", "record"] == "1–0–1"
    assert records.loc["B", "record"] == "0–1"
    assert records.loc["C", "record"] == "0–0–1"
    assert records["record_through_week"].eq(2).all()


def test_margin_poll_story_figures_render_full_weekly_gap(tmp_path):
    poll = pd.DataFrame([
        {
            "week": week,
            "rank": rank,
            "keys_team": team,
            "poll_points": points - week * rank,
            "ballots_seen": 3,
        }
        for week in range(3)
        for rank, team, points in [
            (1, "Ohio State", 75),
            (2, "Oregon", 65),
            (3, "Georgia", 55),
            (4, "Indiana", 45),
            (5, "Notre Dame", 35),
        ]
    ])
    summary = season_poll_podium_summary(poll)
    assert summary["leader_gap"].tolist() == [10, 11, 12]
    race = plot_season_poll_race(
        poll, tmp_path / "race.png", season=2025, objective="margin", dpi=60,
    )
    podium = plot_season_podium_gaps(
        poll, tmp_path / "podium.png", season=2025, objective="margin",
        logo_dir=tmp_path / "logos", dpi=60,
    )
    for output in (race, podium):
        assert output.exists()
        assert not output.with_suffix(".svg").exists()


def test_sunday_scorecard_renders_ap_ranks_next_to_team_names(tmp_path):
    games = pd.DataFrame([{
        "away_team": "Away", "home_team": "Home", "ap_rank_away": 10, "ap_rank_home": 2,
        "away_points": 20, "home_points": 27, "projected_away_points": 21.0,
        "projected_home_points": 28.0, "pred_winner": "Home", "pred_home_margin": 7.0,
        "su_correct": True, "market_spread_close": -3.5, "ats_pick": "Home",
        "ats_result": "win", "margin_absolute_error": 0.0,
        "pred_home_win_probability": 0.7, "actual_home_win": 1.0, "model_count": 2,
    }])
    output = plot_sunday_recap_table(
        games, tmp_path / "scorecard.png", season=2025, week=1, objective="margin",
        season_to_date_games=games,
    )
    assert output.exists()
    assert not output.with_suffix(".svg").exists()


def test_complete_publication_figure_suite_renders(tmp_path):
    rows = []
    for feature in ["F1", "F6", "F7", "F8"]:
        for model in ["M1", "M4"]:
            for seed in [1, 2]:
                rows.append({
                    "objective": "winner", "feature_config": feature, "model_level": model,
                    "seed": seed, "test_season": 2024 + seed, "brier_score": 0.20 + 0.01 * seed,
                    "winner_accuracy": 0.70, "mae": 10.0, "runtime_seconds": 1.0 + seed,
                })
                rows.append({
                    "objective": "margin", "feature_config": feature, "model_level": model,
                    "seed": seed, "test_season": 2024 + seed, "brier_score": 0.22,
                    "winner_accuracy": 0.68, "mae": 9.0 + seed, "runtime_seconds": 1.0 + seed,
                })
    matrix = pd.DataFrame(rows)
    predictions = pd.DataFrame([
        {
            "game_id": game, "season": 2026, "week": 1 + game // 10, "model_name": model,
            "pred_home_win_probability": 0.65 if model == "A" else 0.45,
            "actual_home_win": game % 2 == 0, "pred_winner": "Home" if model == "A" else "Away",
            "home_team": "Home", "away_team": "Away",
        }
        for game in range(30) for model in ["A", "B"]
    ])
    learning = pd.DataFrame([
        {"model_name": "MLP", "epoch": epoch, "train_loss": 1 / (epoch + 1), "validation_loss": 1.1 / (epoch + 1)}
        for epoch in range(5)
    ])
    ablations = pd.DataFrame({"feature_family": ["box", "efficiency"], "delta": [0.01, -0.02]})
    importance = pd.DataFrame({"feature_family": ["box", "box", "efficiency", "efficiency"], "importance": [0.2, 0.3, 0.1, 0.15]})
    controls = pd.DataFrame({"control": ["real", "shuffle", "random"], "value": [0.2, 0.25, 0.26]})
    manifest = PublicationFigureBuilder(tmp_path, dpi=60, strict=True).generate_all(
        matrix_summary=matrix, predictions=predictions, learning_curves=learning,
        ablations=ablations, importance_stability=importance, negative_controls=controls,
    )
    assert manifest["generated_count"] == 15
    assert not manifest["skipped"]


def test_preseason_state_carries_features_but_preserves_schedule_fields():
    frame = pd.DataFrame([
        {"keys_season": 2025, "keys_week": 15, "keys_team": "A", "statOff_score": 7.0, "next_game_id": 1, "next_opponent": "Old"},
        {"keys_season": 2026, "keys_week": 0, "keys_team": "A", "statOff_score": 0.0, "next_game_id": 2, "next_opponent": "New"},
    ])
    state = build_preseason_state_frame(frame, season=2026)
    assert state.loc[0, "statOff_score"] == 7.0
    assert state.loc[0, "next_game_id"] == 2
    assert state.loc[0, "next_opponent"] == "New"
    assert bool(state.loc[0, "preseason_prior_applied"])


def test_weekly_consensus_preserves_market_spread_for_social_graphics():
    rows = pd.DataFrame([
        {
            "game_id": 1, "season": 2026, "week": 1,
            "game_start_time_utc": "2026-08-29T19:00:00Z",
            "home_team": "Home", "away_team": "Away", "neutral_site": False,
            "conference_game": False, "season_type": "regular", "model_name": model,
            "pred_home_margin": margin, "pred_home_win_probability": probability,
            "pred_winner": "Home", "market_spread_close": -3.5,
        }
        for model, margin, probability in (("a", 4.0, .58), ("b", 6.0, .62))
    ])
    consensus = summarize_weekly_predictions(rows)
    assert consensus.loc[0, "market_spread_close"] == -3.5
    assert consensus.loc[0, "vegas_spread_as_of_publish"] == -3.5


def test_cfbd_provider_average_lines_merge_without_fabricating_empty_games():
    from gridiron_ml.publication.weekly import merge_cfbd_market_lines

    schedule = pd.DataFrame([
        {"id": 1, "season": 2026, "week": 1},
        {"id": 2, "season": 2026, "week": 1},
    ])
    lines = pd.DataFrame([
        {"id": 1, "season": 2026, "week": 1,
         "lines": [
             {"provider": "A", "spread": -7},
             {"provider": "B", "spread": -8},
             {"provider": "C", "spread": -12},
         ]},
        {"id": 2, "season": 2026, "week": 1, "lines": []},
    ])
    merged = merge_cfbd_market_lines(schedule, lines, season=2026, week=1)
    assert merged.loc[merged["game_id"].eq(1), "market_spread_close"].item() == -9.0
    assert pd.isna(merged.loc[merged["game_id"].eq(2), "market_spread_close"].item())


def test_preseason_new_team_uses_owner_approved_conference_mean(tmp_path):
    membership = tmp_path / "data/raw/cfbd/v2/teams_fbs"
    membership.mkdir(parents=True)
    pd.DataFrame({"school": ["Peer", "New"], "conference": ["League", "League"]}).to_parquet(
        membership / "2026.parquet", index=False
    )
    frame = pd.DataFrame([
        {"keys_season": 2025, "keys_week": 15, "keys_team": "Peer", "statOff_score": 14.0},
        {"keys_season": 2026, "keys_week": 0, "keys_team": "Peer", "statOff_score": 0.0},
        {"keys_season": 2026, "keys_week": 0, "keys_team": "New", "statOff_score": 0.0},
    ])
    state = build_preseason_state_frame(frame, season=2026, project_root=tmp_path)
    newcomer = state[state["keys_team"].eq("New")].iloc[0]
    assert newcomer["statOff_score"] == 14.0
    assert newcomer["preseason_prior_method"] == "conference_mean"


def test_preseason_state_overlays_only_registry_approved_current_values(tmp_path):
    registry = tmp_path / "configs/features"
    registry.mkdir(parents=True)
    (registry / "feature_registry.yaml").write_text(
        """features:\n  roster_talent:\n    allowed_preseason: true\npatterns:\n  coach_career_*:\n    allowed_preseason: true\n""",
        encoding="utf-8",
    )
    frame = pd.DataFrame([
        {
            "keys_season": 2025, "keys_week": 15, "keys_team": "A",
            "roster_talent": 700.0, "coach_career_seasons": 4.0, "statOff_score": 21.0,
        },
        {
            "keys_season": 2026, "keys_week": 0, "keys_team": "A",
            "roster_talent": None, "coach_career_seasons": 9.0, "statOff_score": 0.0,
        },
    ])
    state = build_preseason_state_frame(frame, season=2026, project_root=tmp_path)
    assert state.loc[0, "roster_talent"] == 700.0
    assert state.loc[0, "coach_career_seasons"] == 9.0
    assert state.loc[0, "statOff_score"] == 21.0
    assert state.loc[0, "preseason_current_overlay_count"] == 1


def test_preseason_state_overlays_live_raw_returning_context(tmp_path):
    raw = tmp_path / "data/raw/cfbd/v2/returning"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {"season": [2026], "team": ["A"], "percent_p_p_a": [0.88]}
    ).to_parquet(raw / "2026.parquet", index=False)
    frame = pd.DataFrame([
        {
            "keys_season": 2025, "keys_week": 15, "keys_team": "A",
            "roster_return_percent_p_p_a": 0.50,
        },
        {
            "keys_season": 2026, "keys_week": 0, "keys_team": "A",
            "roster_return_percent_p_p_a": None,
        },
    ])
    state = build_preseason_state_frame(frame, season=2026, project_root=tmp_path)
    assert state.loc[0, "roster_return_percent_p_p_a"] == 0.88
    assert state.loc[0, "preseason_current_overlay_count"] == 1


def test_schedule_driven_matchups_keep_two_games_for_same_team():
    class Snapshot:
        def season_snapshot(self, season, week):
            features = pd.DataFrame({"rating": [3.0, 2.0, 1.0], "game_is_home": [0.0, 0.0, 0.0]})
            meta = pd.DataFrame({"keys_team": ["A", "B", "C"]})
            return features, meta, pd.DataFrame(index=meta.index)

    schedule = pd.DataFrame([
        {"game_id": 1, "season": 2026, "week": 1, "game_start_time_utc": "2026-08-29", "home_team": "A", "away_team": "B", "neutral_site": False, "conference_game": False, "season_type": "regular"},
        {"game_id": 2, "season": 2026, "week": 1, "game_start_time_utc": "2026-09-05", "home_team": "C", "away_team": "A", "neutral_site": False, "conference_game": False, "season_type": "regular"},
    ])
    matchups, context = _build_schedule_driven_matchups(
        Snapshot(), schedule, MatchupBuilder(representation="diff"), season=2026, week=1
    )
    assert len(matchups) == 2
    assert context["game_id"].tolist() == [1, 2]
    assert matchups["rating_diff"].tolist() == [1.0, -2.0]


def test_load_nested_cfbd_ap_poll(tmp_path):
    source = tmp_path / "rankings.parquet"
    pd.DataFrame([{
        "season": 2026, "season_type": "regular", "week": 3,
        "polls": [{"poll": "Coaches Poll", "ranks": [{"rank": 1, "school": "Wrong"}]},
                  {"poll": "AP Top 25", "ranks": [{"rank": i, "school": f"AP {i}"} for i in range(1, 26)]}],
    }]).to_parquet(source, index=False)
    poll = load_ap_top25(source, season=2026, week=3)
    assert len(poll) == 25
    assert poll.iloc[0]["team"] == "AP 1"
    assert "Wrong" not in set(poll["team"])
