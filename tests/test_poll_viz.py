import pandas as pd

from gridiron_ml.td_run import TDEval
from gridiron_ml.td_run.poll_viz import plot_weekly_top25_table, resolve_team_logo_path, load_team_logo_image


def test_plot_weekly_top25_table_writes_png(tmp_path):
    weekly_poll = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2],
            "rank": [1, 2, 1, 2],
            "keys_team": ["Notre Dame", "Ohio State", "Ohio State", "Notre Dame"],
        }
    )

    path = plot_weekly_top25_table(weekly_poll, tmp_path / "poll.png", top_n=2)

    assert path.exists()


def test_resolve_team_logo_path_uses_normalized_team_slug(tmp_path):
    logo_path = tmp_path / "notre_dame.png"
    logo_path.write_bytes(b"placeholder")

    assert resolve_team_logo_path("Notre Dame", tmp_path) == logo_path


def test_resolve_team_logo_path_handles_ampersand_alias(tmp_path):
    logo_path = tmp_path / "texas_aandm.png"
    logo_path.write_bytes(b"placeholder")

    assert resolve_team_logo_path("Texas A&M", tmp_path) == logo_path


def test_logo_loader_crops_transparent_canvas_to_visible_mark(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np

    image = np.zeros((100, 200, 4), dtype=float)
    image[35:65, 80:120, :3] = [1.0, 0.0, 0.0]
    image[35:65, 80:120, 3] = 1.0
    path = tmp_path / "padded.png"
    plt.imsave(path, image)
    cropped = load_team_logo_image(path)
    assert cropped.shape[:2] == (30, 40)


def test_evaluator_build_weekly_poll_outputs_uses_nested_output_dirs(tmp_path):
    evaluator = TDEval({}, fingerprints=object(), matchup_builder=object(), model=object())

    def fake_poll(models, season, week, average_scope="season", top_n=25):
        evaluator.poll_ballots_ = pd.DataFrame(
            {
                "keys_team": ["A"],
                "ballot_model": ["m"],
                "ballot_rank": [1],
                "poll_points": [25],
                "top25_vote": [True],
                "first_place_vote": [True],
            }
        )
        return pd.DataFrame(
            {
                "rank": [1],
                "keys_team": ["A"],
                "poll_points": [25],
                "ballots_seen": [1],
                "top25_votes": [1],
                "first_place_votes": [1],
                "average_rank": [1.0],
                "best_rank": [1],
                "worst_rank": [1],
            }
        )

    evaluator.poll = fake_poll

    evaluator.build_weekly_poll_outputs(
        models=[object()],
        season=2025,
        weeks=[1],
        top_n=1,
        output_dir=tmp_path,
    )

    assert (tmp_path / "tables" / "weekly_poll_top25.csv").exists()
    assert (tmp_path / "plots" / "weekly_poll_top25_table.png").exists()


def test_build_weekly_poll_outputs_merges_existing_week_history(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 2],
            "rank": [1, 1],
            "keys_team": ["Existing Week 1", "Old Week 2"],
            "poll_points": [25, 25],
        }
    ).to_csv(tables_dir / "weekly_poll_top25.csv", index=False)

    evaluator = TDEval({}, fingerprints=object(), matchup_builder=object(), model=object())
    called_weeks = []

    def fake_poll(models, season, week, average_scope="season", top_n=25):
        called_weeks.append(week)
        evaluator.poll_ballots_ = pd.DataFrame(
            {
                "keys_team": [f"New Week {week}"],
                "ballot_model": ["m"],
                "ballot_rank": [1],
                "poll_points": [25],
                "top25_vote": [True],
                "first_place_vote": [True],
            }
        )
        return pd.DataFrame(
            {
                "rank": [1],
                "keys_team": [f"New Week {week}"],
                "poll_points": [25],
                "ballots_seen": [1],
                "top25_votes": [1],
                "first_place_votes": [1],
                "average_rank": [1.0],
                "best_rank": [1],
                "worst_rank": [1],
            }
        )

    evaluator.poll = fake_poll

    tables = evaluator.build_weekly_poll_outputs(
        models=[object()],
        season=2026,
        weeks=[2],
        top_n=1,
        output_dir=tmp_path,
        merge_existing=True,
    )

    top25 = tables["weekly_poll_top25"]
    assert called_weeks == [2]
    assert top25["keys_team"].tolist() == ["Existing Week 1", "New Week 2"]
    assert pd.read_csv(tables_dir / "weekly_poll_top25.csv")["keys_team"].tolist() == ["Existing Week 1", "New Week 2"]


def test_build_weekly_poll_outputs_ignores_empty_existing_skip_file(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    (tables_dir / "weekly_poll_skipped_weeks.csv").write_text("")

    evaluator = TDEval({}, fingerprints=object(), matchup_builder=object(), model=object())

    def fake_poll(models, season, week, average_scope="season", top_n=25):
        evaluator.poll_ballots_ = pd.DataFrame()
        return pd.DataFrame(
            {
                "rank": [1],
                "keys_team": ["A"],
                "poll_points": [25],
                "ballots_seen": [1],
                "top25_votes": [1],
                "first_place_votes": [1],
                "average_rank": [1.0],
                "best_rank": [1],
                "worst_rank": [1],
            }
        )

    evaluator.poll = fake_poll

    tables = evaluator.build_weekly_poll_outputs(
        models=[object()],
        season=2026,
        weeks=[1],
        top_n=1,
        output_dir=tmp_path,
        merge_existing=True,
    )

    assert tables["weekly_poll_top25"].loc[0, "keys_team"] == "A"
