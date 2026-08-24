from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pytest

from gridiron_ml.publication.social_top10 import (
    SOCIAL_STYLE,
    SocialTeam,
    _context_lines,
    _display_name,
    _human_generation_date,
    _rank_label,
    low_resolution_logo_teams,
    logo_needs_light_plate,
    noteworthy_badge,
    prepare_top10,
    require_high_resolution_logos,
    render_top10_social,
    validate_no_collision,
)
from gridiron_ml.publication.weekly import load_top25
from gridiron_ml.publication import poll_recaps


def _poll():
    names = [
        "Ohio State", "Oregon", "Georgia", "Texas Tech", "Indiana", "Miami",
        "Notre Dame", "Texas A&M", "Ole Miss", "A Very Long University Team Name",
    ]
    return pd.DataFrame(
        {
            "season": 2026,
            "week": 0,
            "rank": range(1, 11),
            "keys_team": names,
            "reference_rank": [1, 2, 3, 12, 6, 7, 4, 8, 9, 13],
            "poll_points": [804, 760, 712, 589, 565, 557, 555, 535, 535, 493],
            "first_place_votes": [24, 3, 3, 1, 0, 0, 1, 0, 0, 0],
            "ballots_seen": [33] * 10,
            "top25_votes": [33, 33, 33, 29, 29, 32, 31, 33, 31, 32],
            "best_rank": [1, 1, 1, 1, 3, 4, 1, 4, 5, 3],
        }
    )


def _logos(directory, poll, *, size=(500, 500)):
    directory.mkdir()
    for team in poll["keys_team"]:
        path = directory / (team.lower().replace("&", "and").replace(" ", "_") + ".png")
        logo = Image.new("RGBA", size, (0, 0, 0, 0))
        inset = max(5, min(size) // 10)
        ImageDraw.Draw(logo).ellipse(
            (inset, inset, size[0] - inset, size[1] - inset),
            fill=(255, 95, 162, 255),
        )
        logo.save(path)


def test_prepare_top10_requires_unique_complete_ranks():
    teams = prepare_top10(_poll())
    assert [team.rank for team in teams] == list(range(1, 11))
    assert len({team.team for team in teams}) == 10
    assert teams[0].points == 804
    assert teams[0].first_place_votes == 24
    assert teams[0].ballots_seen == 33
    assert teams[0].top25_votes == 33
    assert teams[0].best_rank == 1
    assert teams[7].social_rank == 8 and teams[7].tied
    assert teams[8].social_rank == 8 and teams[8].tied
    assert teams[9].social_rank == 10 and not teams[9].tied
    assert _rank_label(teams[7]) == "T-8"
    assert _rank_label(teams[8]) == "T-8"
    assert _rank_label(teams[9]) == "#10"
    assert _display_name(teams[3]) == "Texas Tech"
    invalid = _poll()
    invalid.loc[9, "rank"] = 9
    with pytest.raises(ValueError, match="unique ranks 1 through 10"):
        prepare_top10(invalid)


def test_badges_only_show_noteworthy_reference_gaps():
    assert noteworthy_badge(SocialTeam(4, "Texas Tech", 12)) == "↑8 vs AP"
    assert noteworthy_badge(SocialTeam(7, "Notre Dame", 4)) is None
    assert noteworthy_badge(SocialTeam(10, "Example", 5)) == "↓5 vs AP"
    assert noteworthy_badge(SocialTeam(10, "Example", 6)) is None


def test_compact_fpv_label_and_landscape_column_span():
    assert _context_lines(SocialTeam(2, "Oregon", points=760, first_place_votes=3)) == [
        "760 PTS", "3 FPV",
    ]
    assert _context_lines(
        SocialTeam(8, "Texas A&M", points=535, ballots_seen=33, top25_votes=33, best_rank=4)
    ) == ["535 PTS", "BEST: #4"]
    assert _context_lines(
        SocialTeam(5, "Indiana", points=565, ballots_seen=33, top25_votes=29, best_rank=3)
    ) == [
        "565 PTS", "ON 29/33 BALLOTS",
    ]
    assert _context_lines(
        SocialTeam(5, "Indiana", points=565, ballots_seen=33, top25_votes=30, best_rank=3)
    ) == ["565 PTS"]
    assert _context_lines(
        SocialTeam(6, "Miami", points=557, ballots_seen=33, top25_votes=32, best_rank=4)
    ) == ["557 PTS"]
    geometry = SOCIAL_STYLE["landscape_rank_column"]
    total_height = geometry["bottom"] - geometry["top"]
    panel_height = (total_height - geometry["gap"] * 6) // 7
    assert panel_height * 7 + geometry["gap"] * 6 == 550
    assert _human_generation_date("2026-08-24T12:34:56Z") == "AUG 24, 2026"


def test_collision_guard_rejects_overlap_and_crowding():
    validate_no_collision("rank", (0, 0, 40, 40), "logo", (55, 0, 95, 40), gap=10)
    with pytest.raises(ValueError, match="layout collision"):
        validate_no_collision("rank", (0, 0, 40, 40), "logo", (45, 0, 85, 40), gap=10)


def test_logo_resolution_audit_flags_small_sources(tmp_path):
    poll = _poll()
    logos = tmp_path / "logos"
    _logos(logos, poll, size=(120, 80))
    assert low_resolution_logo_teams(poll, logos, minimum_px=256) == poll["keys_team"].tolist()
    assert low_resolution_logo_teams(poll, logos, minimum_px=80) == []


def test_social_render_contract_rejects_undersized_resolved_logos(tmp_path):
    poll = _poll()
    logos = tmp_path / "logos"
    _logos(logos, poll, size=(499, 499))
    with pytest.raises(ValueError, match="require 500px logo sources"):
        require_high_resolution_logos(poll, logos)


def test_dark_logo_detection_selects_a_light_plate():
    dark = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(dark).rectangle((10, 10, 90, 90), fill=(10, 20, 35, 255))
    light = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(light).rectangle((10, 10, 90, 90), fill=(245, 245, 245, 255))
    assert logo_needs_light_plate(np.asarray(dark))
    assert not logo_needs_light_plate(np.asarray(light))


def test_weekly_top25_loader_preserves_social_point_totals(tmp_path):
    path = tmp_path / "poll.csv"
    _poll().to_csv(path, index=False)
    loaded = load_top25(path, season=2026, week=0)
    assert loaded.loc[0, "poll_points"] == 804
    assert loaded.loc[0, "first_place_votes"] == 24
    assert loaded.loc[0, "ballots_seen"] == 33
    assert loaded.loc[0, "top25_votes"] == 33
    assert loaded.loc[0, "best_rank"] == 1
    assert loaded.loc[3, "reference_rank"] == 12


def test_social_variants_have_exact_dimensions_and_are_deterministic(tmp_path):
    poll = _poll()
    logos = tmp_path / "logos"
    _logos(logos, poll)
    hashes = []
    for variant, dimensions in SOCIAL_STYLE["canvases"].items():
        path = render_top10_social(
            poll,
            tmp_path / f"top10_{variant}.png",
            season=2026,
            week=0,
            logo_dir=logos,
            variant=variant,
            generated_at_utc="2026-08-24T12:00:00Z",
            git_commit="a" * 40,
        )
        assert path.exists() and path.stat().st_size > 10_000
        with Image.open(path) as image:
            assert image.size == dimensions
        first_hash = sha256(path.read_bytes()).hexdigest()
        render_top10_social(
            poll, path, season=2026, week=0, logo_dir=logos, variant=variant,
            generated_at_utc="2026-08-24T12:00:00Z", git_commit="a" * 40,
        )
        assert sha256(path.read_bytes()).hexdigest() == first_hash
        hashes.append(first_hash)
    assert hashes[0] != hashes[1]


def test_sunday_margin_poll_recap_emits_both_social_variants(tmp_path, monkeypatch):
    tables = tmp_path / "tables"
    tables.mkdir()
    poll = _poll().rename(columns={"reference_rank": "ap_rank"})
    poll.to_csv(tables / "weekly_poll_top25.csv", index=False)
    pd.DataFrame(
        {
            "week": [0] * 10,
            "ballot_model": ["model_a"] * 10,
            "keys_team": poll["keys_team"],
            "ballot_rank": range(1, 11),
            "poll_points": range(25, 15, -1),
            "top25_vote": [True] * 10,
        }
    ).to_csv(tables / "weekly_poll_ballots.csv", index=False)

    for name in (
        "plot_consensus_poll_table",
        "plot_weekly_top25_table",
        "plot_ballot_logo_grid",
        "plot_model_disagreement",
        "plot_full_season_poll_grid",
    ):
        monkeypatch.setattr(poll_recaps, name, lambda *args, **kwargs: Path(args[1]))

    rendered = []

    def fake_social(frame, path, **kwargs):
        path = Path(path)
        path.write_bytes(b"social")
        rendered.append((path.name, kwargs["variant"], kwargs["season"], kwargs["week"]))
        return path

    monkeypatch.setattr(poll_recaps, "render_top10_social", fake_social)
    output = tmp_path / "recaps"
    poll_recaps.build_season_poll_recaps(
        tables, output / "margin", objective="margin", top_n=10, season=2026,
    )
    assert rendered == [
        ("tdnet_top10_social_4x5.png", "4x5", 2026, 0),
        ("tdnet_top10_social_16x9.png", "16x9", 2026, 0),
    ]
    assert (output / "margin/week_00/tdnet_top10_social_4x5.png").exists()
    assert (output / "margin/week_00/tdnet_top10_social_16x9.png").exists()

    poll_recaps.build_season_poll_recaps(
        tables, output / "winner", objective="winner", top_n=10, season=2026,
    )
    assert len(rendered) == 2
