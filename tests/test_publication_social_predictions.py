from hashlib import sha256

import pandas as pd
from PIL import Image, ImageDraw

from gridiron_ml.publication.social_predictions import (
    PREDICTIONS_STYLE,
    render_predictions_social,
    select_featured_games,
)


def _games():
    return pd.DataFrame([
        {"game_id": "both", "away_team": "Ranked 8", "home_team": "Ranked 3",
         "tdnet_rank_away": 8, "tdnet_rank_home": 3, "pred_home_margin": 7,
         "predicted_margin": 7, "pred_winner": "Ranked 3", "pred_home_win_probability": .62,
         "market_spread_close": -6.5},
        {"game_id": "one-close", "away_team": "Plain A", "home_team": "Ranked 4",
         "tdnet_rank_home": 4, "pred_home_margin": 2, "predicted_margin": 2,
         "pred_winner": "Ranked 4", "pred_home_win_probability": .54,
         "market_spread_close": 2.5},
        {"game_id": "one-high", "away_team": "Ranked 1", "home_team": "Plain B",
         "tdnet_rank_away": 1, "pred_home_margin": -14, "predicted_margin": 14,
         "pred_winner": "Ranked 1", "pred_home_win_probability": .2},
        {"game_id": "both-low", "away_team": "Ranked 21", "home_team": "Ranked 20",
         "tdnet_rank_away": 21, "tdnet_rank_home": 20, "pred_home_margin": 4,
         "predicted_margin": 4, "pred_winner": "Ranked 20", "pred_home_win_probability": .57},
        {"game_id": "sickos", "away_team": "Plain C", "home_team": "Plain D",
         "pred_home_margin": .5, "predicted_margin": .5, "pred_winner": "Plain D",
         "pred_home_win_probability": .51},
        {"game_id": "fallback", "away_team": "Plain E", "home_team": "Plain F",
         "pred_home_margin": 1.5, "predicted_margin": 1.5, "pred_winner": "Plain F",
         "pred_home_win_probability": None},
    ])


def _logos(directory, games):
    directory.mkdir()
    for team in sorted(set(games["away_team"]) | set(games["home_team"])):
        image = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
        ImageDraw.Draw(image).ellipse((45, 45, 455, 455), fill=(30, 167, 255, 255))
        image.save(directory / (team.lower().replace(" ", "_") + ".png"))


def test_selection_prefers_ranked_buckets_and_reserves_sickos():
    featured, sickos = select_featured_games(_games())
    assert [game.game_id for game in featured] == ["both", "both-low", "one-high"]
    assert featured[0].vegas_favorite == "Ranked 3"
    assert featured[0].vegas_line == -6.5
    all_featured, _ = select_featured_games(_games(), count=10)
    away_favorite = next(game for game in all_featured if game.game_id == "one-close")
    assert away_favorite.vegas_favorite == "Plain A"
    assert away_favorite.vegas_line == -2.5
    assert sickos.game_id == "sickos"
    featured, sickos = select_featured_games(
        _games().query("game_id not in ['one-close', 'both-low']")
    )
    assert [game.game_id for game in featured] == ["both", "one-high", "fallback"]
    assert sickos.game_id == "sickos"


def test_feature_cards_share_equal_geometry_and_sickos_is_visually_separate():
    for variant in ("4x5", "16x9"):
        boxes = PREDICTIONS_STYLE["layouts"][variant]["featured"]
        sizes = {(x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes}
        assert len(sizes) == 1
        assert PREDICTIONS_STYLE["layouts"][variant]["divider_y"] < PREDICTIONS_STYLE["layouts"][variant]["sickos"][1]
    assert PREDICTIONS_STYLE["divider_width"] >= 4
    sickos = PREDICTIONS_STYLE["sickos"]
    assert sickos["background_tint"][3] >= 240
    assert sickos["border_color"] == PREDICTIONS_STYLE["colors"]["gridiron_violet"]
    assert sickos["watermark"]["path"].endswith("Sickos_White.png")
    assert sickos["watermark"]["4x5"]["scale"] != sickos["watermark"]["16x9"]["scale"]
    for variant in ("4x5", "16x9"):
        sizes = PREDICTIONS_STYLE["font_sizes"][variant]
        assert 0 < sizes["prediction"] - sizes["probability"] <= 4

def test_prediction_variants_are_exact_and_deterministic(tmp_path):
    games = _games()
    logos = tmp_path / "logos"
    _logos(logos, games)
    hashes = []
    for variant, dimensions in PREDICTIONS_STYLE["canvases"].items():
        path = render_predictions_social(
            games, tmp_path / f"predictions_{variant}.png", season=2026, week=1,
            logo_dir=logos, variant=variant, generated_at_utc="2026-08-24T12:00:00Z",
            git_commit="a" * 40, source_sha256="b" * 64,
        )
        with Image.open(path) as image:
            assert image.size == dimensions
        first = sha256(path.read_bytes()).hexdigest()
        render_predictions_social(
            games, path, season=2026, week=1, logo_dir=logos, variant=variant,
            generated_at_utc="2026-08-24T12:00:00Z", git_commit="a" * 40,
            source_sha256="b" * 64,
        )
        assert sha256(path.read_bytes()).hexdigest() == first
        hashes.append(first)
    assert hashes[0] != hashes[1]
