"""Procedural TDNet Top 3 + Sickos weekly prediction graphics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from gridiron_ml.td_run.poll_viz import resolve_team_logo_path

from .figure_theme import TDNET_COLORS
from .social_top10 import (
    SOCIAL_STYLE,
    _draw_data_field,
    _draw_footer,
    _font,
    _panel,
    _panel_accent_polygon,
    _place_logo,
    _rgba,
)


# Single edit point for the prediction companion's copy, geometry, type, and
# semantic accents. Shared brand primitives continue to come from SOCIAL_STYLE.
PREDICTIONS_STYLE = {
    "canvases": SOCIAL_STYLE["canvases"],
    "background": SOCIAL_STYLE["background"],
    "panel_radii": SOCIAL_STYLE["panel_radii"],
    "safe_margin": SOCIAL_STYLE["safe_margin"],
    "minimum_logo_source_px": SOCIAL_STYLE["minimum_logo_source_px"],
    "colors": TDNET_COLORS,
    "title": "TDNet PREDICTIONS",
    "subtitle": "3 FEATURED GAMES + SICKOS PICK",
    "sickos_title": "SICKOS GAME OF THE WEEK",
    "sickos_subtitle": "Closest unranked matchup",
    "header_brand_mark": {
        "path": "docs/style/logos/dark_trans.png",
        "opacity": 50,
        "angle": 6,
        "boxes": {"4x5": (900, 18, 1020, 138), "16x9": (1136, 8, 1234, 100)},
    },
    "sickos": {
        "background_tint": (20, 17, 66, 242),
        "border_color": TDNET_COLORS["gridiron_violet"],
        "title_color": TDNET_COLORS["white"],
        "subtitle_color": TDNET_COLORS["polar_mist"],
        "subtitle_alpha": 215,
        "watermark": {
            "path": "docs/style/logos/Sickos_White.png",
            "4x5": {
                "opacity": 11, "glow_opacity": 15, "scale": 0.18,
                "glow_radius": 13,
                "positions": ((0.03, 0.08), (0.47, 0.08), (0.91, 0.08),
                              (0.25, 0.78), (0.69, 0.78), (1.08, 0.78)),
            },
            "16x9": {
                "opacity": 11, "glow_opacity": 15, "scale": 0.13,
                "glow_radius": 10,
                "positions": ((0.02, 0.08), (0.40, 0.08), (0.78, 0.08),
                              (0.21, 0.78), (0.59, 0.78), (0.98, 0.78)),
            },
        },
    },
    "divider_width": 5,
    "logo_sizes": {
        "4x5": {"feature": 85, "sickos": 132},
        "16x9": {"feature": 82, "sickos": 94},
    },
    "font_sizes": {
        "4x5": {"title": 55, "meta": 20, "label": 19, "team": 25,
                "prediction": 30, "probability": 26, "detail": 17,
                "market": 12, "sickos_team": 29},
        "16x9": {"title": 43, "meta": 16, "label": 15, "team": 21,
                  "prediction": 23, "probability": 20, "detail": 14,
                  "market": 11, "sickos_team": 23},
    },
    "layouts": {
        "4x5": {
            "featured": ((58, 168, 1022, 372), (58, 384, 1022, 588),
                         (58, 600, 1022, 804)),
            "divider_y": 824,
            "sickos": (58, 844, 1022, 1289),
        },
        "16x9": {
            "featured": ((40, 122, 426, 370), (447, 122, 833, 370),
                         (854, 122, 1240, 370)),
            "divider_y": 390,
            "sickos": (40, 408, 1240, 672),
        },
    },
}


@dataclass(frozen=True)
class FeaturedGame:
    game_id: str
    away_team: str
    home_team: str
    away_rank: int | None
    home_rank: int | None
    predicted_winner: str
    predicted_margin: float
    winner_probability: float | None
    vegas_favorite: str | None = None
    vegas_line: float | None = None
    neutral_site: bool = False

    @property
    def ranked_count(self) -> int:
        return int(self.away_rank is not None) + int(self.home_rank is not None)

    @property
    def best_rank(self) -> int:
        ranks = [rank for rank in (self.away_rank, self.home_rank) if rank is not None]
        return min(ranks, default=10_000)

    @property
    def combined_rank(self) -> int:
        return sum(rank if rank is not None else 26 for rank in (self.away_rank, self.home_rank))


def select_featured_games(
    games: pd.DataFrame,
    tdnet_poll: pd.DataFrame | None = None,
    *,
    count: int = 3,
) -> tuple[list[FeaturedGame], FeaturedGame | None]:
    """Select ranked features and the closest unranked Sickos matchup.

    The closest unranked game is reserved for Sickos. If fewer than ``count``
    ranked games exist, remaining slots use the next-closest unranked games.
    """
    normalized = _normalize_games(games, tdnet_poll)
    ranked = [game for game in normalized if game.ranked_count]
    unranked = [game for game in normalized if not game.ranked_count]
    ranked.sort(key=lambda game: (
        -game.ranked_count, game.best_rank, abs(game.predicted_margin),
        game.combined_rank, game.game_id,
    ))
    unranked.sort(key=lambda game: (
        abs(game.predicted_margin),
        abs((game.winner_probability if game.winner_probability is not None else 0.5) - 0.5),
        game.game_id,
    ))
    sickos = unranked[0] if unranked else None
    fallback = unranked[1:] if sickos else unranked
    featured = (ranked + fallback)[: int(count)]
    return featured, sickos


def render_predictions_social(
    games: pd.DataFrame,
    output_path: str | Path,
    *,
    season: int,
    week: int,
    logo_dir: str | Path | None,
    tdnet_poll: pd.DataFrame | None = None,
    variant: str = "4x5",
    generated_at_utc: str | None = None,
    git_commit: str | None = None,
    source_sha256: str | None = None,
) -> Path:
    """Render the prediction companion to the TDNet Top 10 graphic."""
    if variant not in PREDICTIONS_STYLE["canvases"]:
        raise ValueError(f"Unsupported social variant: {variant}")
    featured, sickos = select_featured_games(games, tdnet_poll)
    if not featured:
        raise ValueError("Prediction social rendering requires at least one valid game.")
    _require_high_resolution_game_logos([*featured, *([sickos] if sickos else [])], logo_dir)
    size = PREDICTIONS_STYLE["canvases"][variant]
    image = Image.new("RGB", size, PREDICTIONS_STYLE["background"])
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_data_field(draw, size)
    _draw_prediction_header(draw, variant, season, week)
    _place_prediction_brand_mark(image, variant)
    layout = PREDICTIONS_STYLE["layouts"][variant]
    for index, box in enumerate(layout["featured"]):
        game = featured[index] if index < len(featured) else None
        _draw_matchup_card(draw, image, game, box, logo_dir, variant=variant,
                           role="feature", label=f"FEATURED {index + 1}")
    _draw_section_divider(draw, variant, int(layout["divider_y"]))
    _draw_matchup_card(draw, image, sickos, layout["sickos"], logo_dir,
                       variant=variant, role="sickos", label=PREDICTIONS_STYLE["sickos_title"])
    _draw_footer(
        draw, size, variant, generated_at_utc, git_commit, source_sha256,
        left_text="TDNet • FULL PREDICTION SLATE IN THE ARTICLE",
        source_label="predictions",
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)
    return path


def _normalize_games(games: pd.DataFrame, poll: pd.DataFrame | None) -> list[FeaturedGame]:
    if games is None or games.empty:
        return []
    frame = games.copy()
    required = {"away_team", "home_team"}
    if not required.issubset(frame.columns):
        raise ValueError("Prediction games require away_team and home_team columns.")
    ranks: dict[str, int] = {}
    if poll is not None and not poll.empty:
        team_col = next((c for c in ("team", "keys_team", "school") if c in poll), None)
        rank_col = next((c for c in ("rank", "poll_rank") if c in poll), None)
        if team_col and rank_col:
            ranks = {
                str(team): int(rank) for team, rank in zip(poll[team_col], poll[rank_col])
                if pd.notna(team) and pd.notna(rank)
            }
    output = []
    for index, row in frame.iterrows():
        away, home = str(row["away_team"]), str(row["home_team"])
        away_rank = _optional_int(row.get("tdnet_rank_away"))
        home_rank = _optional_int(row.get("tdnet_rank_home"))
        away_rank = away_rank if away_rank is not None else ranks.get(away)
        home_rank = home_rank if home_rank is not None else ranks.get(home)
        signed_margin = pd.to_numeric(row.get("pred_home_margin"), errors="coerce")
        margin = pd.to_numeric(row.get("predicted_margin"), errors="coerce")
        if pd.isna(margin) and pd.notna(signed_margin):
            margin = abs(float(signed_margin))
        if pd.isna(margin):
            continue
        winner = row.get("pred_winner")
        if pd.isna(winner) or not str(winner).strip():
            winner = home if pd.notna(signed_margin) and float(signed_margin) >= 0 else away
        home_probability = pd.to_numeric(row.get("pred_home_win_probability"), errors="coerce")
        winner_probability = None
        if pd.notna(home_probability):
            winner_probability = float(home_probability) if str(winner) == home else 1 - float(home_probability)
        spread = _market_spread(row)
        vegas_favorite = None
        vegas_line = None
        if spread is not None and spread != 0:
            vegas_favorite = home if spread < 0 else away
            vegas_line = -abs(spread)
        output.append(FeaturedGame(
            game_id=str(row.get("game_id", index)), away_team=away, home_team=home,
            away_rank=away_rank, home_rank=home_rank, predicted_winner=str(winner),
            predicted_margin=abs(float(margin)), winner_probability=winner_probability,
            vegas_favorite=vegas_favorite, vegas_line=vegas_line,
            neutral_site=bool(row.get("neutral_site", False)),
        ))
    return output


def _draw_prediction_header(draw, variant: str, season: int, week: int) -> None:
    style, colors = PREDICTIONS_STYLE, PREDICTIONS_STYLE["colors"]
    sizes, margin = style["font_sizes"][variant], style["safe_margin"][variant]
    y = 25 if variant == "4x5" else 14
    draw.text((margin, y), "TDNet", font=_font(sizes["title"], bold=True), fill=colors["white"])
    box = draw.textbbox((margin, y), "TDNet", font=_font(sizes["title"], bold=True))
    draw.text((box[2] + (16 if variant == "4x5" else 11), y), "PREDICTIONS",
              font=_font(sizes["title"], bold=True), fill=colors["edge_pink"])
    meta_y = 98 if variant == "4x5" else 66
    draw.text((margin + 2, meta_y), f"{season}  •  WEEK {week}",
              font=_font(sizes["meta"]), fill=colors["polar_mist"])
    width = style["canvases"][variant][0]
    draw.text((width // 2, meta_y), style["subtitle"],
              font=_font(sizes["meta"], bold=True),
              fill=_rgba(colors["medium_gray"], 190), anchor="ma")
    line_y = 145 if variant == "4x5" else 104
    draw.line((margin, line_y, width - margin, line_y), fill=colors["ion_blue"], width=2)


def _draw_matchup_card(draw, image, game, box, logo_dir, *, variant, role, label) -> None:
    colors, sizes = PREDICTIONS_STYLE["colors"], PREDICTIONS_STYLE["font_sizes"][variant]
    sickos_style = PREDICTIONS_STYLE["sickos"]
    accent = sickos_style["border_color"] if role == "sickos" else colors["ion_blue"]
    radius_key = "hero" if role == "sickos" else "support"
    _prediction_panel(
        draw, box, accent=accent, radius=PREDICTIONS_STYLE["panel_radii"][radius_key],
        bright=role == "sickos",
    )
    if role == "sickos":
        _place_sickos_watermark(image, variant, box)
    x1, y1, x2, y2 = map(int, box)
    width, height = x2 - x1, y2 - y1
    label_fill = sickos_style["title_color"] if role == "sickos" else accent
    draw.text((x1 + 22, y1 + 15), label, font=_font(sizes["label"], bold=True), fill=label_fill)
    if role == "sickos":
        draw.text((x1 + 22, y1 + 42), PREDICTIONS_STYLE["sickos_subtitle"],
                  font=_font(max(11, sizes["detail"] - 2)),
                  fill=_rgba(sickos_style["subtitle_color"], sickos_style["subtitle_alpha"]))
    if game is None:
        draw.text(((x1 + x2) // 2, (y1 + y2) // 2), "NO ELIGIBLE MATCHUP",
                  font=_font(sizes["team"], bold=True), fill=colors["medium_gray"], anchor="mm")
        return
    if role == "sickos" and variant == "16x9":
        _draw_landscape_sickos_content(draw, image, game, box, logo_dir, sizes)
        return
    compact = role == "feature"
    top_pad = (22 if compact and variant == "4x5" else 48) if role != "sickos" else 95
    logo_size = int(PREDICTIONS_STYLE["logo_sizes"][variant]["feature" if compact else "sickos"])
    center_y = y1 + top_pad + logo_size // 2
    left_cx, right_cx = x1 + width * .28, x1 + width * .72
    away_box = (round(left_cx - logo_size / 2), center_y - logo_size // 2,
                round(left_cx + logo_size / 2), center_y + logo_size // 2)
    home_box = (round(right_cx - logo_size / 2), center_y - logo_size // 2,
                round(right_cx + logo_size / 2), center_y + logo_size // 2)
    _place_logo(image, game.away_team, logo_dir, away_box, fallback_size=max(20, logo_size // 3))
    _place_logo(image, game.home_team, logo_dir, home_box, fallback_size=max(20, logo_size // 3))
    draw.text(((x1 + x2) // 2, center_y), "VS" if game.neutral_site else "@",
              font=_font(sizes["label"] + 5, bold=True),
              fill=_rgba(colors["polar_mist"], 225), anchor="mm")
    team_y = center_y + logo_size // 2 + (8 if compact else 14)
    team_size = sizes["team"] if compact else sizes["sickos_team"]
    _team_label(draw, game.away_team, game.away_rank, left_cx, team_y, width * .42, team_size, colors["white"])
    _team_label(draw, game.home_team, game.home_rank, right_cx, team_y, width * .42, team_size, colors["white"])
    _draw_market_line(draw, game, left_cx, right_cx, team_y + team_size + 1,
                      sizes["market"], colors["signal_orange"])
    prediction_gap = 30 if compact and variant == "16x9" else (22 if compact else 35)
    prediction_y = team_y + team_size + prediction_gap
    probability_y = prediction_y + sizes["prediction"] + 5
    if compact and variant == "4x5":
        probability_y = y2 - 14
        prediction_y = probability_y - sizes["probability"] - 2
    winner = _short_name(game.predicted_winner, 22 if compact else 28)
    margin = _format_margin(game.predicted_margin)
    draw.text(((x1 + x2) // 2, prediction_y), f"{winner} BY {margin}",
              font=_font(sizes["prediction"], bold=True), fill=colors["soft_mint"], anchor="mm")
    if game.winner_probability is not None:
        draw.text(((x1 + x2) // 2, probability_y),
                  f"{game.winner_probability:.0%} WIN PROB",
                  font=_font(sizes["probability"], bold=True),
                  fill=_rgba(colors["polar_mist"], 220), anchor="mm")


def _prediction_panel(draw, box, *, accent, radius, bright=False) -> None:
    if not bright:
        _panel(draw, box, accent=accent, radius=radius, alpha=235)
        return
    sickos_style = PREDICTIONS_STYLE["sickos"]
    draw.rounded_rectangle(
        box, radius=radius, fill=sickos_style["background_tint"],
        outline=_rgba(sickos_style["border_color"], 225), width=3,
    )
    draw.polygon(
        _panel_accent_polygon(box, radius=radius, width=8),
        fill=_rgba(sickos_style["border_color"], 245),
    )


def _place_sickos_watermark(image: Image.Image, variant: str, box) -> None:
    """Tile the official Sickos mark as a pattern clipped to its card."""
    settings = PREDICTIONS_STYLE["sickos"]["watermark"]
    repository_root = Path(__file__).resolve().parents[3]
    path = repository_root / str(settings["path"])
    if not path.exists():
        return
    with Image.open(path) as source:
        mark = source.convert("RGBA")
    visible = mark.getchannel("A").getbbox()
    if visible is None:
        return
    mark = mark.crop(visible)
    geometry = settings[variant]
    scale = float(geometry["scale"])
    mark = mark.resize(
        (max(1, round(mark.width * scale)), max(1, round(mark.height * scale))),
        Image.Resampling.LANCZOS,
    )
    alpha = mark.getchannel("A")
    glow_alpha = alpha.point(
        lambda value: round(value * int(geometry["glow_opacity"]) / 255)
    ).filter(ImageFilter.GaussianBlur(float(geometry["glow_radius"])))
    glow = Image.new("RGBA", mark.size, _rgba(PREDICTIONS_STYLE["colors"]["gridiron_violet"], 0))
    glow.putalpha(glow_alpha)
    mark_alpha = alpha.point(
        lambda value: round(value * int(geometry["opacity"]) / 255)
    )
    mark.putalpha(mark_alpha)
    pattern = Image.new("RGBA", image.size, (0, 0, 0, 0))
    x1, y1, x2, y2 = map(int, box)
    for relative_x, relative_y in geometry["positions"]:
        center_x = round(x1 + (x2 - x1) * float(relative_x))
        center_y = round(y1 + (y2 - y1) * float(relative_y))
        position = (center_x - mark.width // 2, center_y - mark.height // 2)
        pattern.alpha_composite(glow, position)
        pattern.alpha_composite(mark, position)

    clip = Image.new("L", image.size, 0)
    clip_draw = ImageDraw.Draw(clip)
    radius = int(PREDICTIONS_STYLE["panel_radii"]["hero"])
    clip_draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=255)
    pattern.putalpha(ImageChops.multiply(pattern.getchannel("A"), clip))
    composited = Image.alpha_composite(image.convert("RGBA"), pattern).convert("RGB")
    image.paste(composited)


def _place_prediction_brand_mark(image: Image.Image, variant: str) -> None:
    """Place the faint TDNet mark in each prediction header's safe area."""
    settings = PREDICTIONS_STYLE["header_brand_mark"]
    repository_root = Path(__file__).resolve().parents[3]
    path = repository_root / str(settings["path"])
    if not path.exists():
        return
    with Image.open(path) as source:
        mark = source.convert("RGBA")
    visible = mark.getchannel("A").getbbox()
    if visible is None:
        return
    mark = mark.crop(visible)
    alpha = mark.getchannel("A").point(
        lambda value: round(value * int(settings["opacity"]) / 255)
    )
    white_mark = Image.new("RGBA", mark.size, (255, 255, 255, 0))
    white_mark.putalpha(alpha)
    x1, y1, x2, y2 = map(int, settings["boxes"][variant])
    scale = min((x2 - x1) / white_mark.width, (y2 - y1) / white_mark.height)
    white_mark = white_mark.resize(
        (round(white_mark.width * scale), round(white_mark.height * scale)),
        Image.Resampling.LANCZOS,
    ).rotate(
        float(settings["angle"]), resample=Image.Resampling.BICUBIC, expand=True,
    )
    px = x1 + (x2 - x1 - white_mark.width) // 2
    py = y1 + (y2 - y1 - white_mark.height) // 2
    image.paste(white_mark, (px, py), white_mark)


def _draw_section_divider(draw, variant: str, y: int) -> None:
    margin = int(PREDICTIONS_STYLE["safe_margin"][variant])
    width = int(PREDICTIONS_STYLE["canvases"][variant][0])
    colors = PREDICTIONS_STYLE["colors"]
    draw.line((margin, y, width - margin, y), fill=colors["gridiron_violet"],
              width=int(PREDICTIONS_STYLE["divider_width"]))


def _draw_landscape_sickos_content(draw, image, game, box, logo_dir, sizes) -> None:
    colors = PREDICTIONS_STYLE["colors"]
    x1, y1, x2, _ = map(int, box)
    logo_size = int(PREDICTIONS_STYLE["logo_sizes"]["16x9"]["sickos"])
    centers = (x1 + 480, x1 + 720)
    center_y = y1 + 109
    for team, rank, center_x in (
        (game.away_team, game.away_rank, centers[0]),
        (game.home_team, game.home_rank, centers[1]),
    ):
        logo_box = (center_x - logo_size // 2, center_y - logo_size // 2,
                    center_x + logo_size // 2, center_y + logo_size // 2)
        _place_logo(image, team, logo_dir, logo_box, fallback_size=25)
        _team_label(draw, team, rank, center_x, center_y + logo_size // 2 + 7,
                    210, sizes["sickos_team"], colors["white"])
    draw.text(((centers[0] + centers[1]) // 2, center_y),
              "VS" if game.neutral_site else "@",
              font=_font(sizes["label"] + 5, bold=True), fill=colors["white"], anchor="mm")
    team_y = center_y + logo_size // 2 + 7
    _draw_market_line(draw, game, centers[0], centers[1],
                      team_y + sizes["sickos_team"] + 1,
                      sizes["market"], colors["signal_orange"])
    prediction_x = (x1 + x2) // 2
    winner = _short_name(game.predicted_winner, 24)
    prediction_y = center_y + logo_size // 2 + sizes["sickos_team"] + 34
    draw.text((prediction_x, prediction_y),
              f"{winner} BY {_format_margin(game.predicted_margin)}",
              font=_font(sizes["prediction"], bold=True),
              fill=colors["soft_mint"], anchor="mm")
    if game.winner_probability is not None:
        draw.text((prediction_x, prediction_y + sizes["prediction"] + 5),
                  f"{game.winner_probability:.0%} WIN PROB",
                  font=_font(sizes["probability"], bold=True),
                  fill=_rgba(colors["white"], 230), anchor="mm")


def _team_label(draw, team, rank, x, y, max_width, preferred, fill) -> None:
    label = f"#{rank} {_short_name(team, 24)}" if rank is not None else _short_name(team, 24)
    size = preferred
    font = _font(size, bold=True)
    while size > 14 and draw.textlength(label, font=font) > max_width:
        size -= 1
        font = _font(size, bold=True)
    draw.text((x, y), label, font=font, fill=fill, anchor="ma")


def _draw_market_line(draw, game, away_x, home_x, y, size, fill) -> None:
    if game.vegas_favorite is None or game.vegas_line is None:
        return
    x = away_x if game.vegas_favorite == game.away_team else home_x
    draw.text(
        (x, y), f"(VEGAS {game.vegas_line:.1f})",
        font=_font(size, bold=True), fill=fill, anchor="ma",
    )


def _require_high_resolution_game_logos(games, logo_dir) -> None:
    threshold = int(PREDICTIONS_STYLE["minimum_logo_source_px"])
    undersized = []
    teams = {team for game in games for team in (game.away_team, game.home_team)}
    for team in sorted(teams):
        path = resolve_team_logo_path(team, logo_dir)
        if path is None:
            continue
        with Image.open(path) as logo:
            if min(logo.size) < threshold:
                undersized.append(f"{team} ({logo.width}x{logo.height})")
    if undersized:
        raise ValueError("TDNet prediction graphics require 500px logo sources: " + ", ".join(undersized))


def _optional_int(value) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else None


def _market_spread(row: pd.Series) -> float | None:
    for column in (
        "market_spread_close", "vegas_spread_close_as_of_prediction",
        "vegas_spread", "home_spread", "spread",
    ):
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return float(value)
    return None


def _format_margin(value: float) -> str:
    return str(int(round(value))) if abs(value - round(value)) < 0.05 else f"{value:.1f}"


def _short_name(team: str, limit: int) -> str:
    text = str(team)
    aliases = {"North Dakota State": "NDSU", "Jacksonville State": "Jacksonville St.",
               "New Mexico State": "New Mexico St.", "San José State": "San José St."}
    text = aliases.get(text, text)
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"
