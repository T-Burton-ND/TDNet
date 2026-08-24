"""Mobile-first TDNet Top 10 social graphics.

The renderer is deliberately separate from the analytical Top 25 figures.  All
visual tuning belongs in ``SOCIAL_STYLE`` so weekly automation and one-off
prototype generation use the same design system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from math import sqrt
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from gridiron_ml.td_run.poll_viz import load_team_logo_image, resolve_team_logo_path

from .figure_theme import TDNET_COLORS


# This is the single edit point for colors, dimensions, type, spacing, and badge
# behavior in the TDNet social visual system.
SOCIAL_STYLE = {
    "canvases": {"4x5": (1080, 1350), "16x9": (1280, 720)},
    "fonts": {
        "body_regular": ("DejaVuSans.ttf",),
        "body_bold": ("DejaVuSans-Bold.ttf",),
        "rank": (
            "/usr/share/fonts/adobe-source-code-pro/SourceCodePro-Black.otf",
            "DejaVuSansCondensed-Bold.ttf",
        ),
    },
    "badge_min_rank_difference": 5,
    "minimum_logo_source_px": 500,
    "team_name_aliases": {},
    "logo_scale_overrides": {"Georgia": 1.14},
    "background": "#172B61",
    "panel_fill": (5, 15, 43, 235),
    "panel_accent_width": 7,
    "panel_radii": {"hero": 28, "support": 22, "row": 12},
    "lower_rank_box": {"left_inset": 18, "logo_gap": 10},
    "lower_text_optical_y": 1,
    "fixed_line_height_ratio": {"name": 0.87, "context": 0.80},
    "portrait_section_gap": 12,
    "context_line_gap": 6,
    "detail_text_alpha": 190,
    "weak_ballot_support_ratio": 30 / 33,
    "tie_hyphen_width_ratio": 0.18,
    "tie_hyphen_gap_ratio": 0.08,
    "discrepancy_stamp_angle": -9,
    "discrepancy_stamp_scale": 1.10,
    "discrepancy_stamp_right_offset": {"4x5": -4, "16x9": -8},
    "discrepancy_stamp_y_offset": 4,
    "title_segment_gap": {"4x5": 18, "16x9": 12},
    "header_rule_width": 2,
    "ballot_count_last_week": None,
    "landscape_header_label": "WEEKLY MODEL CONSENSUS",
    "landscape_rank_column": {"top": 122, "bottom": 672, "gap": 3},
    "brand_mark": {
        "path": "docs/style/logos/dark_trans.png",
        "opacity": 55,
        "angle": 6,
        "boxes": {
            "16x9": (455, 408, 725, 678),
            "4x5": (900, 18, 1020, 138),
        },
        "scale": {"16x9": 1.05, "4x5": 1.0},
    },
    "safe_margin": {"4x5": 58, "16x9": 40},
    "colors": TDNET_COLORS,
    "font_sizes": {
        "4x5": {"title": 58, "meta": 22, "hero_rank": 165, "hero_name": 54,
                 "pod_rank": 64, "pod_name": 34, "row_rank": 40, "row_name": 31,
                 "points": 15, "badge": 20, "footer": 16},
        "16x9": {"title": 45, "meta": 18, "hero_rank": 150, "hero_name": 45,
                  "pod_rank": 52, "pod_name": 29, "row_rank": 31, "row_name": 24,
                  "points": 12, "badge": 18, "footer": 13},
    },
}


@dataclass(frozen=True)
class SocialTeam:
    rank: int
    team: str
    reference_rank: int | None = None
    points: int | None = None
    first_place_votes: int = 0
    ballots_seen: int | None = None
    top25_votes: int | None = None
    best_rank: int | None = None
    display_rank: int | None = None
    tied: bool = False

    @property
    def social_rank(self) -> int:
        return int(self.display_rank if self.display_rank is not None else self.rank)

    @property
    def discrepancy(self) -> int | None:
        if self.reference_rank is None:
            return None
        return self.reference_rank - self.social_rank


def prepare_top10(poll: pd.DataFrame) -> list[SocialTeam]:
    """Normalize a public TDNet poll and enforce one team at every rank 1--10."""
    if poll is None or poll.empty:
        raise ValueError("TDNet Top 10 social rendering requires a non-empty poll.")
    rank_column = next((c for c in ("rank", "poll_rank") if c in poll), None)
    team_column = next((c for c in ("team", "keys_team", "school") if c in poll), None)
    if rank_column is None or team_column is None:
        raise ValueError("TDNet poll needs rank and team/keys_team columns.")
    frame = poll.copy()
    frame[rank_column] = pd.to_numeric(frame[rank_column], errors="coerce")
    frame = frame.loc[frame[rank_column].between(1, 10)].copy()
    frame[rank_column] = frame[rank_column].astype(int)
    if frame[rank_column].duplicated().any() or set(frame[rank_column]) != set(range(1, 11)):
        raise ValueError("TDNet Top 10 must contain unique ranks 1 through 10.")
    reference_column = next(
        (c for c in ("reference_rank", "ap_rank") if c in frame), None
    )
    points_column = next((c for c in ("poll_points", "points") if c in frame), None)
    first_place_column = next(
        (c for c in ("first_place_votes", "first_place_vote_count") if c in frame), None
    )
    ballots_column = next((c for c in ("ballots_seen", "voter_count") if c in frame), None)
    top25_votes_column = next(
        (c for c in ("top25_votes", "ballot_support") if c in frame), None
    )
    best_rank_column = next(
        (c for c in ("best_rank", "highest_rank") if c in frame), None
    )
    teams = []
    for row in frame.sort_values(rank_column).itertuples(index=False):
        values = row._asdict()
        team = str(values[team_column]).strip()
        if not team or team.casefold() == "nan":
            raise ValueError("TDNet Top 10 contains a missing team name.")
        reference = values.get(reference_column) if reference_column else None
        reference = pd.to_numeric(reference, errors="coerce")
        points = values.get(points_column) if points_column else None
        points = pd.to_numeric(points, errors="coerce")
        first_place = values.get(first_place_column) if first_place_column else 0
        first_place = pd.to_numeric(first_place, errors="coerce")
        ballots = values.get(ballots_column) if ballots_column else None
        ballots = pd.to_numeric(ballots, errors="coerce")
        top25_votes = values.get(top25_votes_column) if top25_votes_column else None
        top25_votes = pd.to_numeric(top25_votes, errors="coerce")
        best_rank = values.get(best_rank_column) if best_rank_column else None
        best_rank = pd.to_numeric(best_rank, errors="coerce")
        teams.append(
            SocialTeam(
                rank=int(values[rank_column]),
                team=team,
                reference_rank=int(reference) if pd.notna(reference) else None,
                points=int(round(points)) if pd.notna(points) else None,
                first_place_votes=int(first_place) if pd.notna(first_place) else 0,
                ballots_seen=int(ballots) if pd.notna(ballots) else None,
                top25_votes=int(top25_votes) if pd.notna(top25_votes) else None,
                best_rank=int(best_rank) if pd.notna(best_rank) else None,
            )
        )
    if len({team.team.casefold() for team in teams}) != 10:
        raise ValueError("TDNet Top 10 must contain ten unique teams.")
    point_groups: dict[int, list[int]] = {}
    for index, team in enumerate(teams):
        if team.points is not None:
            point_groups.setdefault(team.points, []).append(index)
    for indexes in point_groups.values():
        if len(indexes) > 1:
            shared_rank = min(teams[index].rank for index in indexes)
            for index in indexes:
                teams[index] = replace(teams[index], display_rank=shared_rank, tied=True)
    return teams


def noteworthy_badge(team: SocialTeam, *, reference_label: str = "AP") -> str | None:
    """Return a curiosity badge only for a large, clearly interpretable gap."""
    difference = team.discrepancy
    if difference is None or abs(difference) < int(SOCIAL_STYLE["badge_min_rank_difference"]):
        return None
    arrow = "↑" if difference > 0 else "↓"
    return f"{arrow}{abs(difference)} vs {reference_label}"


def low_resolution_logo_teams(
    poll: pd.DataFrame,
    logo_dir: str | Path | None,
    *,
    minimum_px: int | None = None,
) -> list[str]:
    """List ranked teams whose available source mark is too small for social use."""
    threshold = int(minimum_px or SOCIAL_STYLE["minimum_logo_source_px"])
    low_resolution = []
    for team in prepare_top10(poll):
        path = resolve_team_logo_path(team.team, logo_dir)
        if path is None:
            low_resolution.append(team.team)
            continue
        with Image.open(path) as logo:
            if min(logo.size) < threshold:
                low_resolution.append(team.team)
    return low_resolution


def require_high_resolution_logos(
    poll: pd.DataFrame,
    logo_dir: str | Path | None,
) -> None:
    """Reject resolved social-logo sources smaller than the 500px contract."""
    threshold = int(SOCIAL_STYLE["minimum_logo_source_px"])
    undersized = []
    for team in prepare_top10(poll):
        path = resolve_team_logo_path(team.team, logo_dir)
        if path is None:
            continue
        with Image.open(path) as logo:
            if min(logo.size) < threshold:
                undersized.append(f"{team.team} ({logo.width}x{logo.height})")
    if undersized:
        raise ValueError(
            "TDNet social graphics require 500px logo sources; refresh these marks with "
            "gridiron_ml.cli.publication.refresh_social_logos: " + ", ".join(undersized)
        )


def render_top10_social(
    poll: pd.DataFrame,
    output_path: str | Path,
    *,
    season: int,
    week: int,
    logo_dir: str | Path | None,
    variant: str = "4x5",
    generated_at_utc: str | None = None,
    git_commit: str | None = None,
    source_sha256: str | None = None,
    reference_label: str = "AP",
) -> Path:
    """Render a native portrait or landscape Top 10 PNG from a TDNet poll."""
    if variant not in SOCIAL_STYLE["canvases"]:
        raise ValueError(f"Unsupported social variant: {variant}")
    require_high_resolution_logos(poll, logo_dir)
    teams = prepare_top10(poll)
    size = SOCIAL_STYLE["canvases"][variant]
    colors = SOCIAL_STYLE["colors"]
    image = Image.new("RGB", size, SOCIAL_STYLE["background"])
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_data_field(draw, size)
    if variant == "4x5":
        _draw_portrait(draw, image, teams, season, week, logo_dir, reference_label)
    else:
        _draw_landscape(draw, image, teams, season, week, logo_dir, reference_label)
    _draw_footer(draw, size, variant, generated_at_utc, git_commit, source_sha256)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)
    return path


def _draw_data_field(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    colors = SOCIAL_STYLE["colors"]
    step = max(54, width // 18)
    for x in range(step, width, step):
        alpha = 14 if x % (step * 3) else 28
        draw.line((x, 0, x, height), fill=_rgba(colors["ion_blue"], alpha), width=1)
    for y in range(step, height, step):
        draw.line((0, y, width, y), fill=_rgba(colors["ion_blue"], 12), width=1)
    # A restrained network trace makes the visual identifiably computational.
    nodes = [(0.05, .18), (.21, .12), (.34, .24), (.55, .13), (.74, .25), (.94, .16),
             (.12, .74), (.31, .66), (.51, .79), (.72, .68), (.92, .82)]
    points = [(int(x * width), int(y * height)) for x, y in nodes]
    for start, end in zip(points, points[1:]):
        draw.line((*start, *end), fill=_rgba(colors["edge_pink"], 12), width=2)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=_rgba(colors["ion_blue"], 34))
    # Hash-like ticks, not a literal field.
    for y in range(step, height, step * 2):
        draw.line((width * .48, y, width * .495, y), fill=_rgba(colors["soft_mint"], 22), width=1)
        draw.line((width * .505, y, width * .52, y), fill=_rgba(colors["soft_mint"], 22), width=1)


def _draw_header(draw, variant: str, season: int, week: int, ballot_count: int | None) -> None:
    colors = SOCIAL_STYLE["colors"]
    sizes = SOCIAL_STYLE["font_sizes"][variant]
    margin = SOCIAL_STYLE["safe_margin"][variant]
    title_y = 25 if variant == "4x5" else 14
    draw.text((margin, title_y), "TDNet", font=_font(sizes["title"], bold=True), fill=colors["white"])
    bbox = draw.textbbox((margin, title_y), "TDNet", font=_font(sizes["title"], bold=True))
    x = bbox[2] + int(SOCIAL_STYLE["title_segment_gap"][variant])
    draw.text((x, title_y), "TOP 10", font=_font(sizes["title"], bold=True), fill=colors["edge_pink"])
    meta = f"{int(season)}  •  WEEK {int(week)}"
    ballot_cutoff = SOCIAL_STYLE["ballot_count_last_week"]
    if ballot_count and (ballot_cutoff is None or int(week) <= int(ballot_cutoff)):
        meta += f"  •  {int(ballot_count)} MODEL BALLOTS"
    meta_y = 98 if variant == "4x5" else 66
    draw.text((margin + 2, meta_y), meta, font=_font(sizes["meta"]), fill=colors["polar_mist"])
    if variant == "16x9":
        draw.text(
            (1280 - margin, 78), SOCIAL_STYLE["landscape_header_label"],
            font=_font(sizes["footer"], bold=True),
            fill=_rgba(colors["medium_gray"], 190), anchor="rm",
        )
    line_y = 145 if variant == "4x5" else 104
    draw.line((margin, line_y, (1080 if variant == "4x5" else 1280) - margin, line_y),
              fill=colors["ion_blue"], width=int(SOCIAL_STYLE["header_rule_width"]))


def _draw_portrait(draw, image, teams, season, week, logo_dir, reference_label) -> None:
    _draw_header(draw, "4x5", season, week, teams[0].ballots_seen)
    _place_brand_mark(image, "4x5")
    colors = SOCIAL_STYLE["colors"]
    sizes = SOCIAL_STYLE["font_sizes"]["4x5"]
    section_gap = int(SOCIAL_STYLE["portrait_section_gap"])
    # Hero node.
    hero_box = (58, 168, 1022, 442)
    _panel(
        draw, hero_box, accent=colors["edge_pink"],
        radius=SOCIAL_STYLE["panel_radii"]["hero"],
    )
    hero_logo_box = (335, 193, 585, 417)
    _draw_rank(draw, (86, 184), _rank_label(teams[0]), sizes["hero_rank"],
               colors["edge_pink"], hero_logo_box)
    _place_logo(image, teams[0].team, logo_dir, hero_logo_box, fallback_size=76)
    name_box = _fit_text(draw, (615, 250, 980, 338), _display_name(teams[0]),
                         sizes["hero_name"], 38, colors["white"], bold=True, anchor="lm")
    _draw_context_against_name(
        draw, name_box, teams[0], 618, 980, 19, 12,
        colors["ion_blue"], centered=False,
    )

    # Supporting nodes, intentionally offset from the hero rather than a podium.
    support_top = hero_box[3] + section_gap
    support_bottom = 698
    for idx, box in (
        (1, (58, support_top, 528, support_bottom)),
        (2, (552, support_top, 1022, support_bottom)),
    ):
        team = teams[idx]
        _panel(
            draw, box, accent=colors["ion_blue"],
            radius=SOCIAL_STYLE["panel_radii"]["support"],
        )
        x1, y1, x2, y2 = box
        logo_box = (x1 + 125, y1 + 18, x1 + 272, y2 - 18)
        _draw_rank(draw, (x1 + 22, y1 + 12), _rank_label(team), sizes["pod_rank"],
                   colors["ion_blue"], logo_box)
        _place_logo(image, team.team, logo_dir, logo_box, fallback_size=48)
        _draw_team_text_block(
            draw, (x1 + 290, y1 + 12, x2 - 14, y2 - 12), team,
            sizes["pod_name"], 22, sizes["points"], 9,
            colors["white"], colors["soft_mint"], centered=False,
        )

    row_panel_height = 69
    row_top = support_bottom + section_gap
    row_height = row_panel_height + section_gap
    for offset, team in enumerate(teams[3:]):
        y1 = row_top + offset * row_height
        _rank_row(
            draw, image, team, (58, y1, 1022, y1 + row_panel_height),
            logo_dir, "4x5", reference_label,
        )


def _draw_landscape(draw, image, teams, season, week, logo_dir, reference_label) -> None:
    _draw_header(draw, "16x9", season, week, teams[0].ballots_seen)
    colors = SOCIAL_STYLE["colors"]
    sizes = SOCIAL_STYLE["font_sizes"]["16x9"]
    _panel(
        draw, (40, 122, 430, 672), accent=colors["edge_pink"],
        radius=SOCIAL_STYLE["panel_radii"]["hero"],
    )
    hero_logo_box = (100, 305, 370, 494)
    _draw_rank(draw, (60, 128), _rank_label(teams[0]), sizes["hero_rank"],
               colors["edge_pink"], hero_logo_box)
    _place_logo(image, teams[0].team, logo_dir, hero_logo_box, fallback_size=68)
    name_box = _fit_text(draw, (70, 515, 400, 574), _display_name(teams[0]),
                         sizes["hero_name"], 29, colors["white"], bold=True, anchor="mm")
    _draw_context_against_name(
        draw, name_box, teams[0], 62, 408, 16, 10,
        colors["ion_blue"], centered=True,
    )

    for idx, box, accent in (
        (1, (452, 122, 748, 252), colors["ion_blue"]),
        (2, (452, 272, 748, 402), colors["ion_blue"]),
    ):
        team = teams[idx]
        _panel(
            draw, box, accent=accent,
            radius=SOCIAL_STYLE["panel_radii"]["support"],
        )
        x1, y1, x2, y2 = box
        # Keep the #2/#3 logo plate optically locked to its card: the 100px
        # plate has identical 15px clearance above and below.
        logo_box = (x1 + 90, y1 + 15, x1 + 170, y2 - 15)
        _draw_rank(draw, (x1 + 18, y1 + 8), _rank_label(team), sizes["pod_rank"],
                   colors["ion_blue"], logo_box)
        _place_logo(image, team.team, logo_dir, logo_box, fallback_size=39)
        _draw_team_text_block(
            draw, (x1 + 178, y1 + 12, x2 - 10, y2 - 4), team,
            sizes["pod_name"], 17, sizes["points"], 10,
            colors["white"], colors["soft_mint"], centered=False,
        )
    _place_brand_mark(image, "16x9")

    rank_column = SOCIAL_STYLE["landscape_rank_column"]
    row_top = int(rank_column["top"])
    row_bottom = int(rank_column["bottom"])
    row_gap = int(rank_column["gap"])
    row_panel_height = (row_bottom - row_top - row_gap * 6) // 7
    row_height = row_panel_height + row_gap
    for offset, team in enumerate(teams[3:]):
        y1 = row_top + offset * row_height
        _rank_row(
            draw, image, team, (774, y1, 1240, y1 + row_panel_height),
            logo_dir, "16x9", reference_label,
        )


def _rank_row(draw, image, team, box, logo_dir, variant, reference_label) -> None:
    colors = SOCIAL_STYLE["colors"]
    sizes = SOCIAL_STYLE["font_sizes"][variant]
    _panel(
        draw, box, accent=colors["signal_orange"],
        radius=SOCIAL_STYLE["panel_radii"]["row"], alpha=235,
    )
    x1, y1, x2, y2 = box
    logo_left = x1 + (130 if variant == "4x5" else 105)
    logo_width = 86 if variant == "4x5" else 68
    logo_box = (logo_left, y1 + 5, logo_left + logo_width, y2 - 5)
    rank_geometry = SOCIAL_STYLE["lower_rank_box"]
    rank_left = x1 + int(rank_geometry["left_inset"])
    rank_right = logo_left - int(rank_geometry["logo_gap"])
    _draw_rank(
        draw, ((rank_left + rank_right) / 2, (y1 + y2) // 2),
        _rank_label(team), sizes["row_rank"], colors["signal_orange"],
        logo_box, anchor="mm",
    )
    _place_logo(image, team.team, logo_dir, logo_box, fallback_size=24)
    text_left = logo_left + logo_width + (20 if variant == "4x5" else 13)
    badge = noteworthy_badge(team, reference_label=reference_label)
    text_right = x2 - 18
    inner_pad = 3 if variant == "4x5" else 4
    _draw_team_text_block(
        draw, (text_left, y1 + inner_pad, text_right, y2 - inner_pad), team,
        sizes["row_name"], 20 if variant == "4x5" else 17,
        sizes["points"], 9, colors["white"], colors["soft_mint"], centered=False,
        reserve_second_context_line=True,
        fixed_line_slots=True,
        optical_y=int(SOCIAL_STYLE["lower_text_optical_y"]),
    )
    if badge:
        _draw_discrepancy_stamp(image, badge, box, variant=variant)


def _panel(draw, box, *, accent, radius, alpha=205) -> None:
    colors = SOCIAL_STYLE["colors"]
    panel = SOCIAL_STYLE["panel_fill"]
    fill = (*panel[:3], min(alpha, panel[3]))
    draw.rounded_rectangle(box, radius=radius, fill=fill,
                           outline=_rgba(colors["ion_blue"], 75), width=2)
    draw.polygon(
        _panel_accent_polygon(
            box, radius=radius, width=int(SOCIAL_STYLE["panel_accent_width"]),
        ),
        fill=accent,
    )


def _panel_accent_polygon(box, *, radius, width) -> list[tuple[float, float]]:
    """Return a left-edge accent clipped exactly to a rounded panel silhouette."""
    x1, y1, x2, y2 = map(float, box)
    radius = min(float(radius), (x2 - x1) / 2, (y2 - y1) / 2)
    width = min(float(width), radius)
    arc_delta = sqrt(max(0.0, 2 * radius * width - width * width))
    top_center_y = y1 + radius
    bottom_center_y = y2 - radius
    top = top_center_y - arc_delta
    bottom = bottom_center_y + arc_delta
    boundary = []
    samples = max(8, round(bottom - top))
    for index in range(samples + 1):
        y = bottom - (bottom - top) * index / samples
        if y > bottom_center_y:
            x = x1 + radius - sqrt(
                max(0.0, radius * radius - (y - bottom_center_y) ** 2)
            )
        elif y < top_center_y:
            x = x1 + radius - sqrt(
                max(0.0, radius * radius - (y - top_center_y) ** 2)
            )
        else:
            x = x1
        boundary.append((x, y))
    return [(x1 + width, top), (x1 + width, bottom), *boundary]


def validate_no_collision(
    first_name: str,
    first_box: tuple[int, int, int, int],
    second_name: str,
    second_box: tuple[int, int, int, int],
    *,
    gap: int = 10,
) -> None:
    """Fail rendering when two reserved layout regions collide or crowd."""
    ax1, ay1, ax2, ay2 = first_box
    bx1, by1, bx2, by2 = second_box
    separated = (
        ax2 + gap <= bx1 or bx2 + gap <= ax1
        or ay2 + gap <= by1 or by2 + gap <= ay1
    )
    if not separated:
        raise ValueError(
            f"Social layout collision: {first_name} {first_box} and "
            f"{second_name} {second_box} require a {gap}px gap."
        )


def _draw_rank(draw, position, text, size, fill, logo_box, *, anchor=None) -> None:
    """Draw every rank through the same rank/logo collision gate."""
    font = _font(size, role="rank")
    if text.startswith("T-") and text[2:].isdigit() and anchor in (None, "lm", "mm"):
        x, y = position
        tie_number = text[2:]
        letter_advance = draw.textlength("T", font=font)
        gap = float(size) * float(SOCIAL_STYLE["tie_hyphen_gap_ratio"])
        dash_width = float(size) * float(SOCIAL_STYLE["tie_hyphen_width_ratio"])
        number_advance = draw.textlength(tie_number, font=font)
        total_width = letter_advance + gap + dash_width + gap + number_advance
        component_anchor = "lm" if anchor == "mm" else anchor
        start_x = x - total_width / 2 if anchor == "mm" else x
        number_x = start_x + letter_advance + gap + dash_width + gap
        letter_box = draw.textbbox(
            (start_x, y), "T", font=font, anchor=component_anchor,
        )
        number_box = draw.textbbox(
            (number_x, y), tie_number, font=font, anchor=component_anchor,
        )
        dash_x1 = start_x + letter_advance + gap
        dash_x2 = dash_x1 + dash_width
        dash_y = (letter_box[1] + letter_box[3]) / 2
        rank_box = (
            min(letter_box[0], dash_x1, number_box[0]),
            min(letter_box[1], dash_y, number_box[1]),
            max(letter_box[2], dash_x2, number_box[2]),
            max(letter_box[3], dash_y, number_box[3]),
        )
        validate_no_collision("rank", rank_box, "logo", logo_box)
        draw.text(
            (start_x, y), "T", font=font, fill=fill, anchor=component_anchor,
        )
        draw.line(
            (dash_x1, dash_y, dash_x2, dash_y),
            fill=fill,
            width=max(2, round(float(size) * 0.06)),
        )
        draw.text(
            (number_x, y), tie_number, font=font, fill=fill,
            anchor=component_anchor,
        )
        return
    rank_box = draw.textbbox(position, text, font=font, anchor=anchor)
    validate_no_collision("rank", rank_box, "logo", logo_box)
    draw.text(position, text, font=font, fill=fill, anchor=anchor)


def _place_logo(image, team, logo_dir, box, *, fallback_size) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    x1, y1, x2, y2 = map(int, box)
    pad = max(4, int(min(x2 - x1, y2 - y1) * .05))
    path = resolve_team_logo_path(team, logo_dir)
    if path is None:
        _draw_logo_plate(draw, (x1, y1, x2, y2), light=False)
        initials = "".join(part[0] for part in str(team).split()[:3]).upper()
        draw.text(((x1 + x2) // 2, (y1 + y2) // 2), initials,
                  font=_font(fallback_size, bold=True), fill=SOCIAL_STYLE["colors"]["polar_mist"], anchor="mm")
        return
    array = load_team_logo_image(path)
    _draw_logo_plate(draw, (x1, y1, x2, y2), light=logo_needs_light_plate(array))
    logo = _array_to_pil(array)
    target = (max(1, x2 - x1 - 2 * pad), max(1, y2 - y1 - 2 * pad))
    scale = min(target[0] / max(logo.width, 1), target[1] / max(logo.height, 1))
    scale *= float(SOCIAL_STYLE["logo_scale_overrides"].get(str(team), 1.0))
    logo = logo.resize(
        (max(1, round(logo.width * scale)), max(1, round(logo.height * scale))),
        Image.Resampling.LANCZOS,
    )
    px = x1 + (x2 - x1 - logo.width) // 2
    py = y1 + (y2 - y1 - logo.height) // 2
    image.paste(logo, (px, py), logo)


def _place_brand_mark(image: Image.Image, variant: str) -> None:
    """Place the official white TDNet mark using variant-specific geometry."""
    settings = SOCIAL_STYLE["brand_mark"]
    repository_root = Path(__file__).resolve().parents[3]
    path = repository_root / str(settings["path"])
    if not path.exists():
        return
    with Image.open(path) as source:
        mark = source.convert("RGBA")
    alpha = mark.getchannel("A")
    visible_box = alpha.getbbox()
    if visible_box is None:
        return
    mark = mark.crop(visible_box)
    alpha = mark.getchannel("A").point(
        lambda value: round(value * int(settings["opacity"]) / 255)
    )
    white_mark = Image.new("RGBA", mark.size, (255, 255, 255, 0))
    white_mark.putalpha(alpha)
    x1, y1, x2, y2 = map(int, settings["boxes"][variant])
    scale = min((x2 - x1) / white_mark.width, (y2 - y1) / white_mark.height)
    scale *= float(settings["scale"][variant])
    white_mark = white_mark.resize(
        (round(white_mark.width * scale), round(white_mark.height * scale)),
        Image.Resampling.LANCZOS,
    )
    white_mark = white_mark.rotate(
        float(settings["angle"]), resample=Image.Resampling.BICUBIC, expand=True,
    )
    px = x1 + (x2 - x1 - white_mark.width) // 2
    py = y1 + (y2 - y1 - white_mark.height) // 2
    image.paste(white_mark, (px, py), white_mark)


def _draw_logo_plate(draw, box, *, light: bool) -> None:
    x1, y1, x2, y2 = box
    fill = (255, 255, 255, 242) if light else (255, 255, 255, 22)
    outline = (230, 233, 237, 185) if light else (230, 233, 237, 42)
    draw.rounded_rectangle(box, radius=min(24, (y2 - y1) // 5),
                           fill=fill, outline=outline, width=2)


def logo_needs_light_plate(array: np.ndarray) -> bool:
    """Detect marks whose visible pixels lose contrast on midnight blue."""
    values = np.asarray(array)
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    rgb = values[:, :, :3].astype(float)
    if not np.issubdtype(values.dtype, np.floating) and rgb.max(initial=0) > 1:
        rgb /= 255.0
    alpha = values[:, :, 3].astype(float) if values.shape[2] >= 4 else np.ones(values.shape[:2])
    if alpha.max(initial=0) > 1:
        alpha /= 255.0
    visible_rgb = rgb[alpha > 0.08]
    if not len(visible_rgb):
        return False
    luminance = (
        0.2126 * visible_rgb[:, 0]
        + 0.7152 * visible_rgb[:, 1]
        + 0.0722 * visible_rgb[:, 2]
    )
    # A substantial dark-pixel share indicates lettering or a mark edge that
    # will disappear into Midnight Gridiron. The light plate preserves it.
    return bool(np.mean(luminance < 0.24) >= 0.28)


def _array_to_pil(array: np.ndarray) -> Image.Image:
    values = np.asarray(array)
    if np.issubdtype(values.dtype, np.floating):
        values = np.clip(values * 255.0, 0, 255).astype("uint8")
    else:
        values = values.astype("uint8")
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    if values.shape[2] == 3:
        alpha = np.full((*values.shape[:2], 1), 255, dtype="uint8")
        values = np.concatenate([values, alpha], axis=2)
    return Image.fromarray(values[:, :, :4], mode="RGBA")


def _draw_discrepancy_stamp(image, text, box, *, variant) -> None:
    """Overlay a translucent, rotated AP-gap stamp across a panel edge."""
    colors = SOCIAL_STYLE["colors"]
    stamp_scale = float(SOCIAL_STYLE["discrepancy_stamp_scale"])
    font = _font(
        round(SOCIAL_STYLE["font_sizes"][variant]["badge"] * stamp_scale),
        bold=True,
    )
    measure = ImageDraw.Draw(image, "RGBA")
    text_box = measure.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0] + round(
        (28 if variant == "4x5" else 22) * stamp_scale
    )
    height = text_box[3] - text_box[1] + round(
        (18 if variant == "4x5" else 14) * stamp_scale
    )
    stamp = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    stamp_draw = ImageDraw.Draw(stamp, "RGBA")
    stamp_draw.rounded_rectangle(
        (1, 1, width - 2, height - 2),
        radius=max(6, height // 5),
        fill=_rgba(colors["edge_pink"], 128),
        outline=_rgba(colors["edge_pink"], 235),
        width=2,
    )
    stamp_draw.text(
        (width // 2, height // 2), text, font=font,
        fill=_rgba(colors["white"], 245), anchor="mm",
    )
    rotated = stamp.rotate(
        float(SOCIAL_STYLE["discrepancy_stamp_angle"]),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    x1, y1, x2, _ = box
    # Straddle the upper-right panel boundary so the badge reads as an
    # editorial annotation, not another aligned data field.
    paste_x = x2 - rotated.width + int(
        SOCIAL_STYLE["discrepancy_stamp_right_offset"][variant]
    )
    paste_y = (
        y1 - rotated.height // 3 + int(SOCIAL_STYLE["discrepancy_stamp_y_offset"])
    )
    image.paste(rotated, (paste_x, paste_y), rotated)


def _fit_text(draw, box, text, preferred, minimum, fill, *, bold=False, anchor="lm") -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    size = preferred
    font = _font(size, bold=bold)
    while size > minimum and draw.textlength(text, font=font) > x2 - x1:
        size -= 1
        font = _font(size, bold=bold)
    rendered = text
    if draw.textlength(rendered, font=font) > x2 - x1:
        while rendered and draw.textlength(rendered + "…", font=font) > x2 - x1:
            rendered = rendered[:-1]
        rendered = rendered.rstrip() + "…"
    if anchor.endswith("m"):
        y = (y1 + y2) // 2
    else:
        y = y1
    x = (x1 + x2) // 2 if anchor.startswith("m") else x1
    draw.text((x, y), rendered, font=font, fill=fill, anchor=anchor)
    return draw.textbbox((x, y), rendered, font=font, anchor=anchor)


def _fit_font(draw, text, preferred, minimum, width):
    size = preferred
    font = _font(size, bold=True)
    while size > minimum and draw.textlength(text, font=font) > width:
        size -= 1
        font = _font(size, bold=True)
    return font


def _fit_rendered_text(draw, text, preferred, minimum, width):
    """Fit one bold line and apply an ellipsis only at its minimum size."""
    font = _fit_font(draw, text, preferred, minimum, width)
    rendered = text
    if draw.textlength(rendered, font=font) > width:
        while rendered and draw.textlength(rendered + "…", font=font) > width:
            rendered = rendered[:-1]
        rendered = rendered.rstrip() + "…"
    return rendered, font


def _draw_team_text_block(
    draw,
    box,
    team,
    name_preferred,
    name_minimum,
    context_preferred,
    context_minimum,
    name_fill,
    context_fill,
    *,
    centered,
    reserve_second_context_line=False,
    fixed_line_slots=False,
    optical_y=0,
) -> None:
    """Center a team/name/context stack with equal top and bottom padding."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    gap = int(SOCIAL_STYLE["context_line_gap"])
    name_size = int(name_preferred)
    context_size = int(context_preferred)

    def build_lines():
        rendered_name, name_font = _fit_rendered_text(
            draw, _display_name(team), name_size, name_minimum, width,
        )
        lines = [(rendered_name, name_font, name_fill)]
        context_lines = _context_lines(team)
        if reserve_second_context_line and len(context_lines) == 1:
            context_lines.append(None)
        detail_fill = _rgba(
            SOCIAL_STYLE["colors"]["medium_gray"], SOCIAL_STYLE["detail_text_alpha"],
        )
        for context_index, text in enumerate(context_lines):
            metric_text = text if text is not None else "0 FPV"
            lines.append(
                (
                    text,
                    _fit_font(
                        draw, metric_text, context_size, context_minimum, width,
                    ),
                    context_fill if context_index == 0 else detail_fill,
                    metric_text,
                )
            )
        lines[0] = (*lines[0], lines[0][0])
        if fixed_line_slots:
            ratios = SOCIAL_STYLE["fixed_line_height_ratio"]
            heights = [
                round(font.size * float(ratios["name" if index == 0 else "context"]))
                for index, (_, font, _, _) in enumerate(lines)
            ]
        else:
            heights = [
                draw.textbbox((0, 0), metric_text, font=font, anchor="lt")[3]
                for _, font, _, metric_text in lines
            ]
        return lines, heights, sum(heights) + gap * (len(lines) - 1)

    lines, heights, total_height = build_lines()
    while total_height > y2 - y1 and (
        name_size > name_minimum or context_size > context_minimum
    ):
        if name_size > name_minimum:
            name_size -= 1
        elif context_size > context_minimum:
            context_size -= 1
        lines, heights, total_height = build_lines()

    y = y1 + max(0, (y2 - y1 - total_height) // 2) + int(optical_y)
    x = (x1 + x2) // 2 if centered else x1
    anchor = "mt" if centered else "lt"
    for index, ((text, font, fill, _), height) in enumerate(zip(lines, heights)):
        if text is not None:
            draw.text((x, y), text, font=font, fill=fill, anchor=anchor)
        y += height + (gap if index < len(lines) - 1 else 0)


def _context_lines(team: SocialTeam) -> list[str]:
    lines = [_points_label(team)]
    if team.first_place_votes:
        lines.append(f"{team.first_place_votes} FPV")
    elif team.best_rank is not None and team.social_rank - team.best_rank >= 3:
        lines.append(f"BEST: #{team.best_rank}")
    elif (
        team.top25_votes is not None
        and team.ballots_seen
        and team.top25_votes / team.ballots_seen
        < float(SOCIAL_STYLE["weak_ballot_support_ratio"])
    ):
        lines.append(f"ON {team.top25_votes}/{team.ballots_seen} BALLOTS")
    return lines


def _draw_context_against_name(
    draw,
    name_box,
    team,
    left,
    right,
    preferred,
    minimum,
    fill,
    *,
    centered,
):
    """Stack points and FPV against the name using the shared line gap."""
    x = (left + right) // 2 if centered else left
    anchor = "mt" if centered else "lt"
    line_gap = int(SOCIAL_STYLE["context_line_gap"])
    y = name_box[3] + line_gap
    boxes = []
    lines = _context_lines(team)
    for index, line in enumerate(lines):
        font = _fit_font(draw, line, preferred, minimum, right - left)
        position = (x, y)
        line_fill = (
            fill if index == 0
            else _rgba(
                SOCIAL_STYLE["colors"]["medium_gray"],
                SOCIAL_STYLE["detail_text_alpha"],
            )
        )
        draw.text(position, line, font=font, fill=line_fill, anchor=anchor)
        line_box = draw.textbbox(position, line, font=font, anchor=anchor)
        boxes.append(line_box)
        y = line_box[3] + (line_gap if index < len(lines) - 1 else 0)
    return (
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    )


def _points_label(team: SocialTeam) -> str:
    return f"{team.points:,} PTS" if team.points is not None else "POINTS N/A"


def _display_name(team: SocialTeam) -> str:
    return str(SOCIAL_STYLE["team_name_aliases"].get(team.team, team.team))


def _rank_label(team: SocialTeam) -> str:
    return f"T-{team.social_rank}" if team.tied else f"#{team.social_rank}"


def _draw_footer(draw, size, variant, generated_at_utc, git_commit, source_sha256) -> None:
    colors = SOCIAL_STYLE["colors"]
    font = _font(SOCIAL_STYLE["font_sizes"][variant]["footer"])
    margin = SOCIAL_STYLE["safe_margin"][variant]
    y = size[1] - (28 if variant == "4x5" else 19)
    draw.text((margin, y), "TDNet • FULL TOP 25 IN THE WEEKLY REPORT", font=font,
              fill=_rgba(colors["polar_mist"], 155), anchor="lm")
    details = []
    if generated_at_utc:
        details.append(_human_generation_date(generated_at_utc))
    if git_commit:
        details.append(f"commit {str(git_commit)[:8]}")
    if source_sha256:
        details.append(f"poll {str(source_sha256)[:8]}")
    if details:
        draw.text((size[0] - margin, y), " • ".join(details), font=font,
                  fill=_rgba(colors["medium_gray"], 185), anchor="rm")


def _human_generation_date(value: str) -> str:
    """Turn an ISO generation timestamp into compact, human-readable metadata."""
    raw = str(value).strip()
    try:
        generated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    return f"{generated.strftime('%b').upper()} {generated.day}, {generated.year}"


def _font(size: int, *, bold: bool = False, role: str | None = None) -> ImageFont.FreeTypeFont:
    font_role = role or ("body_bold" if bold else "body_regular")
    candidates = SOCIAL_STYLE["fonts"][font_role]
    for filename in candidates:
        try:
            return ImageFont.truetype(filename, int(size))
        except OSError:
            continue
    raise OSError(f"No usable font found for TDNet social role '{font_role}': {candidates}")


def _rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4)) + (int(alpha),)
