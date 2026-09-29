"""Render a shareable rank-spread chart from the curated Top 25 ratings table."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from gridiron_ml.td_run.poll_viz import resolve_team_logo_path


ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = [
    ("tdnet", "TDNet", "#EF5C9C"),
    ("sp", "SP+", "#138CCC"),
    ("fpi", "FPI", "#E88B37"),
    ("fei", "FEI", "#8064C8"),
    ("sag", "Sagarin", "#20A57E"),
    ("srs", "SRS", "#D4505C"),
    ("elo", "Elo", "#B98A20"),
    ("massey", "Massey", "#4A879E"),
]
NAVY = "#11214F"
BLUE = "#1EA7FF"
INK = "#243257"
MUTED = "#68788F"
BG = "#F6F8FC"
OUTLIER_SPOTS = 15


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_dir = ROOT / "data/fonts/aptos"
    names = ("Aptos-Bold.ttf", "Aptos-Display-Bold.ttf") if bold else (
        "Aptos.ttf", "Aptos-Regular.ttf"
    )
    for name in names:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    fallback = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{fallback}", size)


def _center(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, font, fill) -> None:
    draw.text((x, y), label, font=font, fill=fill, anchor="mm")


def _logo(image: Image.Image, team: str, box: tuple[int, int, int, int], logo_dir: Path) -> None:
    path = resolve_team_logo_path(team, logo_dir)
    if path is None:
        raise FileNotFoundError(f"No logo for {team}")
    mark = Image.open(path).convert("RGBA")
    visible = mark.getchannel("A").getbbox()
    if visible:
        mark = mark.crop(visible)
    x0, y0, x1, y1 = box
    mark.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    x = x0 + (x1 - x0 - mark.width) // 2
    y = y0 + (y1 - y0 - mark.height) // 2
    image.paste(mark, (x, y), mark)


def render(input_path: Path, output_path: Path, logo_dir: Path) -> dict:
    ratings = pd.read_csv(input_path)
    required = {"rank", "team", "average_rank"} | {
        f"{key}_rank" for key, _, _ in SYSTEMS
    }
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings table lacks {sorted(missing)}")
    if len(ratings) != 25 or not ratings["team"].is_unique:
        raise ValueError("Expected 25 unique teams in composite order")
    if ratings["rank"].astype(int).tolist() != list(range(1, 26)):
        raise ValueError("Composite rows must be ranked 1–25")

    system_keys = [key for key, _, _ in SYSTEMS]
    for key in ["average", *system_keys]:
        ranks = pd.to_numeric(ratings[f"{key}_rank"], errors="raise")
        if not ranks.between(1, 138).all():
            raise ValueError(f"Invalid {key} ranks")
    color_by_key = {key: color for key, _, color in SYSTEMS}
    label_by_key = {key: label for key, label, _ in SYSTEMS}
    all_deltas = pd.DataFrame(
        {
            key: ratings[f"{key}_rank"].astype(int)
            - ratings["average_rank"].astype(int)
            for key in system_keys
        }
    )
    average_gaps = all_deltas.abs().mean().sort_values(ascending=False)

    width, height = 2600, 2280
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = _font(67, bold=True)
    subtitle_font = _font(29)
    label_font = _font(26, bold=True)
    row_font = _font(28, bold=True)
    small_font = _font(22)
    number_font = _font(25, bold=True)

    # TDNet masthead and reading key.
    draw.rectangle((0, 0, width, 218), fill=NAVY)
    draw.rectangle((0, 208, width, 218), fill=BLUE)
    draw.text((70, 33), "WHERE THE RATINGS DISAGREE", font=title_font, fill="white")
    draw.text(
        (73, 126),
        "2026 WEEK 5  •  THROUGH WEEK 4  •  RANKS AMONG 138 FBS TEAMS",
        font=subtitle_font,
        fill="#B8DFFF",
    )
    draw.text(
        (72, 244),
        "Each row shows eight systems. The dark tick is the composite rank; the gray line spans the system ranks.",
        font=subtitle_font,
        fill=INK,
    )
    draw.text(
        (72, 284),
        f"Pink callouts mark a system at least {OUTLIER_SPOTS} rank spots from the composite. A positive gap means a lower rating.",
        font=small_font,
        fill=MUTED,
    )

    legend_y = 367
    for index, (key, label, color) in enumerate(SYSTEMS):
        x = 315 + index * 280
        draw.ellipse((x - 12, legend_y - 12, x + 12, legend_y + 12), fill=color)
        draw.text((x + 22, legend_y), label, font=label_font, fill=INK, anchor="lm")
    draw.line((72, 412, 2528, 412), fill="#CBD5E3", width=2)

    x0, x1 = 605, 1940
    first_y, row_height = 503, 52
    last_y = first_y + 24 * row_height
    left_tick, right_tick = 1, 60

    def rank_x(rank: int) -> int:
        return round(x0 + (int(rank) - left_tick) * (x1 - x0) / (right_tick - left_tick))

    draw.text((70, 432), "COMPOSITE / TEAM", font=label_font, fill=NAVY)
    _center(draw, (x0 + x1) // 2, 445, "SYSTEM RANK", label_font, NAVY)
    draw.text((2050, 432), "LARGEST GAP", font=label_font, fill=NAVY)
    for index in range(len(ratings)):
        if index % 2 == 0:
            y = first_y + index * row_height
            draw.rounded_rectangle((55, y - 24, 2535, y + 24), 10, fill="white")
    for tick in [1, 10, 20, 30, 40, 50, 60]:
        x = rank_x(tick)
        draw.line((x, 481, x, last_y + 30), fill="#D8E0EB", width=2)
        _center(draw, x, 472, f"#{tick}", small_font, MUTED)

    team_outliers = []
    for index, row in ratings.iterrows():
        y = first_y + index * row_height
        composite = int(row["average_rank"])
        _center(draw, 83, y, str(composite), row_font, NAVY)
        _logo(image, row["team"], (112, y - 21, 160, y + 21), logo_dir)
        team_name = "FIU" if row["team"] == "Florida International" else row["team"]
        draw.text((181, y), team_name, font=row_font, fill=NAVY, anchor="lm")

        ranks = {key: int(row[f"{key}_rank"]) for key in system_keys}
        draw.line(
            (rank_x(min(ranks.values())), y, rank_x(max(ranks.values())), y),
            fill="#AFBED2", width=6,
        )
        # Draw the composite before the dots, leaving its top diamond visible.
        cx = rank_x(composite)
        draw.line((cx, y - 20, cx, y + 20), fill=NAVY, width=4)
        draw.polygon(
            [(cx, y - 24), (cx + 7, y - 17), (cx, y - 10), (cx - 7, y - 17)],
            fill=NAVY,
        )
        for rank in sorted(set(ranks.values())):
            sharing = [key for key in system_keys if ranks[key] == rank]
            for stack_index, key in enumerate(sharing):
                offset = (stack_index - (len(sharing) - 1) / 2) * 12
                px, py = rank_x(rank), round(y + offset)
                radius = 8
                draw.ellipse(
                    (px - radius - 2, py - radius - 2, px + radius + 2, py + radius + 2),
                    fill="white",
                )
                draw.ellipse(
                    (px - radius, py - radius, px + radius, py + radius),
                    fill=color_by_key[key],
                )

        deltas = {key: ranks[key] - composite for key in system_keys}
        biggest = max(system_keys, key=lambda key: (abs(deltas[key]), -system_keys.index(key)))
        gap = deltas[biggest]
        is_outlier = abs(gap) >= OUTLIER_SPOTS
        if is_outlier:
            draw.rounded_rectangle(
                (2026, y - 22, 2524, y + 22), radius=11, fill="#FCE7F0"
            )
            team_outliers.append(
                {"team": row["team"], "system": label_by_key[biggest],
                 "system_rank": ranks[biggest], "composite_rank": composite,
                 "rank_gap": gap}
            )
        draw.ellipse((2045, y - 9, 2063, y + 9), fill=color_by_key[biggest])
        draw.text(
            (2080, y), f"{label_by_key[biggest]} #{ranks[biggest]}",
            font=number_font, fill=INK, anchor="lm",
        )
        draw.text(
            (2500, y), f"{gap:+d}", font=number_font,
            fill="#A72D62" if is_outlier else MUTED, anchor="rm",
        )

    # System-level disagreement is the mean absolute rank gap over these 25 teams.
    summary_y = 1844
    draw.line((55, summary_y, 2545, summary_y), fill=NAVY, width=3)
    draw.text((72, summary_y + 17), "WHICH SYSTEMS DEPART MOST?", font=_font(35, bold=True), fill=NAVY)
    draw.text(
        (73, summary_y + 68),
        "Mean absolute rank gap from the composite, across the 25 teams above",
        font=small_font, fill=MUTED,
    )
    bar_top, bar_bottom = 1957, 2160
    for index, key in enumerate(average_gaps.index):
        value = float(average_gaps[key])
        center_x = 647 + index * 252
        bar_height = round(value / 10 * 142)
        draw.rounded_rectangle(
            (center_x - 67, bar_bottom - bar_height, center_x + 67, bar_bottom),
            radius=12, fill=color_by_key[key],
        )
        _center(draw, center_x, bar_bottom - bar_height - 25, f"{value:.1f}", number_font, INK)
        _center(draw, center_x, bar_bottom + 27, label_by_key[key], label_font, INK)
    draw.text(
        (72, 2240),
        "A rank gap measures disagreement with this composite; it does not measure prediction accuracy.",
        font=small_font, fill=MUTED,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)
    manifest = {
        "input": str(input_path.relative_to(ROOT)),
        "input_sha256": sha256(input_path.read_bytes()).hexdigest(),
        "figure": str(output_path.relative_to(ROOT)),
        "figure_sha256": sha256(output_path.read_bytes()).hexdigest(),
        "rank_scope": "All 138 FBS teams; ties use minimum competition rank",
        "rank_gap": "system rank minus composite rank; positive means system rates team lower",
        "outlier_threshold_rank_spots": OUTLIER_SPOTS,
        "system_average_absolute_rank_gap_top25": {
            key: round(float(value), 2) for key, value in average_gaps.items()
        },
        "highlighted_team_system_outliers": team_outliers,
    }
    manifest_path = output_path.parent.parent / "metadata/rating_rank_disagreement_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "publication/2026/week_05/pre_game/tables/ratings_comparison_top25.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "publication/2026/week_05/pre_game/figures/week_05_rating_rank_disagreement.png",
    )
    parser.add_argument("--logo-dir", type=Path, default=ROOT / "data/meta/logos/by_team")
    args = parser.parse_args()
    result = render(args.input.resolve(), args.output.resolve(), args.logo_dir.resolve())
    print(json.dumps({"figure": result["figure"], "outliers": len(result["highlighted_team_system_outliers"])}))


if __name__ == "__main__":
    main()
