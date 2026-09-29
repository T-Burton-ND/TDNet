"""Build eight ready-to-schedule graphics and captions from public weekly data.

Run after the postgame and pregame publication jobs::

    python scripts/publication/build_social_media.py --season 2026 --week 5
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from math import isfinite
from pathlib import Path
import re
import shutil

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
NAVY = "#11214F"
BLUE = "#1EA7FF"
PINK = "#FF5FA2"
INK = "#243257"
MUTED = "#68788F"
BG = "#F6F8FC"
LOGOS = ROOT / "data/meta/logos/by_team"


def resolve_team_logo_path(team: str, logo_dir: Path) -> Path | None:
    """Use the publication logo filenames without importing the model stack."""
    raw = str(team).strip().lower()
    primary = re.sub(r"[^a-z0-9]+", "_", raw.replace("&", "and")).strip("_")
    candidates = [primary]
    if "&" in raw:
        candidates.append(re.sub(r"[^a-z0-9]+", "_", raw).strip("_"))
    if "_and_" in primary:
        candidates.append(primary.replace("_and_", "_aand"))
    for stem in dict.fromkeys(candidates):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            path = logo_dir / f"{stem}{ext}"
            if path.exists():
                return path
    return None


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Aptos-Bold.ttf" if bold else "Aptos.ttf"
    path = ROOT / "data/fonts/aptos" / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    fallback = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{fallback}", size)


def logo(canvas: Image.Image, team: str, x: int, y: int, size: int = 100) -> None:
    path = resolve_team_logo_path(team, LOGOS)
    if path is None:
        raise FileNotFoundError(f"Missing team logo: {team}")
    mark = Image.open(path).convert("RGBA")
    bbox = mark.getchannel("A").getbbox()
    if bbox:
        mark = mark.crop(bbox)
    mark.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas.paste(mark, (x + (size - mark.width) // 2, y + (size - mark.height) // 2), mark)


def fit_text(draw: ImageDraw.ImageDraw, value: str, max_width: int, size: int, *, bold: bool = False):
    while size >= 25:
        face = font(size, bold)
        if draw.textbbox((0, 0), value, font=face)[2] <= max_width:
            return face
        size -= 2
    raise ValueError(f"Label too wide for card: {value}")


def base_card(title: str, subtitle: str, footer: str, count: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    row_height = 138 if count == 10 else 243
    height = 365 + count * row_height
    canvas = Image.new("RGB", (1600, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1600, 245), fill=NAVY)
    draw.rectangle((0, 237, 1600, 245), fill=BLUE)
    draw.text((68, 48), title, font=fit_text(draw, title, 1460, 74, bold=True), fill="white")
    draw.text((71, 156), subtitle, font=font(34), fill="#B8DFFF")
    draw.text((70, height - 86), footer, font=fit_text(draw, footer, 1460, 28), fill=MUTED)
    return canvas, draw, row_height


def render_poll(path: Path, poll: pd.DataFrame, season: int, week: int) -> None:
    rows = poll.sort_values("rank").head(10)
    canvas, draw, step = base_card(
        "TDNET TOP 10", f"{season} WEEK {week}  |  MODEL POLL", "Ranked by the published 33-model TDNet ballot.", 10
    )
    for i, row in enumerate(rows.itertuples(index=False)):
        y = 265 + i * step
        draw.rounded_rectangle((55, y, 1545, y + 119), 16, fill="white")
        draw.text((90, y + 61), f"#{int(row.rank)}", font=font(50, True), fill=NAVY, anchor="lm")
        logo(canvas, row.keys_team, 205, y + 10, 98)
        draw.text((330, y + 61), row.keys_team, font=fit_text(draw, row.keys_team, 850, 51, bold=True), fill=INK, anchor="lm")
        draw.text((1490, y + 61), f"{int(row.poll_points)} pts", font=font(39, True), fill=BLUE, anchor="rm")
    canvas.save(path, optimize=True)


def render_games(path: Path, games: pd.DataFrame, title: str, subtitle: str) -> None:
    if not 1 <= len(games) <= 5:
        raise ValueError("A matchup card needs one to five games")
    canvas, draw, step = base_card(
        title, subtitle,
        "AP badge = published Top 25 rank | Vegas line under market favorite | TDNet is an estimate.",
        len(games),
    )

    def ap_rank(row: object, side: str) -> int | None:
        value = getattr(row, f"{side}_rank", None)
        if value is None:
            value = getattr(row, f"ap_rank_{side}", None)
        if value is None or pd.isna(value):
            return None
        rank = int(value)
        if not 1 <= rank <= 25:
            raise ValueError(f"Invalid published AP rank for game {row.game_id}: {rank}")
        return rank

    for i, row in enumerate(games.itertuples(index=False)):
        y = 267 + i * step
        draw.rounded_rectangle((55, y, 1545, y + 221), 18, fill="white")
        logo(canvas, row.away_team, 85, y + 35, 108)
        logo(canvas, row.home_team, 1378, y + 35, 108)
        away_rank, home_rank = ap_rank(row, "away"), ap_rank(row, "home")
        if away_rank is not None:
            draw.rounded_rectangle((197, y + 46, 294, y + 90), radius=18, fill=BLUE)
            draw.text((245, y + 68), f"AP #{away_rank}", font=font(22, True), fill="white", anchor="mm")
        if home_rank is not None:
            draw.rounded_rectangle((1282, y + 46, 1379, y + 90), radius=18, fill=BLUE)
            draw.text((1330, y + 68), f"AP #{home_rank}", font=font(22, True), fill="white", anchor="mm")
        away_text_x = 312 if away_rank is not None else 230
        home_text_x = 1262 if home_rank is not None else 1342
        draw.text((away_text_x, y + 67), row.away_team, font=fit_text(draw, row.away_team, 400, 42, bold=True), fill=NAVY, anchor="lm")
        draw.text((800, y + 70), "AT", font=font(35, True), fill=MUTED, anchor="mm")
        draw.text((home_text_x, y + 67), row.home_team, font=fit_text(draw, row.home_team, 400, 42, bold=True), fill=NAVY, anchor="rm")
        spread = float(row.vegas_spread_as_of_publish)
        if not isfinite(spread):
            raise ValueError(f"No published Vegas line for game {row.game_id}")
        if spread == 0:
            draw.text((800, y + 112), "(Vegas PK)", font=font(29), fill=MUTED, anchor="mm")
        else:
            market_favorite_is_away = spread > 0  # Published spread is from the home team's perspective.
            market_label = f"(Vegas -{abs(spread):.1f})"
            if market_favorite_is_away:
                draw.text((away_text_x, y + 112), market_label, font=font(29), fill=MUTED, anchor="lm")
            else:
                draw.text((home_text_x, y + 112), market_label, font=font(29), fill=MUTED, anchor="rm")
        margin = float(row.predicted_margin)
        pick = "Essentially even" if margin < 0.05 else f"{row.pred_winner} by {margin:.1f}"
        prediction = f"TDNet: {pick}"
        prediction_font = fit_text(draw, prediction, 930, 45, bold=True)
        text_width = draw.textbbox((0, 0), prediction, font=prediction_font)[2]
        half_width = (text_width + 76) // 2
        draw.rounded_rectangle(
            (800 - half_width, y + 130, 800 + half_width, y + 186),
            radius=26,
            fill=NAVY,
        )
        draw.text((800, y + 158), prediction, font=prediction_font, fill=PINK, anchor="mm")
        draw.text((800, y + 205), f"{float(row.model_agreement):.0%} model agreement", font=font(27), fill=MUTED, anchor="mm")
    canvas.save(path, optimize=True)


def render_poll_gaps(path: Path, poll: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    ranked = poll.dropna(subset=["reference_rank"]).copy()
    ranked["gap"] = ranked["rank"].astype(int) - ranked["reference_rank"].astype(int)
    ranked = ranked.assign(abs_gap=ranked["gap"].abs()).sort_values(["abs_gap", "rank"], ascending=[False, True]).head(5)
    canvas, draw, step = base_card(
        "TDNET VS AP POLL", f"{season} WEEK {week}  |  BIGGEST TOP 25 RANK GAPS",
        "Positive gap: TDNet ranks the team lower than AP. Poll ranks are not scores.", 5
    )
    for i, row in enumerate(ranked.itertuples(index=False)):
        y = 267 + i * step
        draw.rounded_rectangle((55, y, 1545, y + 221), 18, fill="white")
        logo(canvas, row.keys_team, 88, y + 53, 110)
        draw.text((230, y + 67), row.keys_team, font=fit_text(draw, row.keys_team, 700, 53, bold=True), fill=NAVY, anchor="lm")
        draw.text((230, y + 139), f"TDNet #{int(row.rank)}   |   AP #{int(row.reference_rank)}", font=font(38), fill=INK, anchor="lm")
        draw.text((1460, y + 112), f"{int(row.gap):+d}", font=font(73, True), fill=PINK if abs(row.gap) >= 6 else BLUE, anchor="rm")
    canvas.save(path, optimize=True)
    return ranked


def write_caption(path: Path, caption: str, alt: str, sources: list[Path]) -> None:
    if len(caption) > 280:
        raise ValueError(f"Caption exceeds 280 characters: {path.name}")
    source_lines = "\n".join(f"- `{source.relative_to(ROOT)}` (SHA-256 `{sha256(source.read_bytes()).hexdigest()}`)" for source in sources)
    path.write_text(f"# Caption\n\n{caption}\n\n## Image alt text\n\n{alt}\n\n## Sources\n\n{source_lines}\n", encoding="utf-8")


def build(season: int, week: int, output: Path) -> list[Path]:
    pre = ROOT / "publication" / str(season) / f"week_{week:02d}" / "pre_game"
    figures = pre / "figures"
    tables = pre / "tables"
    ratings = tables / "ratings_comparison_top25.csv"
    poll_path = tables / "tdnet_top25.csv"
    games_path = tables / "ap_top25_games.csv"
    closest_path = tables / "closest_games.csv"
    all_games_path = tables / "all_games.csv"
    source_figures = [figures / f"week_{week:02d}_ratings_comparison.png", figures / f"week_{week:02d}_rating_rank_disagreement.png"]
    for source in [poll_path, games_path, closest_path, all_games_path]:
        if not source.is_file():
            raise FileNotFoundError(source)
    rating_sources = [ratings, *source_figures]
    available_ratings = all(source.is_file() for source in rating_sources)
    if any(source.is_file() for source in rating_sources) and not available_ratings:
        raise ValueError("Ratings comparison is only partly published; complete it before building the pack")
    poll = pd.read_csv(poll_path)
    games = pd.read_csv(games_path)
    closest = pd.read_csv(closest_path)
    all_games = pd.read_csv(all_games_path)
    if len(poll) != 25 or len(games) < 5 or len(closest) < 5 or len(all_games) < 5:
        raise ValueError("Published tables are incomplete; run and check the weekly pipeline first")
    if available_ratings and len(pd.read_csv(ratings)) != 25:
        raise ValueError("Published ratings comparison is incomplete")
    if games["game_id"].duplicated().any():
        raise ValueError("Duplicate ranked game IDs")
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def add(stem: str, caption: str, alt: str, sources: list[Path], source_image: Path | None = None) -> Path:
        image = output / f"{stem}.png"
        if source_image is not None:
            shutil.copyfile(source_image, image)
        write_caption(output / f"{stem}.md", caption, alt, sources)
        created.append(image)
        return image

    if available_ratings:
        add("01_ratings_comparison", f"How does TDNet stack up against SP+, FPI, FEI, Sagarin, SRS, Elo and Massey? Week {week}'s Top 25, with each system's FBS rank beside its score. #CFB #CollegeFootball #CFBRankings", "Top 25 composite ratings table comparing eight standardized systems, with each score's rank in parentheses.", [ratings, source_figures[0]], source_figures[0])
        add("02_rating_rank_disagreement", f"Where do the rating systems disagree most? This Week {week} chart shows each Top 25 team's system ranks and flags the biggest gaps. #CFB #CollegeFootball #CFBRankings", "Rank spread chart for the composite Top 25. Colored dots represent eight systems; pink callouts mark large gaps.", [ratings, source_figures[1]], source_figures[1])
    poll_image = output / "03_tdnet_top10.png"
    render_poll(poll_image, poll, season, week)
    leaders = ", ".join(poll.sort_values("rank").head(3)["keys_team"])
    add("03_tdnet_top10", f"TDNet's Week {week} Top 10 starts with {leaders}. See how the 33-model ballot stacks up. #CFB #CollegeFootball #CFBRankings", "TDNet model poll Top 10 with team logos and poll points.", [poll_path])

    # Published AP-ranked game order is the weekly editorial order; preserve it.
    ranked_groups = min(3, (len(games) + 4) // 5)
    for part in range(ranked_groups):
        group = games.iloc[part * 5:(part + 1) * 5]
        stem = f"{4 + part:02d}_ranked_games_{part + 1}"
        render_games(output / f"{stem}.png", group, f"RANKED-GAME WATCH {part + 1}/{ranked_groups}", f"{season} WEEK {week}  |  TDNET PREDICTIONS")
        close = group.sort_values("predicted_margin").iloc[0]
        matchup = f"{close.away_team} at {close.home_team}"
        add(stem, f"{len(group)} Week {week} games involving AP-ranked teams. The tightest projection here: {matchup}, {close.pred_winner} by {float(close.predicted_margin):.1f}. #CFB #CollegeFootball #CFBPredictions", "Ranked-team games with team logos and AP rank badges, TDNet projected winner and margin, model agreement, and the published Vegas line beneath the market favorite.", [games_path])

    close5 = closest.head(5)
    render_games(output / "07_closest_games.png", close5, "FIVE GAMES ON THE EDGE", f"{season} WEEK {week}  |  SMALLEST PROJECTED MARGINS")
    add("07_closest_games", f"Week {week}'s five closest TDNet projections are all within {float(close5['predicted_margin'].max()):.1f} points. Which one goes down to the wire? #CFB #CollegeFootball #CFBPredictions", "Five games with the smallest TDNet projected margins, team logos and AP rank badges where applicable, model agreement, and the published Vegas line beneath the market favorite.", [closest_path])

    gaps = render_poll_gaps(output / "08_tdnet_vs_ap.png", poll, season, week)
    leader = gaps.iloc[0]
    add("08_tdnet_vs_ap", f"TDNet vs. AP: {leader.keys_team} is #{int(leader['rank'])} in our Week {week} poll and #{int(leader.reference_rank)} in AP. Here are the five biggest rank gaps. #CFB #CollegeFootball #CFBRankings", "Five largest rank differences between the TDNet Top 25 and AP poll, with team logos and both ranks.", [poll_path])

    split5 = all_games.sort_values(["model_agreement", "game_id"]).head(5)
    render_games(output / "09_model_splits.png", split5, "WHERE MODELS SPLIT", f"{season} WEEK {week}  |  LOWEST PICK AGREEMENT")
    add("09_model_splits", f"The TDNet models disagree most on these five Week {week} games. The closest vote is just {float(split5['model_agreement'].min()):.0%} for one side. #CFB #CollegeFootball #CFBPredictions", "Five games with the lowest model agreement, team logos and AP rank badges where applicable, TDNet projected margins, and the published Vegas line beneath the market favorite.", [all_games_path])

    index = [f"# {season} Week {week} social media pack", "", f"{len(created)} graphics with captions in matching `.md` files. Pick five or six to schedule on Monday. All figures use the published Week {week} snapshot; game predictions can become stale after kickoff.", "", "| Graphic | Topic |", "| --- | --- |"]
    topics = {"01_ratings_comparison": "Eight-system ratings table", "02_rating_rank_disagreement": "Rating-system rank disagreement", "03_tdnet_top10": "TDNet Top 10", "04_ranked_games_1": "Ranked games, part 1", "05_ranked_games_2": "Ranked games, part 2", "06_ranked_games_3": "Ranked games, part 3", "07_closest_games": "Closest projected games", "08_tdnet_vs_ap": "TDNet vs. AP poll gaps", "09_model_splits": "Games with the lowest model agreement"}
    for image in created:
        topic = topics[image.stem]
        index.append(f"| [{image.name}]({image.name}) | {topic}; [caption]({image.stem}.md) |")
    index.extend(["", "Blue AP badges beside team logos show the published AP Top 25 rank; unranked teams have no badge. The parenthetical Vegas line is the spread captured at publication, shown beneath the market favorite; it can differ from the TDNet pick. Review the image and caption before scheduling. Post game previews before the relevant kickoff; avoid sharing outdated predictions after results are known. Captions omit links to keep them short. Team logos come from the repository's logo set.", ""])
    (output / "README.md").write_text("\n".join(index), encoding="utf-8")
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or ROOT / "social_media" / str(args.season) / f"week_{args.week:02d}"
    for image in build(args.season, args.week, destination.resolve()):
        print(image)


if __name__ == "__main__":
    main()
