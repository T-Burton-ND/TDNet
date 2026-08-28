"""src.gridiron_ml.td_run.poll_viz.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Evaluate model outputs, compare predictions to market baselines, and build reporting artifacts.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .season_vs_vegas import color_for_source, load_eval_config, load_plot_color_map


def build_weekly_poll_outputs(
    *,
    evaluator,
    models,
    season,
    weeks=range(1, 17),
    top_n=25,
    average_scope="season",
    output_dir=None,
    logo_dir=None,
    eval_config=None,
    manual_ballots=None,
    merge_existing=False,
):
    """Build poll tables for selected weeks, optionally merging into existing outputs."""
    weeks = [int(week) for week in weeks]
    weekly_frames = []
    ballot_frames = []
    skip_rows = []

    for week in weeks:
        try:
            poll_kwargs = {
                "models": models,
                "season": season,
                "week": week,
                "average_scope": average_scope,
                "top_n": top_n,
            }
            if manual_ballots is not None:
                poll_kwargs["manual_ballots"] = manual_ballots
            poll_df = evaluator.poll(**poll_kwargs)
        except Exception as exc:
            skip_rows.append({"season": season, "week": week, "reason": str(exc)})
            continue

        poll_df = poll_df.head(top_n).copy()
        poll_df.insert(0, "week", int(week))
        poll_df.insert(0, "season", int(season))
        weekly_frames.append(poll_df)

        ballots = evaluator.poll_ballots_.copy()
        if not ballots.empty:
            ballots.insert(0, "week", int(week))
            ballots.insert(0, "season", int(season))
            ballot_frames.append(ballots)

    weekly_poll = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    weekly_ballots = pd.concat(ballot_frames, ignore_index=True) if ballot_frames else pd.DataFrame()
    skipped_weeks = pd.DataFrame(skip_rows)

    if merge_existing and output_dir is not None:
        output_dir = Path(output_dir)
        tables_dir = output_dir / "tables"
        weekly_poll = merge_existing_weekly_table(
            tables_dir / "weekly_poll_top25.csv",
            weekly_poll,
            season=season,
            weeks=weeks,
            sort_cols=["season", "week", "rank"],
        )
        weekly_ballots = merge_existing_weekly_table(
            tables_dir / "weekly_poll_ballots.csv",
            weekly_ballots,
            season=season,
            weeks=weeks,
            sort_cols=["season", "week", "ballot_model", "ballot_rank"],
        )
        skipped_weeks = merge_existing_weekly_table(
            tables_dir / "weekly_poll_skipped_weeks.csv",
            skipped_weeks,
            season=season,
            weeks=weeks,
            sort_cols=["season", "week"],
        )

    tables = {
        "weekly_poll_top25": weekly_poll,
        "weekly_poll_ballots": weekly_ballots,
        "weekly_poll_skipped_weeks": skipped_weeks,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        tables_dir = output_dir / "tables"
        plots_dir = output_dir / "plots"
        tables_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(tables_dir / f"{name}.csv", index=False)
        plot_weekly_top25_table(
            weekly_poll,
            plots_dir / "weekly_poll_top25_table.png",
            top_n=top_n,
            logo_dir=logo_dir,
            eval_config=eval_config,
        )
        if not weekly_ballots.empty:
            for week in sorted(pd.to_numeric(weekly_ballots["week"], errors="coerce").dropna().astype(int).unique()):
                plot_ballot_logo_grid(
                    weekly_ballots.loc[pd.to_numeric(weekly_ballots["week"], errors="coerce") == week],
                    plots_dir / f"weekly_poll_ballot_grid_week_{week:02d}.png",
                    top_n=top_n,
                    logo_dir=logo_dir,
                    eval_config=eval_config,
                    title=f"Week {week} Model Ballots",
                )

    return tables


def merge_existing_weekly_table(path, new_table, season, weeks, sort_cols=None):
    """Replace selected season/weeks in an existing weekly table and append new rows."""
    new_table = pd.DataFrame(new_table).copy()
    path = Path(path)
    if not path.exists():
        return sort_weekly_table(new_table, sort_cols)

    try:
        existing = pd.read_csv(path)
    except EmptyDataError:
        return sort_weekly_table(new_table, sort_cols)
    if existing.empty:
        return sort_weekly_table(new_table, sort_cols)
    if "season" not in existing.columns or "week" not in existing.columns:
        return sort_weekly_table(new_table, sort_cols)

    existing_season = pd.to_numeric(existing["season"], errors="coerce")
    existing_week = pd.to_numeric(existing["week"], errors="coerce")
    replace_weeks = {int(week) for week in weeks}
    keep = ~((existing_season == int(season)) & existing_week.isin(replace_weeks))
    frames = [existing.loc[keep].copy()]
    if not new_table.empty:
        frames.append(new_table)
    merged = pd.concat(frames, ignore_index=True, sort=False) if frames else new_table
    return sort_weekly_table(merged, sort_cols)


def sort_weekly_table(table, sort_cols=None):
    """Sort a weekly poll table while tolerating missing columns."""
    table = pd.DataFrame(table).copy()
    if table.empty:
        return table
    cols = [col for col in (sort_cols or []) if col in table.columns]
    if not cols:
        return table.reset_index(drop=True)
    return table.sort_values(cols, kind="stable").reset_index(drop=True)


def plot_weekly_top25_table(weekly_poll, path, top_n=25, logo_dir=None, eval_config=None):
    """Run the plot_weekly_top25_table step and return its normalized result."""
    if weekly_poll is None or weekly_poll.empty:
        return None

    import matplotlib.pyplot as plt

    eval_cfg = load_eval_config(eval_config=eval_config)
    plot_cfg = dict(eval_cfg.get("plotting", {}))
    color_map = load_plot_color_map(plot_cfg)
    dpi = int(plot_cfg.get("dpi", 150))

    weeks = sorted(pd.to_numeric(weekly_poll["week"], errors="coerce").dropna().astype(int).unique())
    ranks = list(range(1, int(top_n) + 1))
    table = weekly_poll.pivot_table(index="rank", columns="week", values="keys_team", aggfunc="first")
    table = table.reindex(index=ranks, columns=weeks)

    fig_width = max(10, len(weeks) * 1.25)
    fig_height = max(9, top_n * 0.38)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, len(weeks))
    ax.set_ylim(0, top_n)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(weeks)) + 0.5)
    ax.set_xticklabels([f"W{week}" for week in weeks], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(top_n) + 0.5)
    ax.set_yticklabels([str(rank) for rank in ranks], fontsize=8)
    ax.set_ylabel("Rank")
    ax.set_title(f"Top {top_n} Poll by Week", pad=28)

    for x in range(len(weeks) + 1):
        ax.axvline(x, color="#DDE2E7", linewidth=0.8)
    for y in range(top_n + 1):
        ax.axhline(y, color="#DDE2E7", linewidth=0.8)

    week_lookup = {week: idx for idx, week in enumerate(weeks)}
    for rank in ranks:
        for week in weeks:
            team = table.loc[rank, week] if rank in table.index and week in table.columns else None
            if pd.isna(team):
                continue
            x = week_lookup[week] + 0.5
            y = rank - 0.5
            logo_path = resolve_team_logo_path(team, logo_dir)
            if logo_path is not None:
                draw_team_logo(ax, logo_path, x, y)
            else:
                ax.text(
                    x,
                    y,
                    abbreviate_team_name(team),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#0D0D0D",
                    clip_on=True,
                )

    add_poll_legend(ax, color_map)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ballot_logo_grid(ballots, path, top_n=25, logo_dir=None, eval_config=None, title=None):
    """Plot one row per ballot and one logo/name cell per rank."""
    if ballots is None or ballots.empty:
        return None

    import matplotlib.pyplot as plt

    frame = ballots.copy()
    frame["ballot_rank"] = pd.to_numeric(frame["ballot_rank"], errors="coerce")
    frame = frame.loc[frame["ballot_rank"].between(1, int(top_n), inclusive="both")].copy()
    if frame.empty:
        return None

    eval_cfg = load_eval_config(eval_config=eval_config)
    plot_cfg = dict(eval_cfg.get("plotting", {}))
    color_map = load_plot_color_map(plot_cfg)
    dpi = int(plot_cfg.get("dpi", 150))

    ballot_models = list(dict.fromkeys(frame["ballot_model"].astype(str).tolist()))
    ranks = list(range(1, int(top_n) + 1))
    table = frame.pivot_table(index="ballot_model", columns="ballot_rank", values="keys_team", aggfunc="first")
    table = table.reindex(index=ballot_models, columns=ranks)

    fig_width = max(18, top_n * 0.76)
    fig_height = max(5.0, len(ballot_models) * 0.78 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, top_n)
    ax.set_ylim(0, len(ballot_models))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(top_n) + 0.5)
    ax.set_xticklabels([str(rank) for rank in ranks], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(ballot_models)) + 0.5)
    ax.set_yticklabels([display_model_label(model) for model in ballot_models], fontsize=10.5)
    ax.set_title(title or f"Top {top_n} Ballots", fontsize=21, weight="bold", pad=34)

    for x in range(top_n + 1):
        ax.axvline(x, color="#DDE2E7", linewidth=0.7)
    for y in range(len(ballot_models) + 1):
        ax.axhline(y, color="#DDE2E7", linewidth=0.7)

    for y_idx, ballot_model in enumerate(ballot_models):
        row_color = color_for_source(ballot_model, color_map, y_idx)
        ax.axhspan(y_idx, y_idx + 1, color=row_color, alpha=0.045)
        for rank in ranks:
            team = table.loc[ballot_model, rank] if ballot_model in table.index and rank in table.columns else None
            if pd.isna(team):
                continue
            x = rank - 0.5
            y = y_idx + 0.5
            logo_path = resolve_team_logo_path(team, logo_dir)
            if logo_path is not None:
                # Keep the visible mark comfortably inside the rank cell at
                # export DPI; transparent source padding is cropped below.
                draw_team_logo(ax, logo_path, x, y, target_px=23)
            else:
                ax.text(
                    x,
                    y,
                    abbreviate_team_name(team, max_len=12),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#0D0D0D",
                    clip_on=True,
                )

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def resolve_team_logo_path(team, logo_dir):
    """Run the resolve_team_logo_path step and return its normalized result."""
    if logo_dir is None:
        return None
    logo_dir = Path(logo_dir)
    if not logo_dir.exists():
        return None
    stems = logo_slug_candidates(team)
    for stem in stems:
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            path = logo_dir / f"{stem}{ext}"
            if path.exists():
                return path
    return None


def draw_team_logo(ax, logo_path, x, y, target_px=24):
    """Draw a logo with a stable visible-mark size, ignoring canvas padding."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    image = load_team_logo_image(logo_path)
    height, width = image.shape[:2]
    target_px = min(float(target_px), 28.0)
    zoom = target_px / max(float(width), float(height), 1.0)
    # The target-size cap is a final guard against oversized source assets.
    box = OffsetImage(image, zoom=zoom)
    artist = AnnotationBbox(box, (x, y), frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(artist)


def load_team_logo_image(logo_path):
    """Load and crop transparent/background padding around a team logo.

    Source assets use radically different canvas sizes and whitespace. Scaling
    the raw canvas therefore makes the visible marks inconsistent. This helper
    finds the visible-pixel bounding box first, then every renderer scales that
    cropped mark to its requested target size.
    """
    import matplotlib.image as mpimg

    image = np.asarray(mpimg.imread(logo_path))
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    rgb = image[:, :, :3].astype(float)
    scale = 1.0 if np.issubdtype(image.dtype, np.floating) or rgb.max(initial=0) <= 1.0 else 255.0
    rgb = rgb / scale
    alpha = image[:, :, 3].astype(float) if image.shape[2] >= 4 else None
    if alpha is not None:
        alpha = alpha / (1.0 if alpha.max(initial=0) <= 1.0 else 255.0)
    # Prefer true transparency. For opaque canvases, estimate the background
    # from corner pixels and crop pixels sufficiently different from it.
    if alpha is not None and np.any(alpha < 0.98):
        visible = alpha > 0.02
    else:
        h, w = rgb.shape[:2]
        patch = max(1, min(h, w) // 20)
        corners = np.concatenate([
            rgb[:patch, :patch].reshape(-1, 3), rgb[:patch, -patch:].reshape(-1, 3),
            rgb[-patch:, :patch].reshape(-1, 3), rgb[-patch:, -patch:].reshape(-1, 3),
        ])
        background = np.median(corners, axis=0)
        visible = np.linalg.norm(rgb - background, axis=2) > 0.055
        if alpha is not None:
            visible &= alpha > 0.02
    if not visible.any():
        return image
    rows, columns = np.where(visible)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    return image[top:bottom, left:right]


def add_poll_legend(ax, color_map):
    """Run the add_poll_legend step and return its normalized result."""
    handles = []
    labels = []
    for idx, label in enumerate(["Team logo", "Team name fallback"]):
        handles.append(
            ax.scatter(
                [],
                [],
                color=color_for_source(label, color_map, idx),
                label=label,
                s=25,
            )
        )
        labels.append(label)
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=False,
        title="Legend",
    )


def display_model_label(model):
    """Return a figure-friendly model label without objective-only prefixes."""
    label = str(model)
    return re.sub(r"^margin_", "", label).replace("_", " ")


def abbreviate_team_name(team, max_len=16):
    """Run the abbreviate_team_name step and return its normalized result."""
    team = str(team)
    if len(team) <= max_len:
        return team
    words = team.split()
    if len(words) > 1:
        abbreviated = " ".join([word[0] + "." for word in words[:-1]] + [words[-1]])
        if len(abbreviated) <= max_len:
            return abbreviated
    return team[: max_len - 1] + "."


def normalize_team_slug(team):
    """Run the normalize_team_slug step and return its normalized result."""
    text = str(team).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def logo_slug_candidates(team):
    """Run the logo_slug_candidates step and return its normalized result."""
    primary = normalize_team_slug(team)
    candidates = [primary]
    raw = str(team).strip().lower()
    if "&" in raw:
        legacy = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        candidates.append(legacy)
    if "_and_" in primary:
        candidates.append(primary.replace("_and_", "_aand"))
    return list(dict.fromkeys(candidates))
