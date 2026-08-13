"""Canonical TDNet data-point catalog and logo scatter plots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.fingerprints import Fingerprints


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_POINTS_PATH = PROJECT_ROOT / "configs" / "data_points.yaml"


@dataclass(frozen=True)
class DataPoint:
    """One canonical, user-facing data point."""

    name: str
    column: str
    label: str
    group: str = "other"
    description: str = ""
    higher_is_better: bool | None = None


def load_data_point_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the canonical data-point YAML config."""

    path = Path(config_path or DEFAULT_DATA_POINTS_PATH).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_data_point_catalog(
    config_path: str | Path | None = None,
) -> dict[str, DataPoint]:
    """Load data points keyed by canonical name."""

    cfg = load_data_point_config(config_path)
    raw_points = dict(cfg.get("data_points", {}) or {})
    catalog: dict[str, DataPoint] = {}
    for name, raw in raw_points.items():
        if isinstance(raw, str):
            raw = {"column": raw}
        raw = dict(raw or {})
        column = str(raw.get("column") or name)
        catalog[str(name)] = DataPoint(
            name=str(name),
            column=column,
            label=str(raw.get("label") or humanize_column(column)),
            group=str(raw.get("group") or infer_group(column)),
            description=str(raw.get("description") or ""),
            higher_is_better=raw.get("higher_is_better"),
        )
    return catalog


def data_point_options(catalog: dict[str, DataPoint] | None = None) -> pd.DataFrame:
    """Return a compact table of available data points for notebooks."""

    catalog = catalog or load_data_point_catalog()
    rows = [
        {
            "name": point.name,
            "column": point.column,
            "label": point.label,
            "group": point.group,
            "higher_is_better": point.higher_is_better,
        }
        for point in catalog.values()
    ]
    return pd.DataFrame(rows).sort_values(["group", "name"]).reset_index(drop=True)


def print_data_point_options(
    catalog: dict[str, DataPoint] | None = None, max_rows: int | None = None
) -> None:
    """Print the editable names used by the logo scatter notebook."""

    options = data_point_options(catalog)
    if max_rows is not None:
        options = options.head(int(max_rows))
    print(options.to_string(index=False))


def resolve_data_point(
    value: str, catalog: dict[str, DataPoint] | None = None
) -> DataPoint:
    """Resolve a canonical name, raw column, or display label to a DataPoint."""

    catalog = catalog or load_data_point_catalog()
    if value in catalog:
        return catalog[value]

    needle = normalize_lookup_text(value)
    for point in catalog.values():
        if needle in {
            normalize_lookup_text(point.name),
            normalize_lookup_text(point.column),
            normalize_lookup_text(point.label),
        }:
            return point
    raise KeyError(
        f"Unknown data point {value!r}. Use print_data_point_options() to list choices."
    )


def team_snapshot_frame(
    *,
    season: int,
    week: int,
    root: str | Path | None = None,
    fingerprint_version: int = 0,
    postseason: bool = False,
    fbs_only: bool = True,
) -> pd.DataFrame:
    """Load one team row per FBS team for a season/week snapshot."""

    root_path = Path(root or PROJECT_ROOT).expanduser().resolve()
    frame = Fingerprints(
        version=fingerprint_version,
        postseason=postseason,
        root=root_path,
    ).frame(season=int(season), week=int(week))
    if "keys_team" not in frame.columns:
        raise ValueError("Fingerprint frame does not include keys_team.")

    frame = frame.sort_values("keys_team").drop_duplicates("keys_team", keep="first")
    if fbs_only:
        fbs = load_fbs_teams(root_path)
        if fbs:
            frame = frame.loc[frame["keys_team"].astype(str).isin(fbs)].copy()
    return frame.reset_index(drop=True)


def plot_data_point_logo_scatter(
    x_data_point: str,
    y_data_point: str,
    *,
    season: int,
    week: int,
    root: str | Path | None = None,
    fingerprint_version: int = 0,
    postseason: bool = False,
    frame: pd.DataFrame | None = None,
    catalog: dict[str, DataPoint] | None = None,
    config_path: str | Path | None = None,
    logo_dir: str | Path | None = None,
    palette_path: str | Path | None = None,
    style: str = "light",
    figsize: tuple[float, float] = (12.5, 9.0),
    logo_zoom: float = 0.045,
    logo_max_size: int = 96,
    fallback_marker_size: float = 42.0,
    title: str | None = None,
    subtitle: str | None = None,
    annotate_missing_logos: bool = True,
    annotate_all: bool = False,
    alpha: float = 0.92,
    ax=None,
    save_path: str | Path | None = None,
):
    """Plot any two canonical data points with team logos as markers."""

    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    root_path = Path(root or PROJECT_ROOT).expanduser().resolve()
    cfg = load_data_point_config(config_path)
    catalog = catalog or load_data_point_catalog(config_path)
    x_point = resolve_data_point(x_data_point, catalog)
    y_point = resolve_data_point(y_data_point, catalog)

    if frame is None:
        frame = team_snapshot_frame(
            season=season,
            week=week,
            root=root_path,
            fingerprint_version=fingerprint_version,
            postseason=postseason,
            fbs_only=True,
        )
    plot_df = prepare_plot_frame(frame, x_point, y_point)

    plotting_cfg = dict(cfg.get("plotting", {}) or {})
    palette = load_palette(
        palette_path or plotting_cfg.get("palette_path"),
        root=root_path,
    )
    colors = style_colors(style, palette)
    logo_lookup = load_logo_lookup(root_path, logo_dir or plotting_cfg.get("logo_dir"))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor(colors["background"])
    ax.grid(True, color=colors["grid"], linewidth=0.8, alpha=0.55)
    ax.axhline(0, color=colors["axis"], linewidth=0.8, alpha=0.35)
    ax.axvline(0, color=colors["axis"], linewidth=0.8, alpha=0.35)

    missing_logo_rows = []
    for _, row in plot_df.iterrows():
        team = str(row["keys_team"])
        logo_path = logo_lookup.get(normalize_team_key(team))
        if logo_path is not None and logo_path.exists():
            image = load_logo_image(logo_path, max_size=logo_max_size)
            if image is not None:
                artist = AnnotationBbox(
                    OffsetImage(image, zoom=logo_zoom, alpha=alpha),
                    (row["_x"], row["_y"]),
                    frameon=False,
                    pad=0.0,
                )
                ax.add_artist(artist)
                continue
            missing_logo_rows.append(row)
        else:
            missing_logo_rows.append(row)
        ax.scatter(
            row["_x"],
            row["_y"],
            s=fallback_marker_size,
            color=colors["accent"],
            edgecolor=colors["axis"],
            linewidth=0.8,
            alpha=0.8,
            zorder=3,
        )

    if annotate_all:
        annotate_rows(ax, [row for _, row in plot_df.iterrows()], colors["text"])
    elif annotate_missing_logos and missing_logo_rows:
        annotate_rows(ax, missing_logo_rows, colors["text"])

    ax.set_xlabel(x_point.label, color=colors["text"], fontsize=12)
    ax.set_ylabel(y_point.label, color=colors["text"], fontsize=12)
    ax.tick_params(colors=colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["axis"])
        spine.set_alpha(0.5)

    title_text = title or f"{x_point.label} vs {y_point.label}"
    subtitle_text = subtitle or f"{season} season, week {week}"
    ax.set_title(
        f"{title_text}\n{subtitle_text}", color=colors["text"], fontsize=15, pad=14
    )

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path, dpi=int(plotting_cfg.get("dpi", 180)), bbox_inches="tight"
        )
    return fig, ax, plot_df


def prepare_plot_frame(
    frame: pd.DataFrame, x_point: DataPoint, y_point: DataPoint
) -> pd.DataFrame:
    """Return rows with numeric x/y plotting columns."""

    missing = [
        point.column
        for point in (x_point, y_point)
        if point.column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Data point columns missing from frame: {missing}")

    plot_df = frame.loc[:, ["keys_team", x_point.column, y_point.column]].copy()
    plot_df["_x"] = pd.to_numeric(plot_df[x_point.column], errors="coerce")
    plot_df["_y"] = pd.to_numeric(plot_df[y_point.column], errors="coerce")
    plot_df = plot_df.dropna(subset=["_x", "_y"]).reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No non-null rows remain for the selected data points.")
    return plot_df


def load_fbs_teams(root: Path) -> set[str]:
    """Load FBS team names when the local metadata file exists."""

    path = root / "data" / "meta" / "fbs.csv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if "team" not in frame.columns:
        return set()
    return set(frame["team"].dropna().astype(str))


def load_logo_lookup(root: Path, logo_dir: str | Path | None = None) -> dict[str, Path]:
    """Map normalized team names to local logo image paths."""

    lookup: dict[str, Path] = {}
    if (
        logo_dir is None
        and (root / "data" / "meta" / "logos" / "logo_name_manifest.csv").exists()
    ):
        manifest_path = root / "data" / "meta" / "logos" / "logo_name_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        for _, row in manifest.iterrows():
            team_file = row.get("team_file")
            if not isinstance(team_file, str) or not team_file:
                continue
            path = Path(team_file)
            if not path.is_absolute():
                path = root / path
            for value in [row.get("school"), row.get("slug"), row.get("abbreviation")]:
                if isinstance(value, str) and value.strip():
                    lookup[normalize_team_key(value)] = path

    logo_root = resolve_repo_path(logo_dir or "data/meta/logos/by_team", root=root)
    if logo_root.exists():
        for path in logo_root.glob("*.png"):
            lookup.setdefault(normalize_team_key(path.stem), path)
    return lookup


@lru_cache(maxsize=512)
def load_logo_image(path: str | Path, max_size: int = 96):
    """Load a team logo as a small RGBA array for fast repeated plotting."""

    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGBA")
            image.thumbnail((int(max_size), int(max_size)))
            return np.asarray(image)
    except Exception:
        return None


def load_palette(
    palette_path: str | Path | None = None, *, root: Path | None = None
) -> dict[str, str]:
    """Load named colors from a TDNet palette CSV."""

    root = root or PROJECT_ROOT
    path = resolve_repo_path(
        palette_path or "docs/style/color_palettes/tdnet_palette.csv", root=root
    )
    palette: dict[str, str] = {}
    if path.exists():
        frame = pd.read_csv(path)
        if {"name", "hex"}.issubset(frame.columns):
            palette.update(
                dict(zip(frame["name"].astype(str), frame["hex"].astype(str)))
            )
    palette.setdefault("Midnight Gridiron", "#11214F")
    palette.setdefault("Polar Mist", "#E6E9ED")
    palette.setdefault("Ion Blue", "#1EA7FF")
    palette.setdefault("Edge Pink", "#FF5FA2")
    palette.setdefault("Brass", "#D4B56E")
    return palette


def style_colors(style: str, palette: dict[str, str]) -> dict[str, str]:
    """Resolve light/dark chart colors from the TDNet palette."""

    if str(style).lower() == "dark":
        return {
            "background": palette.get("Midnight Gridiron", "#11214F"),
            "text": palette.get("Polar Mist", "#E6E9ED"),
            "grid": palette.get("Ion Blue", "#1EA7FF"),
            "axis": palette.get("Polar Mist", "#E6E9ED"),
            "accent": palette.get("Edge Pink", "#FF5FA2"),
        }
    return {
        "background": "#FFFFFF",
        "text": palette.get("Midnight Gridiron", "#11214F"),
        "grid": palette.get("Polar Mist", "#E6E9ED"),
        "axis": palette.get("Midnight Gridiron", "#11214F"),
        "accent": palette.get("Ion Blue", "#1EA7FF"),
    }


def annotate_rows(ax, rows, color: str) -> None:
    """Annotate fallback markers with compact team names."""

    for row in rows:
        if isinstance(row, pd.Series):
            team = str(row["keys_team"])
            x_value = row["_x"]
            y_value = row["_y"]
        else:
            team = str(row.keys_team)
            x_value = row._x
            y_value = row._y
        ax.annotate(
            compact_team_label(team),
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
            color=color,
            alpha=0.78,
        )


def resolve_repo_path(path: str | Path, *, root: Path) -> Path:
    """Resolve relative config paths against the repository root."""

    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return (root / resolved).resolve()


def humanize_column(column: str) -> str:
    """Convert a raw column name to a readable label."""

    label = str(column)
    replacements = [
        ("statOff_", "Offense "),
        ("statDef_", "Defense "),
        ("statGen_", "General "),
        ("statSpe_", "Special Teams "),
        ("offense_", "Offense "),
        ("defense_", "Defense "),
        ("target_", "Target "),
        ("market_", "Market "),
        ("roster_", "Roster "),
        ("travel_", "Travel "),
        ("coach_", "Coach "),
        ("y_", "Label "),
    ]
    for prefix, display in replacements:
        if label.startswith(prefix):
            label = display + label[len(prefix) :]
            break
    return (
        label.replace("_p_p_a", " PPA")
        .replace("_", " ")
        .title()
        .replace("Ppa", "PPA")
        .replace("Tds", "TDs")
        .replace("Qb", "QB")
        .replace("Db", "DB")
        .replace("Tz", "TZ")
    )


def infer_group(column: str) -> str:
    """Infer a catalog group from a raw column name."""

    if column.startswith("statOff_"):
        return "stat_off"
    if column.startswith("statDef_"):
        return "stat_def"
    if column.startswith("statGen_"):
        return "stat_general"
    if column.startswith("statSpe_"):
        return "stat_special"
    return str(column).split("_", 1)[0] if "_" in str(column) else "other"


def normalize_lookup_text(value: str) -> str:
    """Normalize text for data-point lookup."""

    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def normalize_team_key(value: str) -> str:
    """Normalize team names and logo slugs to one matching key."""

    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def compact_team_label(team: str) -> str:
    """Return a compact fallback label for missing-logo points."""

    words = [word for word in re.split(r"\s+", str(team).strip()) if word]
    if len(words) == 1:
        return words[0][:8]
    return "".join(word[0] for word in words[:3]).upper()
