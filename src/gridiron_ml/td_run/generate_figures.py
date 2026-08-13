"""Regenerate TDNet figure artifacts from comparison CSV tables.

Usage:
    Run `python -m gridiron_ml.td_run.generate_figures --root data/comparisons`
    after rerunning evaluations or restoring local comparison tables.

Logic flow:
    1. Discover season comparison directories that contain `tables/*.csv`.
    2. Load each table bundle and call the existing plotting functions.
    3. Rebuild season-level and poll figures under ignored `plots/` folders.

Responsibility:
    Keep heavyweight image artifacts out of git while preserving a one-command
    path to recreate figures from the smaller performance tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from gridiron_ml.td_run.poll_viz import plot_weekly_top25_table
from gridiron_ml.td_run.season_vs_vegas import load_eval_config, save_evaluation_plots


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_tables(tables_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all CSV files in a tables directory keyed by filename stem."""
    tables_path = Path(tables_dir)
    if not tables_path.exists():
        return {}
    tables = {}
    for csv_path in sorted(tables_path.glob("*.csv")):
        if not csv_path.is_file():
            continue
        try:
            tables[csv_path.stem] = pd.read_csv(csv_path)
        except EmptyDataError:
            tables[csv_path.stem] = pd.DataFrame()
    return tables


def regenerate_comparison_figures(
    root: str | Path = "data/comparisons", eval_config=None
) -> list[Path]:
    """Regenerate season comparison plots from committed comparison tables."""
    root_path = resolve_path(root)
    eval_config = force_png_plots(eval_config)
    saved_dirs: list[Path] = []
    for season_dir in discover_comparison_dirs(root_path):
        tables = load_tables(season_dir / "tables")
        if not tables:
            continue
        plots_dir = save_evaluation_plots(tables, season_dir, eval_config=eval_config)
        saved_dirs.append(plots_dir)
    return saved_dirs


def regenerate_poll_figures(
    root: str | Path = "data/comparisons",
    top_n: int = 25,
    logo_dir=None,
    eval_config=None,
) -> list[Path]:
    """Regenerate poll table images from committed poll table CSVs."""
    root_path = resolve_path(root)
    eval_config = force_png_plots(eval_config)
    saved_paths: list[Path] = []
    for poll_dir in discover_poll_dirs(root_path):
        table_path = poll_dir / "tables" / "weekly_poll_top25.csv"
        if not table_path.exists():
            continue
        weekly_poll = pd.read_csv(table_path)
        plot_path = plot_weekly_top25_table(
            weekly_poll,
            poll_dir / "plots" / "weekly_poll_top25_table.png",
            top_n=top_n,
            logo_dir=logo_dir,
            eval_config=eval_config,
        )
        if plot_path is not None:
            saved_paths.append(plot_path)
    return saved_paths


def discover_comparison_dirs(root: Path) -> list[Path]:
    """Find season directories with comparison metric tables."""
    if not root.exists():
        return []
    return sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "tables").exists()
    )


def discover_poll_dirs(root: Path) -> list[Path]:
    """Find poll output directories with committed weekly poll tables."""
    if not root.exists():
        return []
    return sorted(
        path.parent.parent
        for path in root.glob("**/tables/weekly_poll_top25.csv")
        if path.is_file()
    )


def force_png_plots(eval_config=None) -> dict:
    """Resolve an eval config and enable plot regeneration explicitly."""
    if eval_config is None:
        cfg = load_eval_config()
    elif isinstance(eval_config, (str, Path)):
        cfg = load_eval_config(eval_config_path=eval_config)
    else:
        cfg = dict(eval_config)
    cfg.setdefault("artifacts", {})["png_plots"] = True
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Resolve a possibly relative path against the repository root."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for figure regeneration."""
    parser = argparse.ArgumentParser(
        description="Regenerate TDNet figures from committed CSV tables."
    )
    parser.add_argument(
        "--root", default="data/comparisons", help="Comparison artifact root to scan."
    )
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Optional evaluation config YAML for plot styling.",
    )
    parser.add_argument(
        "--logo-dir", default=None, help="Optional team logo directory for poll plots."
    )
    parser.add_argument(
        "--top-n", type=int, default=25, help="Number of poll rows to draw."
    )
    parser.add_argument(
        "--skip-comparisons",
        action="store_true",
        help="Do not regenerate season comparison plots.",
    )
    parser.add_argument(
        "--skip-polls", action="store_true", help="Do not regenerate poll plots."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the figure regeneration CLI."""
    args = parse_args(argv)
    saved: list[Path] = []
    if not args.skip_comparisons:
        saved.extend(
            regenerate_comparison_figures(root=args.root, eval_config=args.eval_config)
        )
    if not args.skip_polls:
        saved.extend(
            regenerate_poll_figures(
                root=args.root,
                top_n=args.top_n,
                logo_dir=args.logo_dir,
                eval_config=args.eval_config,
            )
        )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
