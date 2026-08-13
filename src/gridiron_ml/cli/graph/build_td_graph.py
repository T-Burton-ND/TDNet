from gridiron_ml.cli._paths import project_root
#!/usr/bin/env python3
"""Build a TDGraph database and conference-styled season visualization."""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from gridiron_ml.graph import build_season_graph, export_season_graph, plot_season_graph


def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--games", type=Path)
    parser.add_argument("--database-output", type=Path)
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--logo-dir", type=Path, default=root / "data/meta/logos/by_team")
    args = parser.parse_args()
    games_path = args.games or root / f"data/raw/cfbd/v2/games/{args.season}.parquet"
    database_output = args.database_output or root / f"data/derived/td_graph/{args.season}"
    figure_output = args.figure_output or root / f"publication/{args.season}/td_graph"
    games = pd.read_parquet(games_path) if games_path.suffix == ".parquet" else pd.read_csv(games_path)
    graph = build_season_graph(games, season=args.season, completed_only=args.completed_only)
    print(export_season_graph(graph, database_output))
    print(plot_season_graph(graph, figure_output / "season_graph.png", logo_dir=args.logo_dir))


if __name__ == "__main__": main()
