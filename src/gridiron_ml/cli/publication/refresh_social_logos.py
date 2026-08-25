"""Refresh high-resolution local logo assets for one TDNet Top 10 poll."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image

from gridiron_ml.publication.social_predictions import select_featured_games
from gridiron_ml.publication.social_top10 import prepare_top10


ESPN_LOGO_URL = "https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--poll", type=Path, help="TDNet Top 25 source.")
    source.add_argument("--games", type=Path, help="Weekly consensus games source.")
    parser.add_argument("--tdnet-poll", type=Path,
                        help="Optional TDNet poll used to rank teams in --games.")
    parser.add_argument("--logo-manifest", type=Path,
                        default=Path("data/meta/logos/logo_name_manifest.csv"))
    parser.add_argument("--logo-root", type=Path, default=Path("data/meta/logos"))
    parser.add_argument("--minimum-px", type=int, default=500)
    args = parser.parse_args()

    if args.poll:
        poll = _read(args.poll)
        selected_teams = [team.team for team in prepare_top10(poll)]
    else:
        poll = _read(args.tdnet_poll) if args.tdnet_poll else None
        featured, sickos = select_featured_games(_read(args.games), poll)
        games = [*featured, *([sickos] if sickos else [])]
        selected_teams = list(dict.fromkeys(
            team for game in games for team in (game.away_team, game.home_team)
        ))
    manifest = pd.read_csv(args.logo_manifest)
    by_school = manifest.drop_duplicates("school").set_index("school")
    missing = [team for team in selected_teams if team not in by_school.index]
    if missing:
        raise ValueError(f"Logo manifest has no team IDs for: {missing}")

    by_team = args.logo_root / "by_team"
    by_team.mkdir(parents=True, exist_ok=True)
    for position, team in enumerate(selected_teams, start=1):
        record = by_school.loc[team]
        team_id = int(record["id"])
        slug = str(record["slug"])
        request = Request(ESPN_LOGO_URL.format(team_id=team_id), headers={"User-Agent": "TDNet/1.0"})
        with urlopen(request, timeout=30) as response, NamedTemporaryFile(
            dir=by_team, suffix=".png", delete=False
        ) as temporary:
            shutil.copyfileobj(response, temporary)
            temporary_path = Path(temporary.name)
        try:
            with Image.open(temporary_path) as logo:
                logo.verify()
            with Image.open(temporary_path) as logo:
                if min(logo.size) < args.minimum_px:
                    raise ValueError(
                        f"{team} logo is only {logo.width}x{logo.height}; "
                        f"expected at least {args.minimum_px}px."
                    )
            numeric_path = args.logo_root / f"{team_id}.png"
            team_path = by_team / f"{slug}.png"
            temporary_path.replace(numeric_path)
            shutil.copy2(numeric_path, team_path)
            print(f"{position:>2}  {team:<24} {team_path}")
        finally:
            temporary_path.unlink(missing_ok=True)

def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


if __name__ == "__main__":
    main()
