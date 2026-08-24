"""Refresh high-resolution local logo assets for one TDNet Top 10 poll."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image

from gridiron_ml.publication.social_top10 import prepare_top10


ESPN_LOGO_URL = "https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=Path, required=True)
    parser.add_argument("--logo-manifest", type=Path,
                        default=Path("data/meta/logos/logo_name_manifest.csv"))
    parser.add_argument("--logo-root", type=Path, default=Path("data/meta/logos"))
    parser.add_argument("--minimum-px", type=int, default=500)
    args = parser.parse_args()

    poll = pd.read_parquet(args.poll) if args.poll.suffix == ".parquet" else pd.read_csv(args.poll)
    teams = prepare_top10(poll)
    manifest = pd.read_csv(args.logo_manifest)
    by_school = manifest.drop_duplicates("school").set_index("school")
    missing = [team.team for team in teams if team.team not in by_school.index]
    if missing:
        raise ValueError(f"Logo manifest has no team IDs for: {missing}")

    by_team = args.logo_root / "by_team"
    by_team.mkdir(parents=True, exist_ok=True)
    for team in teams:
        record = by_school.loc[team.team]
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
                        f"{team.team} logo is only {logo.width}x{logo.height}; "
                        f"expected at least {args.minimum_px}px."
                    )
            numeric_path = args.logo_root / f"{team_id}.png"
            team_path = by_team / f"{slug}.png"
            temporary_path.replace(numeric_path)
            shutil.copy2(numeric_path, team_path)
            print(f"{team.rank:>2}  {team.team:<24} {team_path}")
        finally:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
