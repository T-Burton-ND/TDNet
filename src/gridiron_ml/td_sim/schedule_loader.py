"""src.gridiron_ml.td_sim.schedule_loader.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

from pathlib import Path

import pandas as pd

from gridiron_ml.pipeline.schemas import validate_prediction_rows

from .bootstrap import ensure_schedule_team_game_table


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ScheduleLoader:
    """Load and standardize schedule/game tables for TD Sim simulation."""

    def __init__(self, config=None):
        """Internal helper for the init__ step."""
        self.config = dict(config or {})

    def load_schedule(self, season):
        """Run the load_schedule step and return its normalized result."""
        path = self._schedule_path(season)
        if not path.exists():
            if not bool(self.config.get("auto_bootstrap", True)):
                raise FileNotFoundError(f"Schedule table does not exist: {path}")
            path = ensure_schedule_team_game_table(
                season=season,
                raw_cache_dir=self.config.get("raw_cache_dir", "data/raw/cfbd/v2"),
                team_game_tables_dir=path.parent,
                division=self.config.get("division", "fbs"),
                output_format="parquet" if path.suffix.lower() == ".parquet" else "csv",
                api_key_env=self.config.get("api_key_env", "CFBD_API_KEY"),
                refresh_raw=bool(self.config.get("refresh_raw", False)),
            )
        if path.suffix == ".parquet":
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_csv(path)
        return self.standardize(raw, season=season)

    def standardize(self, frame, season=None):
        """Run the standardize step and return its normalized result."""
        df = frame.copy()
        home_col = "game_is_home" if "game_is_home" in df.columns else None
        if home_col is not None:
            df = df.loc[df[home_col].astype(bool)].copy()

        out = pd.DataFrame(index=df.index)
        out["season"] = self._col(df, ["season", "keys_season"], default=season).astype(int)
        out["week"] = pd.to_numeric(self._col(df, ["week", "keys_week"], default=0), errors="coerce").fillna(0).astype(int)
        out["game_id"] = self._col(df, ["game_id", "keys_game_id"], default=None)
        out["season_type"] = self._col(df, ["season_type", "keys_season_type"], default="regular").fillna("regular")
        out["home_team"] = self._col(df, ["home_team", "keys_team"], default=None).astype(str)
        out["away_team"] = self._col(df, ["away_team", "keys_opponent"], default=None).astype(str)
        out["neutral_site"] = self._col(df, ["neutral_site", "game_neutral_site"], default=False).fillna(False).astype(bool)
        out["conference_game"] = self._col(df, ["conference_game", "game_conference_game"], default=pd.NA)
        out["venue"] = self._col(df, ["venue", "venue_name"], default=pd.NA)
        out["start_date"] = self._col(df, ["start_date", "keys_game_date"], default=pd.NA)
        out["home_points"] = pd.to_numeric(self._col(df, ["home_points", "target_points_for"], default=pd.NA), errors="coerce")
        out["away_points"] = pd.to_numeric(self._col(df, ["away_points", "target_points_against"], default=pd.NA), errors="coerce")
        out = out.drop_duplicates(["season", "week", "game_id", "home_team", "away_team"]).reset_index(drop=True)
        missing_teams = out["home_team"].isin(["", "None", "<NA>", "nan"]) | out["away_team"].isin(["", "None", "<NA>", "nan"])
        if missing_teams.any():
            examples = out.loc[missing_teams, ["season", "week", "game_id", "home_team", "away_team"]].head(10)
            raise ValueError(f"Schedule table has missing home/away teams. Examples:\n{examples.to_string(index=False)}")
        return validate_prediction_rows(out)

    def _schedule_path(self, season):
        """Internal helper for the schedule_path step."""
        template = self.config.get("schedule_path_template", "data/team_game_tables/team_game_table_{season}_fbs.csv")
        return self._resolve(str(template).format(season=int(season)))

    def _resolve(self, path):
        """Internal helper for the resolve step."""
        path = Path(path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _col(self, df, names, default=None):
        """Internal helper for the col step."""
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series(default, index=df.index)
