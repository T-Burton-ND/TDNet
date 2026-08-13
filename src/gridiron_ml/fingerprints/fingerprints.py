"""src.gridiron_ml.fingerprints.fingerprints.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Build, load, and split time-dependent team fingerprints.
"""

from pathlib import Path

import pandas as pd

from gridiron_ml.pipeline.contracts.artifacts import (
    canonical_fingerprint_path,
    fingerprint_version_dir,
    legacy_fingerprints_path,
    legacy_labels_path,
    team_week_fingerprints_path,
    team_week_labels_path,
)
from gridiron_ml.pipeline.contracts.features import (
    DEFAULT_TRAINING_TARGET,
    FINGERPRINT_KEY_COLUMNS,
    HAS_NEXT_GAME_COLUMN,
)
from gridiron_ml.fingerprints.features import DEFAULT_FEATURE_SPEC, split_frame
from gridiron_ml.fingerprints.builders import (
    get_fingerprint_builder,
)
from gridiron_ml.pipeline.schemas import validate_training_feature_frame
from gridiron_ml.pipeline.validation.leakage import (
    assert_default_target_is_next_margin,
    assert_no_leaky_coach_features,
)


class Fingerprints:
    """Load and split team-week fingerprints.

    v0 rows represent team state after the completed ``keys_week``. The intended
    training label is ``y_next_margin``: the team's margin in its next scheduled
    game. ``y_margin_this_week`` is same-row completed-game information and is
    unsafe as a training target when paired with postgame/current-week features.
    """
    def __init__(self, version, postseason=False, root=None, team_game_tables_dir=None):
        """Internal helper for the init__ step."""
        self.version = int(version)
        self.postseason = bool(postseason)
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self.data_root = self.root / "data"
        self.fp_dir = fingerprint_version_dir(self.root, self.version)
        self.canonical_path = canonical_fingerprint_path(self.fp_dir)
        self.legacy_fingerprints_path = legacy_fingerprints_path(self.fp_dir, self.version)
        self.legacy_labels_path = legacy_labels_path(self.fp_dir, self.version)
        self.fbs_path = self.data_root / "meta" / "fbs.csv"
        if team_game_tables_dir is None:
            self.team_game_tables_dir = self.data_root / "team_game_tables"
        else:
            self.team_game_tables_dir = Path(team_game_tables_dir)
        self._frame_cache = None

    def build(self, overwrite=False):
        """Run the build step and return its normalized result."""
        path = self._builder().build(overwrite=overwrite)
        self._frame_cache = None
        return path

    def refresh(self):
        """Run the refresh step and return its normalized result."""
        path = self._builder().refresh()
        self._frame_cache = None
        return path

    def frame(self, seasons=None, season=None, week=None, team=None, columns=None):
        """Run the frame step and return its normalized result."""
        frame = self._load_frame().copy()
        frame = self._apply_postseason_filter(frame)

        if season is not None:
            seasons = [season]
        if seasons is not None:
            season_set = {int(s) for s in seasons}
            frame = frame.loc[pd.to_numeric(frame["keys_season"], errors="coerce").isin(season_set)].copy()
        if week is not None:
            frame = frame.loc[pd.to_numeric(frame["keys_week"], errors="coerce") == int(week)].copy()
        if team is not None:
            team_norm = self._norm_team_name(team)
            frame = frame.loc[frame["keys_team"].astype(str).map(self._norm_team_name) == team_norm].copy()

        if columns is not None:
            keep = [c for c in columns if c in frame.columns]
            frame = frame.loc[:, keep].copy()

        return frame.reset_index(drop=True)

    def training_block(self, years, feature_spec=None):
        """Return training features using y_next_margin as the default label."""
        spec = feature_spec or DEFAULT_FEATURE_SPEC
        assert_default_target_is_next_margin(spec.target_column)
        frame = self.frame(seasons=years)
        if HAS_NEXT_GAME_COLUMN in frame.columns:
            frame = frame.loc[frame[HAS_NEXT_GAME_COLUMN].fillna(False)].copy()
        elif DEFAULT_TRAINING_TARGET in frame.columns:
            frame = frame.loc[pd.to_numeric(frame[DEFAULT_TRAINING_TARGET], errors="coerce").notna()].copy()
        self._assert_training_artifact_is_safe(frame)
        x_df, y, meta_df, market_df = self._split_frame(frame, feature_spec=spec)
        validate_training_feature_frame(
            x_df,
            allow_market_features_for_training=spec.allow_market_features_for_training,
        )
        return x_df, y, meta_df, market_df

    def prediction_block(self, season, predict_week, scheduled_only=False):
        """Return features from the latest completed week before predict_week."""
        source_week = int(predict_week) - 1
        frame = self.frame(season=season, week=source_week)
        if scheduled_only and "next_week" in frame.columns:
            frame = frame.loc[pd.to_numeric(frame["next_week"], errors="coerce") == int(predict_week)].copy()
        x_df, _, meta_df, market_df = self._split_frame(frame)
        return x_df, meta_df, market_df

    def team_fingerprint(self, team, season, week):
        """Run the team_fingerprint step and return its normalized result."""
        frame = self.frame(season=season, week=week, team=team)
        if frame.empty:
            raise ValueError(f"No fingerprint rows found for team='{team}', season={season}, week={week}.")
        x_df, y, meta_df, market_df = self._split_frame(frame.iloc[[0]].copy())
        return x_df, y, meta_df, market_df

    def season_snapshot(self, season, week):
        """Run the season_snapshot step and return its normalized result."""
        frame = self.frame(season=season, week=week)
        if "keys_team" in frame.columns:
            frame = frame.sort_values(["keys_team"]).drop_duplicates(subset=["keys_team"], keep="first")
        x_df, y, meta_df, market_df = self._split_frame(frame.reset_index(drop=True))
        return x_df, y, meta_df, market_df

    def average_team(self, season=None, years=None, scope="season"):
        """Run the average_team step and return its normalized result."""
        scope = str(scope).strip().lower()
        if scope not in {"season", "all_time"}:
            raise ValueError("scope must be one of: 'season', 'all_time'.")

        if scope == "season":
            if season is None and not years:
                raise ValueError("season average requires season= or years=")
            if season is not None:
                frame = self.frame(season=season)
            else:
                frame = self.frame(seasons=years)
        else:
            frame = self.frame(seasons=years) if years is not None else self.frame()

        x_df, _, _, _ = self._split_frame(frame)
        if x_df.empty:
            raise ValueError("average_team source frame is empty.")
        avg = pd.DataFrame([x_df.mean(axis=0, numeric_only=True)], columns=list(x_df.columns))
        avg.index = [0]
        return avg

    def _builder(self):
        """Internal helper for the builder step."""
        builder_cls = get_fingerprint_builder(self.version)
        return builder_cls(self.version, root=self.root, team_game_tables_dir=self.team_game_tables_dir)

    def _load_frame(self):
        """Internal helper for the load_frame step."""
        if self._frame_cache is not None:
            return self._frame_cache

        if self.canonical_path.exists():
            frame = pd.read_parquet(self.canonical_path)
        elif self.legacy_fingerprints_path.exists() and self.legacy_labels_path.exists():
            frame = self._builder()._merge_legacy_parquets()
        else:
            raise FileNotFoundError(
                "No canonical or legacy fingerprint artifacts were found. "
                "Run Fingerprints.build() first."
            )

        frame = self._append_extra_season_artifacts(frame)
        self._frame_cache = frame
        return self._frame_cache

    def _append_extra_season_artifacts(self, frame):
        """Internal helper for the append_extra_season_artifacts step."""
        if "keys_season" not in frame.columns or not self.fp_dir.exists():
            return frame
        existing = set(pd.to_numeric(frame["keys_season"], errors="coerce").dropna().astype(int))
        extras = []
        for season_dir in sorted(self.fp_dir.iterdir()):
            if not season_dir.is_dir() or not season_dir.name.isdigit():
                continue
            season = int(season_dir.name)
            if season in existing:
                continue
            fp_path = team_week_fingerprints_path(self.fp_dir, season)
            if not fp_path.exists():
                continue
            extra = pd.read_parquet(fp_path)
            label_path = team_week_labels_path(self.fp_dir, season)
            if label_path.exists():
                labels = pd.read_parquet(label_path)
                key_cols = [c for c in FINGERPRINT_KEY_COLUMNS if c in extra.columns and c in labels.columns]
                label_cols = [c for c in labels.columns if c not in extra.columns]
                if key_cols and label_cols:
                    extra = extra.merge(labels.loc[:, key_cols + label_cols], on=key_cols, how="left")
            extras.append(extra.reindex(columns=frame.columns).dropna(axis=1, how="all"))
        if not extras:
            return frame
        return pd.concat([frame, *extras], ignore_index=True, sort=False).reindex(columns=frame.columns)

    def _apply_postseason_filter(self, frame):
        """Internal helper for the apply_postseason_filter step."""
        if self.postseason:
            return frame
        if "keys_season_type" not in frame.columns:
            return frame
        season_type = frame["keys_season_type"].astype(str).str.lower()
        return frame.loc[season_type.eq("regular")].copy()

    def split_frame(self, frame, feature_spec=None):
        """Run the split_frame step and return its normalized result."""
        return split_frame(frame, feature_spec or DEFAULT_FEATURE_SPEC)

    def _split_frame(self, frame, feature_spec=None):
        """Internal helper for the split_frame step."""
        return self.split_frame(frame, feature_spec=feature_spec)

    def _assert_training_artifact_is_safe(self, frame):
        """Fail loudly when stale persisted artifacts contain known unsafe columns."""
        try:
            assert_no_leaky_coach_features(frame.columns)
        except ValueError as exc:
            raise ValueError(
                "Fingerprint artifact may be stale and is blocked for training. "
                "Rebuild or clean the v0 artifact before using it for model training. "
                f"{exc}"
            ) from exc

    def _norm_team_name(self, value):
        """Internal helper for the norm_team_name step."""
        s = str(value).casefold()
        return "".join(ch for ch in s if ch.isalnum())
