"""src.gridiron_ml.fingerprints.builders.v0.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Build, load, and split time-dependent team fingerprints.
"""

from datetime import datetime, timezone
import json
import re

import numpy as np
import pandas as pd

from gridiron_ml.pipeline.contracts.artifacts import (
    METADATA_FILENAME,
    TEAM_GAME_TABLE_REGEX,
    metadata_path,
    team_week_fingerprints_path,
    team_week_labels_path,
)
from gridiron_ml.pipeline.contracts.features import (
    DEFAULT_TRAINING_TARGET,
    FINGERPRINT_KEY_COLUMNS,
    HAS_NEXT_GAME_COLUMN,
    LABEL_COLUMNS,
    MARKET_COLUMNS,
    SAME_WEEK_TARGET,
    TARGET_AVG_COLUMNS,
    TARGET_COLUMNS,
    is_market_column,
)
from gridiron_ml.pipeline.contracts.metadata import (
    ARTIFACT_KIND_HISTORICAL_FINGERPRINT,
    fingerprint_metadata_payload,
)
from gridiron_ml.fingerprints.feature_map import map_from_dataframe
from gridiron_ml.fingerprints.builders.common import (
    coerce_numeric_columns,
    normalize_parquet_dtypes,
    regular_season_only,
    sort_team_week_frame,
)
from gridiron_ml.pipeline.canonicalization import canonicalize_team_game_table_columns
from gridiron_ml.pipeline.schemas import (
    validate_fingerprint_feature_artifact,
    validate_fingerprint_frame,
    validate_label_frame,
)
from gridiron_ml.pipeline.validation.leakage import leaky_coach_feature_columns

from .base import BaseFingerprintBuilder


class V0FingerprintBuilder(BaseFingerprintBuilder):
    """Build v0 team-week fingerprints.

    v0 rows are state-after-week fingerprints. ``y_next_margin`` is the
    default training target; ``y_margin_this_week`` is label metadata and unsafe
    with same-row postgame features.
    """
    def build(self, overwrite=False):
        """Run the build step and return its normalized result."""
        if self.canonical_path.exists() and not overwrite:
            return self.canonical_path
        if overwrite:
            self.cleanup_artifacts_for_rebuild()
        return self._build_from_team_game_tables(overwrite=overwrite)

    def _build_from_team_game_tables(self, overwrite=False):
        """Internal helper for the build_from_team_game_tables step."""
        season_paths = self._team_game_table_paths()
        if not season_paths:
            raise FileNotFoundError(
                f"No team-game tables were found under {self.team_game_tables_dir}. "
                "Expected files like team_game_table_2024_fbs.csv."
            )

        build_timestamp = datetime.now(timezone.utc).isoformat()
        season_fingerprints = []
        season_labels = []
        canonical_frames = []

        for path in season_paths:
            year = self._year_from_path(path)
            raw = self._load_team_game_table(path)
            fp_df, label_df, canonical_df = self._build_season_from_team_game_table(
                raw,
                build_timestamp=build_timestamp,
            )

            season_fingerprints.append(fp_df)
            season_labels.append(label_df)
            canonical_frames.append(canonical_df)
            self._write_season_artifacts(year, fp_df, label_df, overwrite=overwrite)

        full_fp = pd.concat(season_fingerprints, ignore_index=True, sort=False)
        full_labels = pd.concat(season_labels, ignore_index=True, sort=False)
        canonical = pd.concat(canonical_frames, ignore_index=True, sort=False)

        full_fp = self._normalize_parquet_dtypes(full_fp)
        full_labels = self._normalize_parquet_dtypes(full_labels)
        canonical = self._normalize_parquet_dtypes(canonical)

        full_fp = self._sort_frame(full_fp)
        full_labels = self._sort_frame(full_labels)
        canonical = self._sort_frame(canonical)

        validate_fingerprint_feature_artifact(full_fp)
        validate_label_frame(full_labels)
        validate_fingerprint_frame(canonical)

        self.fp_dir.mkdir(parents=True, exist_ok=True)
        full_fp.to_parquet(self.legacy_fingerprints_path, index=False)
        full_labels.to_parquet(self.legacy_labels_path, index=False)
        canonical.to_parquet(self.canonical_path, index=False)
        self._write_metadata(metadata_path(self.fp_dir), artifact_kind=ARTIFACT_KIND_HISTORICAL_FINGERPRINT)
        return self.canonical_path

    def _build_season_from_team_game_table(self, frame, build_timestamp):
        """Internal helper for the build_season_from_team_game_table step."""
        if frame.empty:
            raise ValueError("Cannot build fingerprints from an empty team-game table.")

        year = int(pd.to_numeric(frame["keys_season"], errors="coerce").mode().iloc[0])
        frame = frame.loc[pd.to_numeric(frame["keys_season"], errors="coerce") == year].copy()
        frame["keys_week"] = pd.to_numeric(frame["keys_week"], errors="coerce")

        fm = map_from_dataframe(frame, keep_unknown=True)
        raw_coach_cols = [c for c in fm.by_category.get("coach", []) if c != "coach_season_games"]
        blocked_coach_cols = set(leaky_coach_feature_columns(raw_coach_cols))
        coach_cols = [c for c in raw_coach_cols if c not in blocked_coach_cols]
        mean_cats = ("offense", "defense", "statOff", "statDef", "statGen", "statSpe", "travel")
        mean_cols = [c for cat in mean_cats for c in fm.by_category.get(cat, []) if c in frame.columns]
        roster_cols = [c for c in fm.by_category.get("roster", []) if c in frame.columns]
        market_cols = [c for c in fm.by_category.get("market", []) if c in frame.columns]
        target_avg_cols = [c for c in TARGET_AVG_COLUMNS if c in frame.columns]
        perf_cols = list(dict.fromkeys(mean_cols))
        static_cols = [c for c in roster_cols + coach_cols if c in frame.columns]

        base = coerce_numeric_columns(
            frame,
            mean_cols + roster_cols + coach_cols + market_cols + [
            *TARGET_COLUMNS,
            ],
        )

        key = list(FINGERPRINT_KEY_COLUMNS)
        dup_mask = base.duplicated(subset=key, keep=False)
        if dup_mask.any():
            avg_cols = [c for c in mean_cols + roster_cols + coach_cols if c in base.columns]
            first_cols = [c for c in base.columns if c not in avg_cols]
            base = (
                base.groupby(key, as_index=False, observed=True)
                .agg(
                    {
                        **{c: "mean" for c in avg_cols},
                        **{c: "first" for c in first_cols},
                    }
                )
            )

        base = base.sort_values(key).reset_index(drop=True)
        grouped = base.groupby(["keys_season", "keys_team"], sort=False, observed=True)

        if {"target_points_for", "target_points_against"}.issubset(base.columns):
            # Current-inclusive postgame rolling means: safe only for predicting
            # future games from the state after keys_week has completed.
            base["target_points_for_avg"] = (
                grouped["target_points_for"].expanding(1).mean().reset_index(level=[0, 1], drop=True)
            )
            base["target_points_against_avg"] = (
                grouped["target_points_against"].expanding(1).mean().reset_index(level=[0, 1], drop=True)
            )
            target_avg_cols = list(TARGET_AVG_COLUMNS)

        base["games_played"] = grouped.cumcount() + 1

        for col in mean_cols:
            base[col] = grouped[col].expanding(1).mean().reset_index(level=[0, 1], drop=True)
        for col in static_cols:
            base[col] = grouped[col].ffill()

        id_cols = [c for c in fm.id_cols if c in base.columns]
        out_cols = id_cols + ["games_played"]
        feature_cols = list(dict.fromkeys(mean_cols + roster_cols + coach_cols + market_cols + target_avg_cols))
        out_cols.extend([c for c in feature_cols if c in base.columns])
        for col in ["game_is_home", "game_home_away"]:
            if col in base.columns and col not in out_cols:
                out_cols.append(col)

        out = base.loc[:, out_cols].copy()
        out = out.loc[:, ~out.columns.duplicated()].copy()

        if "keys_season_type" in out.columns:
            out = regular_season_only(out)
            frame = regular_season_only(frame)

        if "game_is_home" in out.columns:
            home_counts = (
                out.assign(game_is_home=out["game_is_home"].astype("boolean"))
                .groupby("keys_team", observed=True)["game_is_home"]
                .sum(min_count=1)
            )
            keep_teams = set(home_counts[home_counts >= 2].index.astype(str))
            out = out.loc[out["keys_team"].astype(str).isin(keep_teams)].copy()
            frame = frame.loc[frame["keys_team"].astype(str).isin(keep_teams)].copy()
        else:
            keep_teams = set(out["keys_team"].dropna().astype(str))

        all_weeks = np.sort(pd.to_numeric(out["keys_week"], errors="coerce").dropna().astype(int).unique())
        teams = sorted(keep_teams)
        grid = pd.MultiIndex.from_product(
            [[year], teams, all_weeks],
            names=["keys_season", "keys_team", "keys_week"],
        ).to_frame(index=False)

        out = grid.merge(
            out,
            on=list(FINGERPRINT_KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        out = out.sort_values(["keys_season", "keys_team", "keys_week"]).reset_index(drop=True)
        fill_cols = [c for c in out.columns if c not in set(FINGERPRINT_KEY_COLUMNS)]
        out.loc[:, fill_cols] = (
            out.groupby(["keys_season", "keys_team"], sort=False, observed=True)[fill_cols]
            .ffill()
            .infer_objects(copy=False)
        )

        week0_rows = []
        travel_cols = [c for c in fm.by_category.get("travel", []) if c in out.columns]
        clear_week0_cols = [
            c
            for c in ["keys_game_id", "keys_opponent", "keys_game_date", "game_is_home", "game_home_away"]
            if c in out.columns
        ]
        for (_, _), group in out.groupby(["keys_season", "keys_team"], sort=False, observed=True):
            row = group.iloc[0].copy()
            row["keys_week"] = 0
            row["games_played"] = 0
            for col in perf_cols:
                if col in row.index:
                    row[col] = np.nan
            for col in travel_cols:
                row[col] = 0.0
            for col in market_cols:
                if col in row.index:
                    row[col] = np.nan
            for col in clear_week0_cols:
                row[col] = pd.NA
            week0_rows.append(row.to_dict())

        week0_df = pd.DataFrame(week0_rows, columns=out.columns)
        out_columns = list(out.columns)
        week0_for_concat = week0_df.dropna(axis=1, how="all")
        out_for_concat = out.dropna(axis=1, how="all")
        out = pd.concat([week0_for_concat, out_for_concat], ignore_index=True, sort=False).reindex(columns=out_columns)
        out = self._sort_frame(out)
        for col in target_avg_cols:
            out.loc[pd.to_numeric(out["keys_week"], errors="coerce") == 0, col] = 0.0

        actual = self._build_actual_schedule(frame, keep_teams=keep_teams)
        # Labels are shifted onto the prior state row. This makes y_next_margin
        # the default training target and leaves market columns as eval context.
        label_df = self._build_shifted_labels(actual, out.loc[:, list(FINGERPRINT_KEY_COLUMNS)].copy(), market_cols)

        for col in market_cols:
            if col in out.columns:
                out = out.drop(columns=col)

        canonical = out.merge(
            label_df,
            on=list(FINGERPRINT_KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        canonical["fp_version"] = self.version
        canonical["fp_subversion"] = 0
        canonical["fp_build_timestamp"] = build_timestamp

        fingerprint_cols = [
            c
            for c in canonical.columns
            if c not in {SAME_WEEK_TARGET, DEFAULT_TRAINING_TARGET, HAS_NEXT_GAME_COLUMN}
            and not is_market_column(c)
        ]
        fingerprint_df = canonical.loc[:, fingerprint_cols].copy()

        label_cols = [c for c in LABEL_COLUMNS if c in canonical.columns]
        label_df = canonical.loc[:, label_cols].copy()
        return fingerprint_df, label_df, canonical

    def _build_actual_schedule(self, frame, keep_teams):
        """Internal helper for the build_actual_schedule step."""
        schedule = frame.copy()
        schedule = schedule.loc[schedule["keys_team"].astype(str).isin(keep_teams)].copy()
        schedule["keys_week"] = pd.to_numeric(schedule["keys_week"], errors="coerce")

        if "target_team_margin" in schedule.columns:
            margin = pd.to_numeric(schedule["target_team_margin"], errors="coerce")
        else:
            margin = (
                pd.to_numeric(schedule["target_points_for"], errors="coerce")
                - pd.to_numeric(schedule["target_points_against"], errors="coerce")
            )
        schedule[SAME_WEEK_TARGET] = margin

        agg_map = {SAME_WEEK_TARGET: "mean"}
        for col in [
            "keys_game_id",
            "keys_opponent",
            "game_is_home",
            "game_home_away",
            "keys_game_date",
            "keys_conference",
            "keys_season_type",
            *MARKET_COLUMNS,
        ]:
            if col in schedule.columns:
                agg_map[col] = "first"

        schedule = (
            schedule.groupby(list(FINGERPRINT_KEY_COLUMNS), as_index=False, observed=True)
            .agg(agg_map)
            .sort_values(list(FINGERPRINT_KEY_COLUMNS))
            .reset_index(drop=True)
        )
        return schedule

    def _build_shifted_labels(self, actual, key_frame, market_cols):
        """Internal helper for the build_shifted_labels step."""
        labels = key_frame.copy()
        labels["keys_week"] = pd.to_numeric(labels["keys_week"], errors="coerce")

        current_cols = [*FINGERPRINT_KEY_COLUMNS, SAME_WEEK_TARGET]
        labels = labels.merge(
            actual.loc[:, [c for c in current_cols if c in actual.columns]],
            on=list(FINGERPRINT_KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )

        next_cols = list(FINGERPRINT_KEY_COLUMNS)
        shifted = actual.copy()
        shifted["keys_week"] = pd.to_numeric(shifted["keys_week"], errors="coerce") - 1
        shifted = shifted.rename(
            columns={
                "keys_game_id": "next_game_id",
                "keys_opponent": "next_opponent",
                "game_is_home": "next_game_is_home",
                "game_home_away": "next_game_home_away",
                SAME_WEEK_TARGET: DEFAULT_TRAINING_TARGET,
            }
        )
        next_cols.extend(
            [
                c
                for c in [
                    "next_game_id",
                    "next_opponent",
                    "next_game_is_home",
                    "next_game_home_away",
                    DEFAULT_TRAINING_TARGET,
                ]
                if c in shifted.columns
            ]
        )
        next_cols.extend([c for c in market_cols if c in shifted.columns])
        labels = labels.merge(
            shifted.loc[:, list(dict.fromkeys(next_cols))],
            on=list(FINGERPRINT_KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )

        labels["next_week"] = np.where(labels["next_game_id"].notna(), labels["keys_week"] + 1, np.nan)
        labels[HAS_NEXT_GAME_COLUMN] = labels[DEFAULT_TRAINING_TARGET].notna()
        return self._sort_frame(labels)

    def _load_team_game_table(self, path):
        """Internal helper for the load_team_game_table step."""
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)

        if "keys_season" not in frame.columns and "season" in frame.columns:
            frame = canonicalize_team_game_table_columns(frame)
        required = set(FINGERPRINT_KEY_COLUMNS)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required team-game table columns: {missing}")
        return frame

    def _team_game_table_paths(self):
        """Internal helper for the team_game_table_paths step."""
        season_paths_by_year = {}
        for path in sorted(self.team_game_tables_dir.iterdir()):
            match = re.search(TEAM_GAME_TABLE_REGEX, path.name)
            if not match:
                continue
            year = int(match.group(1))
            ext = match.group(2)
            current = season_paths_by_year.get(year)
            if current is None or (ext == "parquet" and current.suffix.lower() != ".parquet"):
                season_paths_by_year[year] = path
        return [season_paths_by_year[year] for year in sorted(season_paths_by_year)]

    def _year_from_path(self, path):
        """Internal helper for the year_from_path step."""
        match = re.search(TEAM_GAME_TABLE_REGEX, path.name)
        if not match:
            raise ValueError(f"Unrecognized team-game table filename: {path.name}")
        return int(match.group(1))

    def _write_season_artifacts(self, year, fingerprint_df, label_df, overwrite=False):
        """Internal helper for the write_season_artifacts step."""
        season_dir = self.fp_dir / str(year)
        season_dir.mkdir(parents=True, exist_ok=True)
        fp_path = team_week_fingerprints_path(self.fp_dir, year)
        label_path = team_week_labels_path(self.fp_dir, year)
        fingerprint_df = self._normalize_parquet_dtypes(fingerprint_df)
        label_df = self._normalize_parquet_dtypes(label_df)
        validate_fingerprint_feature_artifact(fingerprint_df)
        validate_label_frame(label_df)
        if overwrite or not fp_path.exists():
            fingerprint_df.to_parquet(fp_path, index=False)
        if overwrite or not label_path.exists():
            label_df.to_parquet(label_path, index=False)
        sidecar_path = season_dir / METADATA_FILENAME
        if overwrite or not sidecar_path.exists():
            self._write_metadata(sidecar_path, artifact_kind=ARTIFACT_KIND_HISTORICAL_FINGERPRINT, season=year)

    def _write_metadata(self, path, *, artifact_kind, season=None):
        """Write a small sidecar describing v0 artifact row and market policy."""
        payload = fingerprint_metadata_payload(
            version=self.version,
            default_target=DEFAULT_TRAINING_TARGET,
            unsafe_same_row_target=SAME_WEEK_TARGET,
            artifact_kind=artifact_kind,
            season=season,
        )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _sort_frame(self, frame):
        """Internal helper for the sort_frame step."""
        return sort_team_week_frame(frame)

    def _normalize_parquet_dtypes(self, frame):
        """Internal helper for the normalize_parquet_dtypes step."""
        return normalize_parquet_dtypes(frame)
