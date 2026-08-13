"""src.gridiron_ml.fingerprints.builders.base.

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
    cleanup_fingerprint_artifacts,
    fingerprint_version_dir,
    legacy_fingerprints_path,
    legacy_labels_path,
)


class BaseFingerprintBuilder:
    """Represent the BaseFingerprintBuilder component and its local behavior."""
    def __init__(self, version, root=None, team_game_tables_dir=None):
        """Internal helper for the init__ step."""
        self.version = int(version)
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[4]
        self.data_root = self.root / "data"
        self.fp_dir = fingerprint_version_dir(self.root, self.version)
        self.canonical_path = canonical_fingerprint_path(self.fp_dir)
        self.legacy_fingerprints_path = legacy_fingerprints_path(self.fp_dir, self.version)
        self.legacy_labels_path = legacy_labels_path(self.fp_dir, self.version)
        if team_game_tables_dir is None:
            self.team_game_tables_dir = self.data_root / "team_game_tables"
        else:
            self.team_game_tables_dir = Path(team_game_tables_dir)

    def build(self, overwrite=False):
        """Run the build step and return its normalized result."""
        if self.canonical_path.exists() and not overwrite:
            return self.canonical_path

        if self.legacy_fingerprints_path.exists() and self.legacy_labels_path.exists():
            frame = self._merge_legacy_parquets()
            self.fp_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(self.canonical_path, index=False)
            return self.canonical_path

        return self._build_from_team_game_tables(overwrite=overwrite)

    def refresh(self):
        """Run the refresh step and return its normalized result."""
        return self.build(overwrite=True)

    def cleanup_artifacts_for_rebuild(self):
        """Remove generated fingerprint files before a clean overwrite rebuild.

        Debug files and non-generated folders are intentionally left in place.
        """
        return cleanup_fingerprint_artifacts(self.fp_dir, self.version)

    def _merge_legacy_parquets(self):
        """Internal helper for the merge_legacy_parquets step."""
        fp_df = pd.read_parquet(self.legacy_fingerprints_path)
        y_df = pd.read_parquet(self.legacy_labels_path)

        join_cols = [c for c in fp_df.columns if c.startswith("keys_") and c in y_df.columns]
        if not join_cols:
            join_cols = ["keys_season", "keys_team", "keys_week"]

        merged = fp_df.merge(
            y_df,
            on=join_cols,
            how="left",
            suffixes=("", "_label"),
        )

        duplicate_label_cols = [c for c in merged.columns if c.endswith("_label")]
        for col in duplicate_label_cols:
            base = col[:-6]
            if base not in merged.columns:
                merged = merged.rename(columns={col: base})
        duplicate_label_cols = [c for c in merged.columns if c.endswith("_label")]
        if duplicate_label_cols:
            merged = merged.drop(columns=duplicate_label_cols)

        return merged

    def _build_from_team_game_tables(self, overwrite=False):
        """Internal helper for the build_from_team_game_tables step."""
        raise NotImplementedError(
            "Raw team-game-table fingerprint builds are not wired for this version yet. "
            "If legacy fingerprint and label parquets exist, this builder can materialize the "
            "canonical combined parquet from them."
        )
