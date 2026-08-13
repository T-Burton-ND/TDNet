"""src.gridiron_ml.td_run.matchups.builder.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Convert team fingerprints into matchup feature rows for prediction models.
"""

from pathlib import Path
from fnmatch import fnmatch

import numpy as np
import pandas as pd

from gridiron_ml.td_run.matchups.unit_matchups import (
    default_unit_pairing_specs,
    feature_direction,
    normalize_unit_direction,
)


DEFAULT_ZERO_OFFSET = 1e-8
DEFAULT_UNIT_PASSTHROUGH_PATTERNS = (
    "opp_adj_*",
    "opponent_adj_*",
    "opponent_adjusted_*",
    "adjusted_*",
    "time_adj_*",
    "graph_*",
    "market_*",
    "*_market_*",
    "vegas_*",
    "*_vegas_*",
)


class MatchupBuilder:
    """Represent the MatchupBuilder component and its local behavior."""

    def __init__(
        self,
        representation="unit_matchup",
        eps=None,
        blocks=None,
        safe_math=True,
        zero_offset=None,
        unit_pairings=None,
        unit_pairings_path=None,
        include_unit_secondary=False,
        include_unit_passthrough=True,
        unit_passthrough_patterns=None,
    ):
        """Internal helper for the init__ step."""
        self.representation = str(representation)
        self.eps = float(DEFAULT_ZERO_OFFSET if eps is None else eps)
        self.blocks = dict(blocks or {})
        self.safe_math = bool(safe_math)
        self.zero_offset = float(
            DEFAULT_ZERO_OFFSET if zero_offset is None else zero_offset
        )
        self.unit_pairings = unit_pairings
        self.unit_pairings_path = (
            Path(unit_pairings_path) if unit_pairings_path else None
        )
        self.include_unit_secondary = bool(include_unit_secondary)
        self.include_unit_passthrough = bool(include_unit_passthrough)
        self.unit_passthrough_patterns = tuple(
            unit_passthrough_patterns or DEFAULT_UNIT_PASSTHROUGH_PATTERNS
        )
        self._last_feature_names = []

        if self.representation not in self.available_representations():
            supported = ", ".join(self.available_representations())
            raise ValueError(
                f"Unsupported representation='{self.representation}'. Supported: {supported}"
            )

    def available_representations(self):
        """Run the available_representations step and return its normalized result."""
        return sorted(
            [
                "all_numeric",
                "blockwise_diff_product",
                "blockwise_diffsum",
                "blockwise_full",
                "concat",
                "diff",
                "diff_abs",
                "diff_log_ratio",
                "diff_normalized",
                "diff_product",
                "diff_ratio",
                "diffsum",
                "full_basic",
                "full_interaction",
                "log_ratio",
                "minmax",
                "normalized_diff",
                "off_def_cross",
                "off_def_cross_full",
                "poly2_diff",
                "poly3_diff",
                "ratio",
                "signed_minmax",
                "sum",
                "unit_matchup",
                "unit_matchup_full",
            ]
        )

    def feature_names(self):
        """Run the feature_names step and return its normalized result."""
        return list(self._last_feature_names)

    def build(self, home, away):
        """Run the build step and return its normalized result."""
        out = self.build_many(home, away)
        if len(out) == 0:
            return out
        return out.iloc[[0]].reset_index(drop=True)

    def build_many(self, home, away, meta_df=None, market_df=None):
        """Run the build_many step and return its normalized result."""
        home_df, away_df, base_names = self._coerce_pair(home, away)
        values, names = self._build_representation(home_df, away_df, base_names)
        self._last_feature_names = list(names)
        out = pd.DataFrame(values, columns=names)
        if meta_df is not None:
            out.index = pd.RangeIndex(len(out))
        return out

    def matchups(
        self, fingerprint_block, meta_df, market_df=None, representation=None, y=None
    ):
        """Run the matchups step and return its normalized result."""
        rep = self._builder_for(representation)
        home_df, away_df, paired_meta, paired_market, paired_y = rep._pair_game_rows(
            fingerprint_block,
            meta_df,
            market_df=market_df,
            y=y,
        )
        matchups_df = rep.build_many(home_df, away_df)
        if paired_y is not None:
            paired_meta = paired_meta.copy()
            paired_meta["y"] = pd.to_numeric(
                pd.Series(paired_y), errors="coerce"
            ).reset_index(drop=True)
        return (
            matchups_df.reset_index(drop=True),
            paired_meta.reset_index(drop=True),
            paired_market.reset_index(drop=True),
        )

    def team_vs_average(
        self,
        fingerprint_block,
        meta_df,
        average_team_df,
        market_df=None,
        representation=None,
    ):
        """Run the team_vs_average step and return its normalized result."""
        rep = self._builder_for(representation)
        feature_df = rep._coerce_feature_df(fingerprint_block)
        avg_df = rep._coerce_feature_df(average_team_df)
        if len(avg_df) != 1:
            raise ValueError("average_team_df must contain exactly one row.")
        avg_block = pd.DataFrame(
            np.repeat(avg_df.to_numpy(dtype=float), len(feature_df), axis=0),
            columns=list(avg_df.columns),
        )
        matchups_df = rep.build_many(feature_df, avg_block)
        market_out = (
            market_df.copy().reset_index(drop=True)
            if market_df is not None
            else pd.DataFrame(index=matchups_df.index)
        )
        return (
            matchups_df.reset_index(drop=True),
            meta_df.reset_index(drop=True),
            market_out,
        )

    def _builder_for(self, representation=None):
        """Internal helper for the builder_for step."""
        if representation is None or str(representation) == self.representation:
            return self
        return MatchupBuilder(
            representation=representation,
            eps=self.eps,
            blocks=self.blocks,
            safe_math=self.safe_math,
            zero_offset=self.zero_offset,
            unit_pairings=self.unit_pairings,
            unit_pairings_path=self.unit_pairings_path,
            include_unit_secondary=self.include_unit_secondary,
            include_unit_passthrough=self.include_unit_passthrough,
            unit_passthrough_patterns=self.unit_passthrough_patterns,
        )

    def _coerce_pair(self, home, away):
        """Internal helper for the coerce_pair step."""
        home_df = self._coerce_feature_df(home)
        away_df = self._coerce_feature_df(away)
        if len(home_df) != len(away_df):
            raise ValueError("home and away must have the same row count.")
        if home_df.shape[1] != away_df.shape[1]:
            raise ValueError("home and away must have the same column count.")
        if list(home_df.columns) != list(away_df.columns):
            raise ValueError("home and away must share the same ordered columns.")
        return home_df, away_df, list(home_df.columns)

    def _coerce_feature_df(self, frame):
        """Internal helper for the coerce_feature_df step."""
        if frame is None:
            raise ValueError("Input frame is required.")
        if isinstance(frame, pd.DataFrame):
            out = frame.copy().reset_index(drop=True)
        else:
            arr = np.asarray(frame, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D input, got shape={arr.shape}.")
            out = pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])

        for col in out.columns:
            if pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].astype(float)
            elif not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.astype(float)

    def _pair_game_rows(self, fingerprint_block, meta_df, market_df=None, y=None):
        """Internal helper for the pair_game_rows step."""
        features = self._coerce_feature_df(fingerprint_block)
        meta = meta_df.copy().reset_index(drop=True)
        if len(features) != len(meta):
            raise ValueError(
                "fingerprint_block and meta_df must have the same row count."
            )
        meta["__row_id"] = np.arange(len(meta))

        pair_game_col = "keys_game_id"
        pair_home_col = "game_is_home"
        pair_week_col = "keys_week"
        pair_season_col = "keys_season"
        pair_team_col = "keys_team"
        pair_opp_col = "keys_opponent"

        if (
            "next_game_id" in meta.columns
            and "next_game_is_home" in meta.columns
            and pd.to_numeric(meta["next_game_id"], errors="coerce").notna().sum() >= 2
        ):
            pair_game_col = "next_game_id"
            pair_home_col = "next_game_is_home"
            pair_week_col = (
                "next_week" if "next_week" in meta.columns else pair_week_col
            )
            pair_team_col = "keys_team"
            pair_opp_col = (
                "next_opponent" if "next_opponent" in meta.columns else pair_opp_col
            )

        if pair_game_col not in meta.columns or pair_home_col not in meta.columns:
            raise ValueError(
                "meta_df must contain either current-game pairing fields or next-game pairing fields."
            )

        valid_pair_mask = meta[pair_game_col].notna() & meta[pair_home_col].notna()
        if not valid_pair_mask.all():
            features = features.loc[valid_pair_mask].reset_index(drop=True)
            meta = meta.loc[valid_pair_mask].reset_index(drop=True)
            if y is not None:
                y = (
                    pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")
                    .loc[valid_pair_mask]
                    .reset_index(drop=True)
                )
            if market_df is not None:
                market_df = (
                    market_df.copy()
                    .reset_index(drop=True)
                    .loc[valid_pair_mask]
                    .reset_index(drop=True)
                )
            meta["__row_id"] = np.arange(len(meta))

        join_cols = [
            c
            for c in [pair_season_col, pair_week_col, pair_game_col]
            if c in meta.columns
        ]
        pair_home = meta[pair_home_col].astype(bool)
        home_meta = meta.loc[pair_home == True].copy()
        away_meta = meta.loc[pair_home == False].copy()
        paired = home_meta.merge(
            away_meta,
            on=join_cols,
            how="inner",
            suffixes=("_home", "_away"),
            validate="one_to_one",
        )
        if paired.empty:
            raise ValueError(
                "No home/away game pairs could be built from the provided block."
            )

        home_df = features.iloc[paired["__row_id_home"].to_numpy()].reset_index(
            drop=True
        )
        away_df = features.iloc[paired["__row_id_away"].to_numpy()].reset_index(
            drop=True
        )

        meta_keep = [
            c
            for c in [
                "keys_season",
                "keys_week",
                "keys_game_id",
                "next_week",
                "next_game_id",
                "keys_team_home",
                "keys_team_away",
                "keys_opponent_home",
                "keys_opponent_away",
                "next_opponent_home",
                "next_opponent_away",
            ]
            if c in paired.columns
        ]
        paired_meta = paired.loc[:, meta_keep].copy()
        if (
            f"{pair_team_col}_home" in paired.columns
            and "keys_team_home" not in paired_meta.columns
        ):
            paired_meta["keys_team_home"] = paired[f"{pair_team_col}_home"]
        if (
            f"{pair_team_col}_away" in paired.columns
            and "keys_team_away" not in paired_meta.columns
        ):
            paired_meta["keys_team_away"] = paired[f"{pair_team_col}_away"]
        if (
            f"{pair_opp_col}_home" in paired.columns
            and "keys_opponent_home" not in paired_meta.columns
        ):
            paired_meta["keys_opponent_home"] = paired[f"{pair_opp_col}_home"]
        if (
            f"{pair_opp_col}_away" in paired.columns
            and "keys_opponent_away" not in paired_meta.columns
        ):
            paired_meta["keys_opponent_away"] = paired[f"{pair_opp_col}_away"]

        if market_df is None:
            paired_market = pd.DataFrame(index=paired_meta.index)
        else:
            market = market_df.copy().reset_index(drop=True)
            if len(market) != len(meta):
                raise ValueError("market_df and meta_df must have the same row count.")
            market["__row_id"] = np.arange(len(market))
            keep_market_cols = [c for c in market.columns if c.startswith("market_")]
            keep_market_cols += [
                c
                for c in [
                    "keys_season",
                    "keys_week",
                    "keys_game_id",
                    "next_week",
                    "next_game_id",
                ]
                if c in market.columns
            ]
            home_market = market.loc[
                meta[pair_home_col].astype(bool) == True,
                ["__row_id"] + keep_market_cols,
            ].copy()
            paired_market = paired.loc[:, join_cols].merge(
                home_market.drop(columns="__row_id"),
                on=join_cols,
                how="left",
            )
            paired_market = paired_market.reset_index(drop=True)

        paired_y = None
        if y is not None:
            y_series = pd.to_numeric(
                pd.Series(y).reset_index(drop=True), errors="coerce"
            )
            if len(y_series) != len(meta):
                raise ValueError(
                    "y must align row-wise with fingerprint_block and meta_df."
                )
            paired_y = y_series.iloc[paired["__row_id_home"].to_numpy()].reset_index(
                drop=True
            )

        return home_df, away_df, paired_meta, paired_market, paired_y

    def _build_representation(self, home_df, away_df, base_names):
        """Internal helper for the build_representation step."""
        home = home_df.to_numpy(dtype=float)
        away = away_df.to_numpy(dtype=float)
        rep = self.representation

        if rep == "diff":
            return self._single(home - away, base_names, "diff")
        if rep == "sum":
            return self._single(home + away, base_names, "sum")
        if rep == "concat":
            return self._concat(
                [
                    (home, self._suffix(base_names, "home")),
                    (away, self._suffix(base_names, "away")),
                ]
            )
        if rep == "diffsum":
            return self._concat(
                [
                    (home - away, self._suffix(base_names, "diff")),
                    (home + away, self._suffix(base_names, "sum")),
                ]
            )
        if rep == "diff_product":
            return self._concat(
                [
                    (home - away, self._suffix(base_names, "diff")),
                    (home * away, self._suffix(base_names, "prod")),
                ]
            )
        if rep == "diff_abs":
            delta = home - away
            return self._concat(
                [
                    (delta, self._suffix(base_names, "diff")),
                    (np.abs(delta), self._suffix(base_names, "absdiff")),
                ]
            )
        if rep == "ratio":
            values = self._safe_ratio(home, away)
            return self._single(values, base_names, "ratio")
        if rep == "log_ratio":
            values = self._safe_log_ratio(home, away)
            return self._single(values, base_names, "logratio")
        if rep == "diff_ratio":
            return self._concat(
                [
                    (home - away, self._suffix(base_names, "diff")),
                    (self._safe_ratio(home, away), self._suffix(base_names, "ratio")),
                ]
            )
        if rep == "diff_log_ratio":
            return self._concat(
                [
                    (home - away, self._suffix(base_names, "diff")),
                    (
                        self._safe_log_ratio(home, away),
                        self._suffix(base_names, "logratio"),
                    ),
                ]
            )
        if rep == "normalized_diff":
            values = self._normalized_diff(home, away)
            return self._single(values, base_names, "normdiff")
        if rep == "diff_normalized":
            return self._concat(
                [
                    (home - away, self._suffix(base_names, "diff")),
                    (
                        self._normalized_diff(home, away),
                        self._suffix(base_names, "normdiff"),
                    ),
                ]
            )
        if rep == "poly2_diff":
            delta = home - away
            return self._concat(
                [
                    (delta, self._suffix(base_names, "diff")),
                    (delta**2, self._suffix(base_names, "sqdiff")),
                ]
            )
        if rep == "poly3_diff":
            delta = home - away
            return self._concat(
                [
                    (delta, self._suffix(base_names, "diff")),
                    (delta**2, self._suffix(base_names, "sqdiff")),
                    (delta**3, self._suffix(base_names, "cubediff")),
                ]
            )
        if rep == "minmax":
            return self._concat(
                [
                    (np.minimum(home, away), self._suffix(base_names, "min")),
                    (np.maximum(home, away), self._suffix(base_names, "max")),
                ]
            )
        if rep == "signed_minmax":
            return self._concat(
                [
                    (home - away, self._suffix(base_names, "diff")),
                    (np.minimum(home, away), self._suffix(base_names, "min")),
                    (np.maximum(home, away), self._suffix(base_names, "max")),
                ]
            )
        if rep == "full_basic":
            return self._concat(
                [
                    (home, self._suffix(base_names, "home")),
                    (away, self._suffix(base_names, "away")),
                    (home - away, self._suffix(base_names, "diff")),
                    (home + away, self._suffix(base_names, "sum")),
                ]
            )
        if rep == "full_interaction":
            delta = home - away
            return self._concat(
                [
                    (home, self._suffix(base_names, "home")),
                    (away, self._suffix(base_names, "away")),
                    (delta, self._suffix(base_names, "diff")),
                    (home + away, self._suffix(base_names, "sum")),
                    (home * away, self._suffix(base_names, "prod")),
                    (np.abs(delta), self._suffix(base_names, "absdiff")),
                ]
            )
        if rep == "all_numeric":
            delta = home - away
            return self._concat(
                [
                    (home, self._suffix(base_names, "home")),
                    (away, self._suffix(base_names, "away")),
                    (delta, self._suffix(base_names, "diff")),
                    (home + away, self._suffix(base_names, "sum")),
                    (home * away, self._suffix(base_names, "prod")),
                    (np.abs(delta), self._suffix(base_names, "absdiff")),
                    (
                        self._normalized_diff(home, away),
                        self._suffix(base_names, "normdiff"),
                    ),
                ]
            )
        if rep == "blockwise_diffsum":
            return self._blockwise(home_df, away_df, recipe=["diff", "sum"])
        if rep == "blockwise_diff_product":
            return self._blockwise(home_df, away_df, recipe=["diff", "prod"])
        if rep == "blockwise_full":
            return self._blockwise(home_df, away_df, recipe=["diff", "sum", "prod"])
        if rep == "off_def_cross":
            return self._off_def_cross(home_df, away_df, include_products=False)
        if rep == "off_def_cross_full":
            return self._off_def_cross(home_df, away_df, include_products=True)
        if rep == "unit_matchup":
            return self._unit_matchup(home_df, away_df, include_raw=False)
        if rep == "unit_matchup_full":
            return self._unit_matchup(home_df, away_df, include_raw=True)

        raise ValueError(f"Unsupported representation='{rep}'.")

    def _blockwise(self, home_df, away_df, recipe):
        """Internal helper for the blockwise step."""
        if not self.blocks:
            raise ValueError(
                f"representation='{self.representation}' requires blocks metadata."
            )

        chunks = []
        for block_name, block_cols in self.blocks.items():
            cols = self._resolve_block_cols(home_df, block_cols)
            h = home_df.loc[:, cols].to_numpy(dtype=float)
            a = away_df.loc[:, cols].to_numpy(dtype=float)
            names = list(cols)
            if "diff" in recipe:
                chunks.append((h - a, [f"{block_name}_{name}_diff" for name in names]))
            if "sum" in recipe:
                chunks.append((h + a, [f"{block_name}_{name}_sum" for name in names]))
            if "prod" in recipe:
                chunks.append((h * a, [f"{block_name}_{name}_prod" for name in names]))
            if "absdiff" in recipe:
                chunks.append(
                    (np.abs(h - a), [f"{block_name}_{name}_absdiff" for name in names])
                )
        return self._concat(chunks)

    def _off_def_cross(self, home_df, away_df, include_products=False):
        """Internal helper for the off_def_cross step."""
        if "offense" not in self.blocks or "defense" not in self.blocks:
            raise ValueError(
                f"representation='{self.representation}' requires offense and defense blocks."
            )

        off_cols = self._resolve_block_cols(home_df, self.blocks["offense"])
        def_cols = self._resolve_block_cols(home_df, self.blocks["defense"])
        if len(off_cols) != len(def_cols):
            raise ValueError(
                "offense and defense blocks must have the same length for cross methods."
            )

        h_off = home_df.loc[:, off_cols].to_numpy(dtype=float)
        a_off = away_df.loc[:, off_cols].to_numpy(dtype=float)
        h_def = home_df.loc[:, def_cols].to_numpy(dtype=float)
        a_def = away_df.loc[:, def_cols].to_numpy(dtype=float)

        names_home = [
            f"{off_cols[i]}_home_off_vs_away_def_diff" for i in range(len(off_cols))
        ]
        names_away = [
            f"{off_cols[i]}_away_off_vs_home_def_diff" for i in range(len(off_cols))
        ]
        chunks = [
            (h_off - a_def, names_home),
            (a_off - h_def, names_away),
        ]

        if include_products:
            prod_home = [
                f"{off_cols[i]}_home_off_x_away_def_prod" for i in range(len(off_cols))
            ]
            prod_away = [
                f"{off_cols[i]}_away_off_x_home_def_prod" for i in range(len(off_cols))
            ]
            chunks.append((h_off * a_def, prod_home))
            chunks.append((a_off * h_def, prod_away))

        for optional_block, suffix in [
            ("special_teams", "diff"),
            ("tempo", "sum"),
            ("discipline", "diff"),
            ("form", "diff"),
        ]:
            if optional_block not in self.blocks:
                continue
            cols = self._resolve_block_cols(home_df, self.blocks[optional_block])
            h = home_df.loc[:, cols].to_numpy(dtype=float)
            a = away_df.loc[:, cols].to_numpy(dtype=float)
            if suffix == "sum":
                chunks.append((h + a, [f"{optional_block}_{c}_sum" for c in cols]))
            else:
                chunks.append((h - a, [f"{optional_block}_{c}_diff" for c in cols]))

        return self._concat(chunks)

    def _unit_matchup(self, home_df, away_df, include_raw=False):
        """Build semantic unit-vs-unit matchup features.

        Each source fingerprint feature is paired with the opponent column that
        best represents its on-field counterpart. Values are first converted to
        a strength scale so positive edges consistently favor the row team:
        lower-is-better columns are multiplied by -1, higher-is-better and
        context columns keep their sign.
        """
        home_df = self._derive_unit_matchup_inputs(home_df)
        away_df = self._derive_unit_matchup_inputs(away_df)
        pairings = self._unit_pairing_specs(home_df.columns)
        passthrough_cols = self._unit_passthrough_columns(home_df.columns)
        if not pairings and not passthrough_cols:
            raise ValueError(
                "unit_matchup could not resolve any usable feature pairings."
            )

        chunks = []
        raw_cols = []
        for spec in pairings:
            source = spec["source_feature"]
            counterpart = spec["primary_opponent_counterpart"]
            if source not in home_df.columns or counterpart not in home_df.columns:
                continue

            home_source = self._strength_values(
                home_df, source, spec.get("source_direction")
            )
            away_source = self._strength_values(
                away_df, source, spec.get("source_direction")
            )
            home_counter = self._strength_values(
                home_df, counterpart, spec.get("primary_counterpart_direction")
            )
            away_counter = self._strength_values(
                away_df, counterpart, spec.get("primary_counterpart_direction")
            )

            home_edge = home_source - away_counter
            away_edge = away_source - home_counter
            net_edge = home_edge - away_edge

            chunks.extend(
                [
                    (home_edge, [f"home_{source}_vs_away_{counterpart}"]),
                    (away_edge, [f"away_{source}_vs_home_{counterpart}"]),
                    (net_edge, [f"net_{source}_vs_{counterpart}"]),
                ]
            )
            raw_cols.extend([source, counterpart])

        if passthrough_cols:
            home_raw = home_df.loc[:, passthrough_cols].to_numpy(dtype=float)
            away_raw = away_df.loc[:, passthrough_cols].to_numpy(dtype=float)
            chunks.extend(
                [
                    (home_raw, [f"home_{col}" for col in passthrough_cols]),
                    (away_raw, [f"away_{col}" for col in passthrough_cols]),
                    (home_raw - away_raw, [f"net_{col}" for col in passthrough_cols]),
                ]
            )

        if include_raw:
            raw_cols = list(dict.fromkeys(c for c in raw_cols if c in home_df.columns))
            if raw_cols:
                home_raw = home_df.loc[:, raw_cols].to_numpy(dtype=float)
                away_raw = away_df.loc[:, raw_cols].to_numpy(dtype=float)
                chunks.extend(
                    [
                        (home_raw, [f"home_{col}_raw" for col in raw_cols]),
                        (away_raw, [f"away_{col}_raw" for col in raw_cols]),
                        (home_raw - away_raw, [f"raw_{col}_diff" for col in raw_cols]),
                    ]
                )

        return self._concat(chunks)

    def _unit_passthrough_columns(self, columns):
        """Return context feature columns carried through ``unit_matchup``."""

        if not self.include_unit_passthrough:
            return []
        patterns = [
            str(pattern).strip().lower() for pattern in self.unit_passthrough_patterns
        ]
        patterns = [pattern for pattern in patterns if pattern]
        if not patterns:
            return []

        cols = []
        for column in columns:
            name = str(column)
            lowered = name.lower()
            if any(fnmatch(lowered, pattern) for pattern in patterns):
                cols.append(name)
        return cols

    def _derive_unit_matchup_inputs(self, frame):
        """Add reviewed rate features when their raw fingerprint inputs exist."""
        out = frame.copy()

        def safe_series(col):
            return pd.to_numeric(out[col], errors="coerce")

        def safe_div(numerator, denominator):
            denom = denominator.replace(0.0, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                return numerator / denom

        def add_rate(name, numerator_col, denominator_col):
            if (
                name in out.columns
                or numerator_col not in out.columns
                or denominator_col not in out.columns
            ):
                return
            out[name] = safe_div(
                safe_series(numerator_col), safe_series(denominator_col)
            )

        add_rate("statOff_first_down_rate", "statOff_first_downs", "offense_plays")
        add_rate(
            "statOff_rushing_td_rate", "statOff_rushing_tds", "statOff_rushing_attempts"
        )
        add_rate(
            "statOff_passing_td_rate", "statOff_passing_tds", "statOff_pass_attempts"
        )

        if "statOff_rush_rate" not in out.columns and {
            "statOff_rushing_attempts",
            "statOff_pass_attempts",
        }.issubset(out.columns):
            attempts = safe_series("statOff_rushing_attempts") + safe_series(
                "statOff_pass_attempts"
            )
            out["statOff_rush_rate"] = safe_div(
                safe_series("statOff_rushing_attempts"), attempts
            )
        elif "statOff_rush_rate" not in out.columns:
            add_rate("statOff_rush_rate", "statOff_rushing_attempts", "offense_plays")

        if "statOff_pass_rate" not in out.columns and {
            "statOff_rushing_attempts",
            "statOff_pass_attempts",
        }.issubset(out.columns):
            attempts = safe_series("statOff_rushing_attempts") + safe_series(
                "statOff_pass_attempts"
            )
            out["statOff_pass_rate"] = safe_div(
                safe_series("statOff_pass_attempts"), attempts
            )
        elif "statOff_pass_rate" not in out.columns:
            add_rate("statOff_pass_rate", "statOff_pass_attempts", "offense_plays")

        add_rate(
            "statDef_interception_td_rate", "statDef_interception_tds", "defense_plays"
        )
        add_rate("statDef_interception_rate", "statDef_interceptions", "defense_plays")
        add_rate(
            "statDef_passes_intercepted_rate",
            "statDef_passes_intercepted",
            "defense_plays",
        )
        add_rate("statDef_tfl_rate", "statDef_tackles_for_loss", "defense_plays")
        add_rate("statDef_tackle_rate", "statDef_tackles", "defense_plays")
        add_rate("statDef_defensive_td_rate", "statDef_defensive_tds", "defense_plays")
        add_rate("statDef_sack_rate", "statDef_sacks", "defense_plays")
        add_rate("statDef_qb_hurry_rate", "statDef_qb_hurries", "defense_plays")
        add_rate(
            "statDef_passes_deflected_rate", "statDef_passes_deflected", "defense_plays"
        )

        add_rate("statGen_turnover_rate", "statGen_turnovers", "offense_plays")
        add_rate("statGen_fumble_rate", "statGen_total_fumbles", "offense_plays")
        add_rate(
            "statGen_fumble_recovery_rate",
            "statGen_fumbles_recovered",
            "statGen_total_fumbles",
        )
        add_rate(
            "statGen_fumble_lost_rate", "statGen_fumbles_lost", "statGen_total_fumbles"
        )
        return out

    def _unit_pairing_specs(self, columns):
        """Resolve source-to-opponent unit matchup pairings for available columns."""
        available = set(columns)
        specs = self._configured_unit_pairings()
        if specs is None:
            specs = self._default_unit_pairings(columns)

        rows = []
        seen = set()
        for raw in specs:
            source = str(raw.get("source_feature", "")).strip()
            primary = str(raw.get("primary_opponent_counterpart", "")).strip()
            if (
                not source
                or not primary
                or source not in available
                or primary not in available
            ):
                continue

            candidates = [primary]
            if self.include_unit_secondary:
                secondary = str(raw.get("secondary_opponent_counterparts", "") or "")
                candidates.extend(
                    [item.strip() for item in secondary.split("|") if item.strip()]
                )

            for idx, counterpart in enumerate(candidates):
                if counterpart not in available:
                    continue
                key = (source, counterpart)
                if key in seen:
                    continue
                seen.add(key)
                source_direction = self._normalize_unit_direction(
                    raw.get("source_direction"),
                    fallback=self._feature_direction(source),
                )
                counterpart_direction = self._normalize_unit_direction(
                    raw.get("primary_counterpart_direction") if idx == 0 else None,
                    fallback=self._feature_direction(counterpart),
                )
                rows.append(
                    {
                        "source_feature": source,
                        "source_direction": source_direction,
                        "primary_opponent_counterpart": counterpart,
                        "primary_counterpart_direction": counterpart_direction,
                    }
                )
        return rows

    def _configured_unit_pairings(self):
        """Return explicit unit matchup pairings from config or CSV, if supplied."""
        if self.unit_pairings is not None:
            if isinstance(self.unit_pairings, pd.DataFrame):
                return self.unit_pairings.to_dict("records")
            if isinstance(self.unit_pairings, dict):
                return [
                    {
                        "source_feature": source,
                        "primary_opponent_counterpart": counterpart,
                    }
                    for source, counterpart in self.unit_pairings.items()
                ]
            return list(self.unit_pairings)

        if self.unit_pairings_path and self.unit_pairings_path.exists():
            return pd.read_csv(self.unit_pairings_path).to_dict("records")
        return None

    def _default_unit_pairings(self, columns):
        """Infer unit matchup pairings from canonical TDNet feature prefixes."""
        return default_unit_pairing_specs(columns)

    def _strength_values(self, frame, feature, direction=None):
        """Return a numeric column transformed so larger means better."""
        values = frame.loc[:, [feature]].to_numpy(dtype=float)
        normalized_direction = self._normalize_unit_direction(
            direction, fallback=self._feature_direction(feature)
        )
        if normalized_direction == "lower_better":
            return -values
        return values

    def _feature_direction(self, feature):
        """Infer whether larger raw values are better for a fingerprint feature."""
        return feature_direction(feature)

    def _normalize_unit_direction(self, value, fallback="direction_needs_review"):
        """Normalize direction labels from code, YAML, or CSV pairing specs."""
        if isinstance(value, float) and np.isnan(value):
            return fallback
        return normalize_unit_direction(value, fallback=fallback)

    def _resolve_block_cols(self, frame, block_cols):
        """Internal helper for the resolve_block_cols step."""
        if not isinstance(block_cols, (list, tuple)):
            raise ValueError("Block definitions must be lists or tuples.")
        if not block_cols:
            raise ValueError("Block definition is empty.")

        cols = []
        for item in block_cols:
            if isinstance(item, int):
                cols.append(frame.columns[int(item)])
            else:
                if item not in frame.columns:
                    raise ValueError(f"Block column '{item}' not found in frame.")
                cols.append(item)
        return cols

    def _safe_ratio(self, home, away):
        """Internal helper for the safe_ratio step."""
        return home / (away + self.eps)

    def _safe_log_ratio(self, home, away):
        """Internal helper for the safe_log_ratio step."""
        min_value = float(np.nanmin(np.concatenate([home.ravel(), away.ravel()])))
        shift = 0.0
        if min_value <= 0.0:
            if not self.safe_math:
                raise ValueError(
                    "log_ratio requires positive inputs unless safe_math=True."
                )
            shift = abs(min_value) + self.zero_offset
        return np.log(home + shift + self.zero_offset) - np.log(
            away + shift + self.zero_offset
        )

    def _normalized_diff(self, home, away):
        """Internal helper for the normalized_diff step."""
        return (home - away) / (np.abs(home) + np.abs(away) + self.eps)

    def _single(self, values, names, suffix):
        """Internal helper for the single step."""
        return values, self._suffix(names, suffix)

    def _suffix(self, names, suffix):
        """Internal helper for the suffix step."""
        return [f"{name}_{suffix}" for name in names]

    def _concat(self, chunks):
        """Internal helper for the concat step."""
        values = np.concatenate([chunk[0] for chunk in chunks], axis=1)
        names = []
        for _, chunk_names in chunks:
            names.extend(chunk_names)
        return values, names
