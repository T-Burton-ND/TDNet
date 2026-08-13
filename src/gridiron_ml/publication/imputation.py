"""Strict temporal donor imputation for prospective missing fingerprint fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class TemporalDonorImputer:
    """Fit a deterministic donor pool using only rows available before a target.

    The donor pool is fitted once on a training snapshot. ``transform`` applies
    a season/week precedence rule, reports donor identities and distances, and
    uses a declared training-only median/zero fallback when no donor is valid.
    """

    distance_columns: tuple[str, ...]
    value_columns: tuple[str, ...]
    donor_id_column: str = "row_id"
    season_column: str = "season"
    week_column: str = "week"
    k: int = 5

    def fit(self, donors: pd.DataFrame) -> "TemporalDonorImputer":
        required = set(self.distance_columns) | set(self.value_columns) | {
            self.donor_id_column, self.season_column, self.week_column
        }
        missing = sorted(required - set(donors.columns))
        if missing:
            raise ValueError(f"Donor frame missing columns: {missing}")
        if int(self.k) < 1:
            raise ValueError("k must be at least one.")
        frame = donors.copy().reset_index(drop=True)
        frame[self.season_column] = pd.to_numeric(frame[self.season_column], errors="coerce")
        frame[self.week_column] = pd.to_numeric(frame[self.week_column], errors="coerce")
        if frame[[self.season_column, self.week_column]].isna().any().any():
            raise ValueError("Donor season/week values must be finite.")
        for column in (*self.distance_columns, *self.value_columns):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        distance = frame.loc[:, self.distance_columns]
        self._distance_medians = distance.median().fillna(0.0)
        self._distance_scales = distance.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        self._value_fallbacks = frame.loc[:, self.value_columns].median().fillna(0.0)
        self._donors = frame
        return self

    def transform(self, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not hasattr(self, "_donors"):
            raise RuntimeError("Call fit before transform.")
        required = set(self.distance_columns) | set(self.value_columns) | {
            self.donor_id_column, self.season_column, self.week_column
        }
        missing = sorted(required - set(targets.columns))
        if missing:
            raise ValueError(f"Target frame missing columns: {missing}")
        output = targets.copy().reset_index(drop=True)
        for column in (*self.distance_columns, *self.value_columns):
            output[column] = pd.to_numeric(output[column], errors="coerce")
        audit_rows = []
        for index, target in output.iterrows():
            target_season = float(target[self.season_column])
            target_week = float(target[self.week_column])
            eligible = self._donors.loc[
                (self._donors[self.season_column] < target_season)
                | (
                    self._donors[self.season_column].eq(target_season)
                    & self._donors[self.week_column].lt(target_week)
                )
            ].copy()
            eligible = eligible.loc[eligible[self.donor_id_column].astype(str) != str(target[self.donor_id_column])]
            target_vector = pd.to_numeric(target.loc[list(self.distance_columns)], errors="coerce").fillna(self._distance_medians)
            donor_vectors = eligible.loc[:, self.distance_columns].fillna(self._distance_medians)
            if len(eligible):
                distances = np.sqrt(
                    (((donor_vectors.to_numpy(dtype=float) - target_vector.to_numpy(dtype=float)) / self._distance_scales.to_numpy(dtype=float)) ** 2).sum(axis=1)
                )
                eligible = eligible.assign(_distance=distances)
                # Stable two-pass ordering keeps distance numeric while using
                # the donor ID only as a deterministic tie-break.
                eligible = eligible.sort_values(self.donor_id_column, key=lambda column: column.astype(str), kind="mergesort")
                eligible = eligible.sort_values("_distance", kind="mergesort").head(int(self.k))
            selected_ids = [str(value) for value in eligible[self.donor_id_column]] if len(eligible) else []
            selected_distances = [float(value) for value in eligible["_distance"]] if len(eligible) else []
            imputed_fields = []
            fallback_fields = []
            uncertainty_values = []
            for column in self.value_columns:
                if pd.notna(output.at[index, column]):
                    continue
                numeric_values = pd.to_numeric(eligible[column], errors="coerce")
                valid_values = numeric_values.notna()
                values = numeric_values.loc[valid_values].to_numpy(dtype=float)
                if len(values):
                    distances = eligible.loc[valid_values, "_distance"].to_numpy(dtype=float)
                    weights = 1.0 / np.maximum(distances, 1e-12)
                    estimate = float(np.average(values, weights=weights))
                    uncertainty = float(np.sqrt(np.average((values - estimate) ** 2, weights=weights)))
                    output.at[index, column] = estimate
                    imputed_fields.append(column)
                    uncertainty_values.append(uncertainty)
                else:
                    output.at[index, column] = float(self._value_fallbacks[column])
                    fallback_fields.append(column)
            audit_rows.append(
                {
                    self.donor_id_column: target[self.donor_id_column],
                    "donor_ids_json": json.dumps(selected_ids),
                    "donor_distances_json": json.dumps(selected_distances),
                    "donor_count": len(selected_ids),
                    "imputed_fields_json": json.dumps(imputed_fields),
                    "fallback_fields_json": json.dumps(fallback_fields),
                    "fallback_used": bool(fallback_fields),
                    "imputation_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else None,
                    "availability_rule": "prior_season_or_prior_week_same_season",
                }
            )
        return output, pd.DataFrame(audit_rows)


class LeakageSafeKNNImputer:
    """K=10 training-only imputer with an explicit fit-boundary audit."""

    def __init__(self, columns: Sequence[str], *, k: int = 10):
        self.columns = tuple(columns)
        self.k = int(k)
        if self.k != 10:
            raise ValueError("Canonical TDNet imputation requires k=10.")

    def fit(self, training: pd.DataFrame, *, holdout_seasons: Sequence[int] = (), fit_boundary: str = ""):
        missing = sorted(set(self.columns) - set(training.columns))
        if missing:
            raise ValueError(f"Training frame missing imputation columns: {missing}")
        seasons = pd.to_numeric(training.get("season", pd.Series(dtype=float)), errors="coerce")
        forbidden = set(map(int, holdout_seasons)) & set(seasons.dropna().astype(int))
        if forbidden:
            raise ValueError(f"Imputer fit includes forbidden seasons: {sorted(forbidden)}")
        self.fit_rows = int(len(training))
        self.fit_boundary = str(fit_boundary)
        values = training.loc[:, self.columns].apply(pd.to_numeric, errors="coerce")
        self.scaler = StandardScaler().fit(values.fillna(values.median()))
        scaled = self.scaler.transform(values.fillna(values.median()))
        self.imputer = KNNImputer(n_neighbors=self.k).fit(scaled)
        self.fallback = values.median().fillna(0.0)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not hasattr(self, "imputer"):
            raise RuntimeError("Call fit before transform.")
        output = frame.copy()
        values = output.loc[:, self.columns].apply(pd.to_numeric, errors="coerce")
        missing = values.isna()
        scaled = self.scaler.transform(values.fillna(self.fallback))
        transformed = self.imputer.transform(scaled)
        restored = self.scaler.inverse_transform(transformed)
        output.loc[:, self.columns] = restored
        audit = pd.DataFrame({"row_index": output.index, "imputed_feature_count": missing.sum(axis=1).to_numpy(), "imputed_any": missing.any(axis=1).to_numpy(), "imputer_k": self.k, "imputer_fit_rows": self.fit_rows, "imputer_fit_boundary": self.fit_boundary})
        return output, audit
