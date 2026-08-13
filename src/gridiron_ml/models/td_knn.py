"""Leakage-safe historical-matchup K-nearest-neighbor models.

The estimator is deliberately a margin regressor.  A positive margin always
means that the canonical home team won by that many points, matching the
existing TDNet matchup contract.  The fitted preprocessing pipeline is kept
inside the checkpoint, while the optional validation calibrator is fitted only
on validation predictions supplied to :meth:`train`.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .names import normalize_identifier, normalize_model_family
from .td_linear import TDLinear


KNN_DEFAULTS = {
    "uniform": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
    "distance": {"n_neighbors": 10, "weights": "distance", "metric": "euclidean"},
    "compact": {"n_neighbors": 10, "weights": "distance", "metric": "euclidean"},
    "full_fingerprint": {"n_neighbors": 10, "weights": "distance", "metric": "euclidean"},
}


class TDKNN(TDLinear):
    """Sklearn-compatible historical-matchup KNN margin model.

    ``meta_train`` and ``meta_val`` are optional extensions to the shared
    training interface.  When supplied, they make the selected historical
    neighbors auditable without changing the feature matrix or target
    semantics used by existing models.
    """

    def __init__(self, config=None):
        raw = self._load_config(config)
        # TDLinear provides the shared loss, safety, context, and persistence
        # setup.  Its estimator-specific model type is replaced immediately.
        base = dict(raw)
        base["model_type"] = "ridge"
        base["_knn_variant"] = raw.get("model_type", raw.get("variant", "distance"))
        super().__init__(base)
        self.config = raw
        self.model_family = normalize_model_family("knn")
        self.model_type = self._normalize_variant(
            raw.get("model_type", raw.get("variant", "distance"))
        )
        self.model_variant = self.model_type
        self.model_name = normalize_identifier(
            raw.get("model_name", raw.get("name", f"knn_{self.model_type}"))
        )
        self.standardize = True
        self.params = self._model_params()
        self.neighbor_meta_ = pd.DataFrame()
        self.neighbor_audit_ = pd.DataFrame()
        self.calibrator_ = None
        self.requested_n_neighbors_ = int(self.params["n_neighbors"])
        self.effective_n_neighbors_ = None

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        market_train=None,
        market_val=None,
        sample_weight=None,
        meta_train=None,
        meta_val=None,
    ):
        """Fit preprocessing, KNN, and optional validation-only calibration."""
        del market_train, market_val, sample_weight  # KNN has no row weights.
        X_train = self._coerce_feature_df(X_train)
        y_train = self._coerce_target(y_train)
        X_val = self._coerce_feature_df(X_val, allow_none=True)
        y_val = self._coerce_target(y_val, allow_none=True)
        self._assert_training_features_are_safe(X_train, X_val)

        valid_train = y_train.notna().to_numpy()
        if not valid_train.all():
            X_train = X_train.loc[valid_train].reset_index(drop=True)
            y_train = y_train.loc[valid_train].reset_index(drop=True)
            if meta_train is not None:
                meta_train = pd.DataFrame(meta_train).loc[valid_train].reset_index(drop=True)
        if len(X_train) == 0:
            raise ValueError("TDKNN requires at least one finite training target.")

        self.feature_names_ = list(X_train.columns)
        self.neighbor_meta_ = self._coerce_neighbor_meta(meta_train, len(X_train))
        self.neighbor_meta_["actual_margin"] = y_train.to_numpy(dtype=float)
        self.medians_ = X_train.median(numeric_only=True).fillna(0.0)
        self.means_ = X_train.fillna(self.medians_).mean(numeric_only=True).fillna(0.0)
        self.stds_ = (
            X_train.fillna(self.medians_).std(numeric_only=True)
            .replace(0.0, 1.0)
            .fillna(1.0)
        )

        self.effective_n_neighbors_ = min(
            self.requested_n_neighbors_, len(X_train)
        )
        self.params["n_neighbors"] = int(self.effective_n_neighbors_)
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X_train.loc[:, self.feature_names_], y_train.to_numpy(dtype=float))
        self.model_ = self.pipeline_
        self.is_trained_ = True

        # This is a Platt-style calibration on validation predictions only.
        # If validation is absent or degenerate, the inherited margin logistic
        # link remains the explicitly documented fallback.
        self.calibrator_ = None
        if X_val is not None and y_val is not None and len(X_val):
            valid_val = y_val.notna().to_numpy()
            if valid_val.sum() >= 8:
                val_margins = self._predict_margin_array(X_val.loc[valid_val])
                val_outcomes = (y_val.loc[valid_val].to_numpy(dtype=float) > 0).astype(int)
                if np.unique(val_outcomes).size == 2:
                    self.calibrator_ = LogisticRegression(
                        C=float(self.config.get("calibration_C", 1.0)),
                        solver="lbfgs",
                        max_iter=1000,
                    )
                    self.calibrator_.fit(val_margins.reshape(-1, 1), val_outcomes)

        self.training_history_ = pd.DataFrame(
            [self._history_row("train", y_train, self._predict_margin_array(X_train))]
        )
        if X_val is not None and y_val is not None and len(X_val):
            self.training_history_ = pd.concat(
                [
                    self.training_history_,
                    pd.DataFrame(
                        [self._history_row("val", y_val, self._predict_margin_array(X_val))]
                    ),
                ],
                ignore_index=True,
            )
        return self

    def predict(self, X, meta_df=None, market_df=None):
        """Return standard predictions plus compact neighbor diagnostics."""
        X = self._coerce_feature_df(X)
        details = self._neighbor_prediction_details(X, meta_df=meta_df)
        pred_df = pd.DataFrame(
            {
                "pred_margin": details["pred_margin"],
                "pred_proba_home_win": self.margin_to_probability(details["pred_margin"]),
                "pred_pick_home": (details["pred_margin"] > 0).astype(int),
            }
        )
        for key, values in details.items():
            if key != "pred_margin":
                pred_df[key] = values
        return self._attach_context(pred_df, meta_df=meta_df, market_df=market_df)

    def margin_to_probability(self, margin):
        """Use validation-fitted calibration when available, otherwise TDNet's link."""
        values = np.asarray(margin, dtype=float).reshape(-1)
        if self.calibrator_ is None:
            return super().margin_to_probability(values)
        return np.asarray(self.calibrator_.predict_proba(values.reshape(-1, 1))[:, 1], dtype=float)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.update(
            {
                "model_family": self.model_family,
                "model_type": self.model_type,
                "weights": self.params["weights"],
                "metric": self.params["metric"],
                "requested_n_neighbors": self.requested_n_neighbors_,
                "effective_n_neighbors": self.effective_n_neighbors_,
                "calibrated_on_validation_predictions": self.calibrator_ is not None,
                "neighbor_metadata_columns": list(self.neighbor_meta_.columns),
            }
        )
        return metadata

    def get_feature_importance(self):
        """Return a stable feature manifest; KNN has no native coefficients."""
        return pd.DataFrame(
            {
                "feature": list(self.feature_names_),
                "importance": np.nan,
                "feature_family": [self._feature_family(name) for name in self.feature_names_],
            }
        )

    def save(self, path):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as handle:
            pickle.dump(self, handle)
        return save_path

    @classmethod
    def load(cls, path, config=None):
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a saved {cls.__name__} instance, got {type(model).__name__}.")
        if config is not None:
            model.config = model._load_config(config)
        return model

    def _build_pipeline(self):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("estimator", KNeighborsRegressor(**self.params)),
            ]
        )

    def _model_params(self):
        variant = getattr(self, "model_type", None)
        if variant not in KNN_DEFAULTS:
            variant = self.config.get("_knn_variant", self.config.get("variant", "distance"))
            variant = self._normalize_variant(variant)
        defaults = dict(KNN_DEFAULTS[variant])
        knn_cfg = dict(self.config.get("knn", {}) or {})
        knn_cfg.update(dict(self.config.get(self.model_type, {}) or {}))
        knn_cfg.update(dict(self.config.get("params", {}) or {}))
        defaults.update({key: value for key, value in knn_cfg.items() if value is not None})
        defaults["n_neighbors"] = max(1, int(defaults["n_neighbors"]))
        defaults["weights"] = str(defaults["weights"]).lower()
        defaults["metric"] = str(defaults["metric"]).lower()
        defaults["n_jobs"] = int(defaults.get("n_jobs", -1))
        defaults.pop("model_type", None)
        defaults.pop("variant", None)
        defaults.pop("seed", None)
        defaults.pop("_knn_variant", None)
        return defaults

    @staticmethod
    def _normalize_variant(value):
        variant = normalize_identifier(value or "distance")
        aliases = {
            "uniform": "uniform",
            "knn_uniform": "uniform",
            "distance": "distance",
            "distance_weighted": "distance",
            "knn_distance": "distance",
            "compact": "compact",
            "full_fingerprint": "full_fingerprint",
        }
        if variant not in aliases:
            raise ValueError(f"Unsupported TDKNN variant='{value}'.")
        return aliases[variant]

    def _neighbor_prediction_details(self, X, meta_df=None):
        if not self.is_trained_:
            raise RuntimeError("Model is not trained. Call train() first.")
        aligned = self._feature_adapter().align_frame(X, self.feature_names_)
        transformed = self.pipeline_[:-1].transform(aligned)
        estimator = self.pipeline_.named_steps["estimator"]
        distances, indices = estimator.kneighbors(
            transformed, n_neighbors=self.effective_n_neighbors_, return_distance=True
        )
        margins = self.neighbor_meta_["actual_margin"].to_numpy(dtype=float)
        predicted = np.empty(len(aligned), dtype=float)
        rows = []
        compact = {key: [] for key in self._compact_columns()}
        for row_idx, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            neighbor_margins = margins[row_indices]
            weights = self._normalized_neighbor_weights(
                row_distances, uniform=self.params["weights"] == "uniform"
            )
            predicted_margin = float(np.sum(weights * neighbor_margins))
            predicted[row_idx] = self._cap_margin([predicted_margin])[0]
            predicted_team_home = predicted[row_idx] >= 0.0
            neighbor_wins = neighbor_margins > 0.0 if predicted_team_home else neighbor_margins < 0.0
            row_meta = self.neighbor_meta_.iloc[row_indices].copy()
            row_meta["neighbor_distance"] = row_distances
            row_meta["neighbor_weight"] = weights
            row_meta["neighbor_index"] = row_indices
            rows.append(row_meta)
            compact["selected_k"].append(int(len(row_indices)))
            compact["knn_weights"].append(self.params["weights"])
            compact["knn_metric"].append(self.params["metric"])
            compact["mean_neighbor_margin"].append(float(np.mean(neighbor_margins)))
            compact["median_neighbor_margin"].append(float(np.median(neighbor_margins)))
            compact["std_neighbor_margin"].append(float(np.std(neighbor_margins)))
            compact["min_neighbor_distance"].append(float(np.min(row_distances)))
            compact["mean_neighbor_distance"].append(float(np.mean(row_distances)))
            compact["max_neighbor_distance"].append(float(np.max(row_distances)))
            compact["neighbor_distance_ratio"].append(
                float(np.min(row_distances) / max(np.max(row_distances), 1e-12))
            )
            compact["effective_neighbor_count"].append(
                float(1.0 / np.sum(weights**2))
            )
            compact["fraction_neighbors_won_by_predicted_team"].append(
                float(np.sum(weights * neighbor_wins.astype(float)))
            )
            for field in ["game_id", "season", "week", "home_team", "away_team", "actual_margin"]:
                compact[f"neighbor_{field}s"].append(
                    json.dumps(row_meta[field].tolist(), default=str)
                )
            compact["neighbor_distances"].append(json.dumps(row_distances.tolist()))
            compact["neighbor_weights"].append(json.dumps(weights.tolist()))
        self.neighbor_audit_ = self._long_audit(rows, meta_df=meta_df)
        return {"pred_margin": predicted, **compact}

    def _long_audit(self, rows, meta_df=None):
        if not rows:
            return pd.DataFrame()
        records = []
        for prediction_row, frame in enumerate(rows):
            frame = frame.copy()
            frame.insert(0, "prediction_row", prediction_row)
            if meta_df is not None and prediction_row < len(meta_df):
                source = pd.DataFrame(meta_df).iloc[prediction_row]
                for key in ["keys_game_id", "keys_season", "keys_week", "keys_team_home", "keys_team_away"]:
                    if key in source.index:
                        frame[f"target_{key}"] = source[key]
            records.append(frame)
        return pd.concat(records, ignore_index=True)

    def _coerce_neighbor_meta(self, meta_df, length):
        frame = pd.DataFrame(meta_df).reset_index(drop=True) if meta_df is not None else pd.DataFrame(index=range(length))
        if len(frame) != length:
            raise ValueError("meta_train must align row-wise with X_train.")
        out = pd.DataFrame(index=range(length))
        out["game_id"] = self._first_existing(frame, ["keys_game_id", "game_id"], default=np.arange(length))
        out["season"] = self._first_existing(frame, ["keys_season", "season"], default=np.nan)
        out["week"] = self._first_existing(frame, ["keys_week", "week"], default=np.nan)
        out["home_team"] = self._first_existing(frame, ["keys_team_home", "home_team"], default="")
        out["away_team"] = self._first_existing(frame, ["keys_team_away", "away_team"], default="")
        out["actual_margin"] = np.nan
        return out.reset_index(drop=True)

    @staticmethod
    def _first_existing(frame, candidates, default):
        for column in candidates:
            if column in frame.columns:
                return frame[column].to_numpy(copy=True)
        if np.isscalar(default):
            return np.repeat(default, len(frame))
        return np.asarray(default)

    def _compact_columns(self):
        return [
            "selected_k", "knn_weights", "knn_metric", "mean_neighbor_margin",
            "median_neighbor_margin", "std_neighbor_margin", "min_neighbor_distance",
            "mean_neighbor_distance", "max_neighbor_distance", "neighbor_distance_ratio",
            "effective_neighbor_count", "fraction_neighbors_won_by_predicted_team",
            "neighbor_game_ids", "neighbor_seasons", "neighbor_weeks", "neighbor_home_teams",
            "neighbor_away_teams", "neighbor_actual_margins", "neighbor_distances", "neighbor_weights",
        ]

    @staticmethod
    def _normalized_neighbor_weights(distances, *, uniform=False):
        distances = np.asarray(distances, dtype=float)
        if len(distances) == 0:
            return np.array([], dtype=float)
        if uniform:
            return np.full(len(distances), 1.0 / len(distances), dtype=float)
        if np.allclose(distances, 0.0):
            weights = np.zeros(len(distances), dtype=float)
            weights[distances == 0.0] = 1.0 / max(np.sum(distances == 0.0), 1)
            return weights
        weights = 1.0 / np.maximum(distances, 1e-12)
        return weights / weights.sum()

    @staticmethod
    def _feature_family(name):
        text = str(name).lower()
        for family, markers in {
            "efficiency": ("eff", "ppa", "success"),
            "opponent_adjusted": ("opp_adj", "opponent_adjusted", "adjusted"),
            "schedule_graph": ("graph", "network", "centrality"),
            "derived_matchup": ("diff", "net", "ratio", "product"),
        }.items():
            if any(marker in text for marker in markers):
                return family
        return "basic"
