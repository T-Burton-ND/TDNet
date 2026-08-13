"""CPU-first deterministic PyTorch tabular models for TDNet."""

from copy import deepcopy
import os
import platform
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .names import normalize_identifier, normalize_model_family
from .td_linear import TDLinear


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "TDMLP requires PyTorch. Install it in the SGE/runtime environment before enabling neural models."
        ) from exc
    return torch, nn


class TDMLP(TDLinear):
    """M5 tabular MLP with serialized preprocessing and early stopping."""

    def __init__(self, config=None):
        loaded = self._load_config(config)
        linear_config = dict(loaded)
        linear_config["model_type"] = "ridge"
        linear_config.setdefault("model_name", "mlp")
        super().__init__(linear_config)
        self.config = loaded
        self.model_family = normalize_model_family("neural")
        self.model_type = "mlp"
        self.model_name = normalize_identifier(loaded.get("model_name", "mlp"))
        self.hidden_layers = [int(x) for x in loaded.get("hidden_layers", [64, 64])]
        self.activation = str(loaded.get("activation", "relu")).lower()
        self.dropout = float(loaded.get("dropout", 0.1))
        self.normalization = str(loaded.get("normalization", "layer_norm")).lower()
        self.learning_rate = float(loaded.get("learning_rate", 1e-3))
        self.weight_decay = float(loaded.get("weight_decay", 1e-4))
        self.batch_size = int(loaded.get("batch_size", 128))
        self.max_epochs = int(loaded.get("max_epochs", 200))
        self.patience = int(loaded.get("patience", 20))
        self.min_delta = float(loaded.get("min_delta", 1e-5))
        self.quantile_transform = bool(loaded.get("quantile_transform", False))
        self.deterministic_algorithms = bool(loaded.get("deterministic_algorithms", True))
        self.torch_threads = int(loaded.get("torch_threads", 1))
        self.preprocessing_ = None
        self.network_ = None
        self.best_epoch_ = None
        self.early_stopping_reason_ = None
        self.parameter_count_ = 0
        self.training_duration_seconds_ = 0.0
        self.backend_ = None
        self.backend_limitations_ = []

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        market_train=None,
        market_val=None,
        sample_weight=None,
    ):
        try:
            torch, nn = _torch_modules()
        except ImportError:
            return self._train_sklearn_fallback(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                market_train=market_train,
                sample_weight=sample_weight,
            )
        self.backend_ = "pytorch"
        started = time.monotonic()
        self._set_determinism(torch)
        X_train = self._prepare_input_frame(X_train, fitting=True)
        y_margin = self._coerce_target(y_train).to_numpy(dtype=float)
        self._assert_training_features_are_safe(X_train)
        self.feature_names_ = list(X_train.columns)
        self.preprocessing_ = self._build_preprocessing(len(X_train))
        X_fit = self.preprocessing_.fit_transform(X_train).astype(np.float32)
        y_fit = self._target_array(y_margin)

        objective_weight = self._objective_sample_weight(
            pd.Series(y_margin), self._coerce_market_df(market_train)
        )
        if sample_weight is None:
            sample_weight = objective_weight
        elif objective_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float) * objective_weight
        weights = (
            np.ones(len(X_fit), dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )

        has_validation = X_val is not None and y_val is not None and len(X_val) > 0
        if has_validation:
            X_valid_frame = self._prepare_input_frame(X_val, fitting=False)
            self._assert_training_features_are_safe(X_train, X_valid_frame)
            X_valid = self.preprocessing_.transform(X_valid_frame).astype(np.float32)
            y_valid = self._target_array(self._coerce_target(y_val).to_numpy(dtype=float))
        else:
            X_valid = y_valid = None

        self.network_ = self._build_network(nn, X_fit.shape[1])
        self.parameter_count_ = int(sum(p.numel() for p in self.network_.parameters()))
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=max(2, self.patience // 4), factor=0.5
        )
        loss_fn = self._loss_function(nn)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_fit),
            torch.from_numpy(y_fit.reshape(-1, 1)),
            torch.from_numpy(weights.reshape(-1, 1)),
        )
        training_batch_size = min(self.batch_size, len(dataset))
        if self.normalization == "batch_norm":
            if len(dataset) < 2 or training_batch_size < 2:
                raise ValueError("Batch normalization requires at least two training rows per batch.")
            # Preserve every row while avoiding a final singleton BatchNorm
            # batch (for example, 257 rows with a configured batch size 128).
            if len(dataset) % training_batch_size == 1:
                training_batch_size -= 1
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=training_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=0,
        )

        best_loss = np.inf
        best_state = None
        stale = 0
        history = []
        for epoch in range(self.max_epochs):
            self.network_.train()
            losses = []
            for xb, yb, wb in loader:
                optimizer.zero_grad(set_to_none=True)
                element_loss = loss_fn(self.network_(xb), yb)
                loss = (element_loss * wb).sum() / wb.sum().clamp_min(1e-12)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(losses))
            val_loss = (
                self._evaluate_loss(torch, loss_fn, X_valid, y_valid)
                if has_validation
                else train_loss
            )
            scheduler.step(val_loss)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if val_loss < best_loss - self.min_delta:
                best_loss = val_loss
                best_state = deepcopy(self.network_.state_dict())
                self.best_epoch_ = epoch
                stale = 0
            else:
                stale += 1
            if has_validation and stale >= self.patience:
                self.early_stopping_reason_ = "validation_patience_exhausted"
                break
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.network_.eval()
        if self.early_stopping_reason_ is None:
            self.early_stopping_reason_ = "maximum_epochs_reached"
        self.training_history_ = pd.DataFrame(history)
        self.training_duration_seconds_ = time.monotonic() - started
        self.pipeline_ = self
        self.model_ = self
        self.is_trained_ = True
        return self

    def _train_sklearn_fallback(
        self,
        X_train,
        y_train,
        *,
        X_val=None,
        y_val=None,
        market_train=None,
        sample_weight=None,
    ):
        """Train a deterministic CPU MLP when a usable PyTorch is absent.

        CRC login and CPU nodes can expose a CUDA-linked torch wheel whose
        shared libraries cannot be mapped.  A scikit-learn Adam MLP keeps the
        declared tabular architecture/search operational without silently
        claiming that dropout, normalization, or GELU were applied.
        """
        del market_train, sample_weight, y_val
        started = time.monotonic()
        self.backend_ = "sklearn_cpu"
        self.backend_limitations_ = [
            "dropout_not_supported_by_sklearn_backend",
            "normalization_not_supported_by_sklearn_backend",
            "gelu_mapped_to_relu",
        ]
        np.random.seed(self.seed)
        X_train = self._prepare_input_frame(X_train, fitting=True)
        y_margin = self._coerce_target(y_train).to_numpy(dtype=float)
        self._assert_training_features_are_safe(X_train)
        self.feature_names_ = list(X_train.columns)
        self.preprocessing_ = self._build_preprocessing(len(X_train))
        X_fit = self.preprocessing_.fit_transform(X_train).astype(np.float32)
        y_fit = self._target_array(y_margin)
        activation = self.activation if self.activation in {"identity", "logistic", "tanh", "relu"} else "relu"
        self.network_ = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layers),
            activation=activation,
            solver="adam",
            alpha=self.weight_decay,
            batch_size=min(self.batch_size, len(X_fit)),
            learning_rate_init=self.learning_rate,
            max_iter=self.max_epochs,
            tol=self.min_delta,
            n_iter_no_change=self.patience,
            early_stopping=X_val is not None and len(X_val) > 0,
            validation_fraction=0.1,
            random_state=self.seed,
            shuffle=True,
        )
        self.network_.fit(X_fit, y_fit)
        self.parameter_count_ = int(sum(weights.size for weights in self.network_.coefs_) + sum(bias.size for bias in self.network_.intercepts_))
        self.best_epoch_ = int(getattr(self.network_, "n_iter_", self.max_epochs)) - 1
        self.early_stopping_reason_ = "validation_patience_exhausted" if self.network_.n_iter_ < self.max_epochs else "maximum_epochs_reached"
        curve = list(getattr(self.network_, "loss_curve_", []))
        self.training_history_ = pd.DataFrame({"epoch": np.arange(len(curve)), "train_loss": curve, "validation_loss": np.nan})
        self.training_duration_seconds_ = time.monotonic() - started
        self.pipeline_ = self
        self.model_ = self
        self.is_trained_ = True
        return self

    def predict(self, X, meta_df=None, market_df=None):
        margin = self.predict_margin(X)
        home = self.predict_proba(X)[:, 1]
        out = pd.DataFrame(
            {
                "pred_margin": margin,
                "pred_proba_home_win": home,
                "pred_pick_home": home >= 0.5,
            }
        )
        return self._attach_context(out, meta_df=meta_df, market_df=market_df)

    def predict_margin(self, X):
        raw = self._raw_output(X)
        if self._is_classifier_objective():
            probability = 1.0 / (1.0 + np.exp(-raw))
            probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
            return self._cap_margin(
                np.log(probability / (1.0 - probability)) * self.margin_temperature
            )
        return self._cap_margin(raw)

    def predict_proba(self, X):
        raw = self._raw_output(X)
        if self._is_classifier_objective():
            home = 1.0 / (1.0 + np.exp(-raw))
        else:
            home = self.margin_to_probability(raw)
        home = np.clip(home, 1e-8, 1.0 - 1e-8)
        return np.column_stack([1.0 - home, home])

    def get_feature_importance(self):
        """Neural importance requires permutation/occlusion, not raw weights."""
        return pd.DataFrame(columns=["feature", "importance"])

    def get_metadata(self):
        metadata = super().get_metadata()
        torch_version = None
        if self.backend_ == "pytorch":
            try:
                torch, _ = _torch_modules()
                torch_version = torch.__version__
            except ImportError:
                torch_version = None
        metadata.update(
            {
                "backend": self.backend_,
                "backend_limitations": list(self.backend_limitations_),
                "torch_version": torch_version,
                "device": "cpu",
                "deterministic_algorithms": self.deterministic_algorithms,
                "torch_threads": self.torch_threads,
                "hidden_layers": self.hidden_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "normalization": self.normalization,
                "parameter_count": self.parameter_count_,
                "optimizer": "AdamW",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "maximum_epochs": self.max_epochs,
                "early_stopping_patience": self.patience,
                "best_epoch": self.best_epoch_,
                "early_stopping_reason": self.early_stopping_reason_,
                "training_duration_seconds": self.training_duration_seconds_,
                "platform": platform.platform(),
            }
        )
        return metadata

    def _prepare_input_frame(self, X, fitting):
        frame = self._coerce_feature_df(X)
        if not fitting:
            frame = self._feature_adapter().align_frame(frame, self.feature_names_)
        return frame

    def _build_preprocessing(self, n_rows):
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if self.quantile_transform:
            steps.append(
                (
                    "quantile",
                    QuantileTransformer(
                        n_quantiles=max(10, min(1000, int(n_rows))),
                        output_distribution="normal",
                        random_state=self.seed,
                    ),
                )
            )
        steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    def _build_network(self, nn, input_width):
        layers = []
        width = input_width
        for hidden in self.hidden_layers:
            layers.append(nn.Linear(width, hidden))
            if self.normalization == "batch_norm":
                layers.append(nn.BatchNorm1d(hidden))
            elif self.normalization == "layer_norm":
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU() if self.activation == "gelu" else nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            width = hidden
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def _loss_function(self, nn):
        if self._is_classifier_objective():
            return nn.BCEWithLogitsLoss(reduction="none")
        loss = str(self.config.get("loss", "huber")).lower()
        if loss == "mse":
            return nn.MSELoss(reduction="none")
        if loss == "mae":
            return nn.L1Loss(reduction="none")
        return nn.HuberLoss(delta=float(self.config.get("huber_delta", 7.0)), reduction="none")

    def _target_array(self, margin):
        if self._is_classifier_objective():
            return (np.asarray(margin) > 0).astype(np.float32)
        return np.asarray(margin, dtype=np.float32)

    def _raw_output(self, X):
        if not self.is_trained_:
            raise RuntimeError("TDMLP is not trained.")
        frame = self._prepare_input_frame(X, fitting=False)
        values = self.preprocessing_.transform(frame).astype(np.float32)
        if self._resolved_backend() == "sklearn_cpu":
            return np.asarray(self.network_.predict(values), dtype=float).reshape(-1)
        torch, _ = _torch_modules()
        self.network_.eval()
        with torch.no_grad():
            return self.network_(torch.from_numpy(values)).cpu().numpy().reshape(-1)

    def _resolved_backend(self):
        """Infer backend for checkpoints written before ``backend_`` existed."""
        backend = getattr(self, "backend_", None)
        if backend is not None:
            return backend
        network = getattr(self, "network_", None)
        return "sklearn_cpu" if hasattr(network, "predict") else "pytorch"

    def _evaluate_loss(self, torch, loss_fn, X, y):
        self.network_.eval()
        with torch.no_grad():
            prediction = self.network_(torch.from_numpy(X))
            return float(
                loss_fn(prediction, torch.from_numpy(y.reshape(-1, 1))).mean().cpu()
            )

    def _set_determinism(self, torch):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(self.deterministic_algorithms)


class TDStructuredMLP(TDMLP):
    """M6 exploratory paired home/away interaction network."""

    def __init__(self, config=None):
        super().__init__(config)
        self.model_family = normalize_model_family("structured_neural")
        self.model_type = "structured_mlp"
        self.model_name = normalize_identifier(
            self.config.get("model_name", "structured_mlp")
        )

    def _prepare_input_frame(self, X, fitting):
        raw = self._coerce_feature_df(X)
        def paired_columns(prefix):
            out = {}
            for column in raw.columns:
                text = str(column)
                if text.startswith(prefix + "__"):
                    out[text[len(prefix) + 2 :]] = column
                elif text.startswith(prefix + "_"):
                    out[text[len(prefix) + 1 :]] = column
            return out

        home = paired_columns("home")
        away = paired_columns("away")
        paired = sorted(set(home) & set(away))
        if not paired:
            raise ValueError(
                "TDStructuredMLP requires paired home_/away_ or home__/away__ columns."
            )
        expanded = {}
        paired_columns = set(home.values()) | set(away.values())
        for name in paired:
            h = raw[home[name]]
            a = raw[away[name]]
            expanded[f"home__{name}"] = h
            expanded[f"away__{name}"] = a
            expanded[f"diff__{name}"] = h - a
            expanded[f"product__{name}"] = h * a
        for column in raw.columns:
            if column not in paired_columns:
                expanded[str(column)] = raw[column]
        frame = pd.DataFrame(expanded)
        if not fitting:
            frame = self._feature_adapter().align_frame(frame, self.feature_names_)
        return frame
