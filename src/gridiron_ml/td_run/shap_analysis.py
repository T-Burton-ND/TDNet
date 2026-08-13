"""SHAP artifact generation for TDNet model evaluations.

Usage:
    Call ``save_shap_analysis_for_models`` from evaluation code after building
    the matchup feature matrix used for model predictions.

Logic flow:
    1. Sample a small deterministic background and explanation set.
    2. Use exact linear contributions for TDLinear/TDStat-style models.
    3. Use SHAP's estimator explainer for tree pipelines.
    4. Save SHAP summary PNGs and a mean-absolute-SHAP feature table.

Responsibility:
    Keep model-explanation artifacts out of notebooks and make them available
    consistently for every evaluation run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.td_run.artifacts import ArtifactPolicy


DEFAULT_SHAP_CONFIG = {
    "enabled": False,
    "max_background": 80,
    "max_explain": 200,
    "max_display": 25,
    "random_seed": 42,
    "summary_plots": False,
    "bar_plots": False,
}


def save_shap_analysis_for_models(model_entries, X, output_dir, eval_config=None):
    """Generate SHAP artifacts for every model entry and return a status table."""
    policy = ArtifactPolicy.from_config(eval_config)
    shap_config = dict((eval_config or {}).get("shap", {}) or {})
    cfg = dict(DEFAULT_SHAP_CONFIG)
    cfg.update(shap_config)
    enabled = bool(shap_config.get("enabled", policy.shap))
    if not policy.shap or not enabled:
        return pd.DataFrame()

    X = coerce_numeric_frame(X)
    if X.empty:
        return pd.DataFrame()

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots" / "shap"
    tables_dir = output_dir / "tables" / "shap"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("random_seed", 42))
    background = sample_frame(X, int(cfg.get("max_background", 80)), seed)
    explain = sample_frame(X, int(cfg.get("max_explain", 200)), seed + 1)
    max_display = int(cfg.get("max_display", 25))
    save_summary_plots = bool(
        policy.shap_summary_plots or cfg.get("summary_plots", False)
    )
    save_bar_plots = bool(policy.shap_bar_plots or cfg.get("bar_plots", False))

    rows = []
    for idx, entry in enumerate(model_entries):
        name = str(entry.get("name") or f"model_{idx + 1}")
        model = entry.get("model")
        try:
            result = save_model_shap_artifacts(
                model=model,
                model_name=name,
                background=background,
                explain=explain,
                plots_dir=plots_dir,
                tables_dir=tables_dir,
                max_display=max_display,
                save_summary_plot=save_summary_plots,
                save_bar_plot=save_bar_plots,
            )
            rows.append({"model": name, **result})
        except Exception as exc:
            rows.append({"model": name, "status": "failed", "reason": str(exc)})
    return pd.DataFrame(rows)


def save_model_shap_artifacts(
    model,
    model_name,
    background,
    explain,
    plots_dir,
    tables_dir,
    max_display=25,
    save_summary_plot=False,
    save_bar_plot=False,
):
    """Save SHAP summary plots and importance table for one model."""
    safe = safe_name(model_name)
    explanation, feature_frame, method = build_shap_explanation(
        model, background, explain
    )
    if explanation is None:
        fallback = None
        if save_bar_plot:
            fallback = save_native_importance_plot(
                model,
                model_name,
                plots_dir / f"{safe}_feature_importance.png",
                max_display=max_display,
            )
        if fallback is None:
            return {
                "status": "skipped",
                "method": method or "unsupported",
                "reason": "No SHAP or native feature importance available.",
            }
        return {
            "status": "fallback",
            "method": "native_feature_importance",
            "summary_plot": str(fallback),
            "reason": "",
        }

    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]
    mean_abs = np.nanmean(np.abs(values), axis=0)
    total_abs = float(np.nansum(mean_abs))
    importance = pd.DataFrame(
        {
            "feature": list(explanation.feature_names),
            "mean_abs_shap": mean_abs,
            "mean_shap": np.nanmean(values, axis=0),
            "std_abs_shap": np.nanstd(np.abs(values), axis=0),
            "positive_shap_share": np.nanmean(values > 0.0, axis=0),
            "shap_importance_share": (
                mean_abs / total_abs if total_abs > 0.0 else np.zeros_like(mean_abs)
            ),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.insert(0, "shap_rank", np.arange(1, len(importance) + 1))
    table_path = tables_dir / f"{safe}_shap_importance.csv"
    importance.to_csv(table_path, index=False)

    summary_path = ""
    bar_path = ""
    if save_summary_plot:
        summary_path = plots_dir / f"{safe}_shap_summary.png"
        save_shap_summary_plot(
            explanation,
            feature_frame,
            summary_path,
            max_display=max_display,
            plot_type=None,
        )
    if save_bar_plot:
        bar_path = plots_dir / f"{safe}_shap_bar.png"
        save_shap_summary_plot(
            explanation,
            feature_frame,
            bar_path,
            max_display=max_display,
            plot_type="bar",
        )
    return {
        "status": "ok",
        "method": method,
        "summary_plot": str(summary_path),
        "bar_plot": str(bar_path),
        "importance_table": str(table_path),
        "n_explained": int(values.shape[0]),
        "n_features": int(values.shape[1]),
    }


def build_shap_explanation(model, background, explain):
    """Build a SHAP explanation for a supported TDNet model."""
    shap = import_shap()
    if shap is None:
        return None, None, "shap_not_installed"

    linear = linear_shap_inputs(model, background, explain)
    if linear is not None:
        values, data, features, base = linear
        explanation = shap.Explanation(
            values=values,
            base_values=np.repeat(float(base), values.shape[0]),
            data=data,
            feature_names=features,
        )
        return explanation, pd.DataFrame(data, columns=features), "exact_linear"

    tree = tree_shap_inputs(model, background, explain)
    if tree is not None:
        estimator, background_tx, explain_tx, features = tree
        explainer = shap.Explainer(estimator, background_tx, feature_names=features)
        raw = explainer(explain_tx)
        explanation = shap.Explanation(
            values=np.asarray(raw.values),
            base_values=getattr(raw, "base_values", None),
            data=explain_tx,
            feature_names=features,
        )
        return explanation, pd.DataFrame(explain_tx, columns=features), "tree_explainer"

    return None, None, "unsupported"


def linear_shap_inputs(model, background, explain):
    """Return exact linear SHAP-style values for linear/stat TDNet wrappers."""
    if hasattr(model, "selected_feature_names_") and hasattr(model, "beta_"):
        features = list(getattr(model, "selected_feature_names_", []) or [])
        weights = np.asarray(getattr(model, "beta_", []), dtype=float).reshape(-1)
        if not features or len(weights) != len(features):
            return None
        bg = model._transform_features(align_features(background, features))
        ex = model._transform_features(align_features(explain, features))
        center = np.nanmean(bg, axis=0)
        values = (ex - center.reshape(1, -1)) * weights.reshape(1, -1)
        base = float(getattr(model, "intercept_", 0.0)) + float(center @ weights)
        return values, ex, features, base

    features = list(getattr(model, "feature_names_", []) or [])
    weights = np.asarray(getattr(model, "weights_", []), dtype=float).reshape(-1)
    if not features or len(weights) != len(features):
        return None
    if hasattr(model, "_transform_features"):
        bg = model._transform_features(align_features(background, features))
        ex = model._transform_features(align_features(explain, features))
    else:
        bg = align_features(background, features).to_numpy(dtype=float)
        ex = align_features(explain, features).to_numpy(dtype=float)
    center = np.nanmean(bg, axis=0)
    values = (ex - center.reshape(1, -1)) * weights.reshape(1, -1)
    base = float(getattr(model, "intercept_", 0.0)) + float(center @ weights)
    return values, ex, features, base


def tree_shap_inputs(model, background, explain):
    """Return estimator and transformed matrices for tree SHAP."""
    pipeline = getattr(model, "pipeline_", None)
    features = list(getattr(model, "feature_names_", []) or [])
    if pipeline is None or not features or not hasattr(pipeline, "named_steps"):
        return None
    estimator = pipeline.named_steps.get("estimator")
    if estimator is None:
        return None
    if not hasattr(estimator, "feature_importances_"):
        return None
    preprocessor = pipeline[:-1]
    background_aligned = align_features(background, features)
    explain_aligned = align_features(explain, features)
    background_tx = np.asarray(preprocessor.transform(background_aligned), dtype=float)
    explain_tx = np.asarray(preprocessor.transform(explain_aligned), dtype=float)
    return estimator, background_tx, explain_tx, features


def save_shap_summary_plot(
    explanation, feature_frame, path, max_display=25, plot_type=None
):
    """Save one SHAP summary plot to disk."""
    shap = import_shap()
    if shap is None:
        return None
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        explanation.values,
        features=feature_frame,
        feature_names=explanation.feature_names,
        show=False,
        max_display=max_display,
        plot_type=plot_type,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def save_native_importance_plot(model, model_name, path, max_display=25):
    """Save a fallback model-native feature-importance plot."""
    if not hasattr(model, "get_feature_importance"):
        return None
    frame = pd.DataFrame(model.get_feature_importance())
    if frame.empty or "feature" not in frame.columns:
        return None
    value_col = (
        "importance"
        if "importance" in frame.columns
        else "coefficient" if "coefficient" in frame.columns else None
    )
    if value_col is None:
        return None
    frame = frame.copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce").abs()
    frame = (
        frame.dropna(subset=[value_col])
        .sort_values(value_col, ascending=False)
        .head(max_display)
    )
    if frame.empty:
        return None

    import matplotlib.pyplot as plt

    plot_df = frame.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.32 * len(plot_df) + 1.3)))
    ax.barh(
        plot_df["feature"].astype(str), plot_df[value_col], color="#1EA7FF", alpha=0.88
    )
    ax.set_title(f"{model_name} Feature Importance")
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def coerce_numeric_frame(X):
    """Coerce a feature matrix to numeric dataframe form."""
    frame = pd.DataFrame(X).copy().reset_index(drop=True)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.astype(float)


def sample_frame(frame, max_rows, random_seed):
    """Sample at most max_rows rows from a dataframe deterministically."""
    frame = pd.DataFrame(frame).reset_index(drop=True)
    if len(frame) <= int(max_rows):
        return frame.copy()
    return (
        frame.sample(n=int(max_rows), random_state=int(random_seed))
        .sort_index()
        .reset_index(drop=True)
    )


def align_features(frame, features):
    """Align a frame to feature names and keep numeric values."""
    aligned = pd.DataFrame(frame).reindex(columns=list(features))
    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
    return aligned.astype(float)


def import_shap():
    """Import SHAP lazily so tests and non-explanation workflows stay lightweight."""
    try:
        import shap
    except Exception:
        return None
    return shap


def safe_name(value):
    """Return a filesystem-safe lowercase identifier."""
    text = str(value).strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in text.split("_") if part) or "model"
