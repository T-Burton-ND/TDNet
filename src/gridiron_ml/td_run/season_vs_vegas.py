"""src.gridiron_ml.td_run.season_vs_vegas.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Evaluate model outputs, compare predictions to market baselines, and build reporting artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.td_run.market import (
    DEFAULT_VEGAS_CONVENTION,
    market_home_margin,
    normalize_vegas_frame,
)
from gridiron_ml.td_run.artifacts import (
    ArtifactPolicy,
    filter_tables_for_policy,
    print_evaluation_artifact_summary,
)
from gridiron_ml.td_run.shap_analysis import save_shap_analysis_for_models
from gridiron_ml.models import load_model_checkpoint, model_label, normalize_identifier


FAVORITE_BUCKET_BINS = [-0.001, 3, 7, 14, 21, np.inf]
FAVORITE_BUCKET_LABELS = ["0-3", "3-7", "7-14", "14-21", "21+"]
EDGE_BUCKET_BINS = [-0.001, 1, 3, 7, np.inf]
EDGE_BUCKET_LABELS = ["0-1", "1-3", "3-7", "7+"]
CONFIDENCE_BUCKET_BINS = [0.5, 0.55, 0.60, 0.70, 0.80, 0.90, 1.0]
CONFIDENCE_BUCKET_LABELS = ["50-55%", "55-60%", "60-70%", "70-80%", "80-90%", "90%+"]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval" / "model_vs_vegas.yaml"
FAMILY_STYLE = {
    "linear": {"marker": "o", "linestyle": "-"},
    "stat": {"marker": "s", "linestyle": "--"},
    "tree": {"marker": "^", "linestyle": "-."},
    "vegas": {"marker": "D", "linestyle": ":"},
}
FAMILY_COLOR_WHEELS = {
    "linear": [
        "#1EA7FF",
        "#0476D9",
        "#5FF2D2",
        "#078A78",
        "#7BA7FF",
        "#005B8F",
        "#00A8C8",
        "#79C7FF",
    ],
    "stat": [
        "#FF5FA2",
        "#D8327D",
        "#D4B56E",
        "#A98224",
        "#F28E2B",
        "#B65F00",
        "#E66C37",
        "#C44E52",
    ],
    "tree": [
        "#00C853",
        "#2E7D32",
        "#8BC34A",
        "#4E9F3D",
        "#00A676",
        "#5AA469",
        "#96CE54",
        "#367E18",
    ],
    "vegas": ["#11214F"],
    "model": ["#6A37C8", "#7E57C2", "#9C6ADE", "#455A64", "#607D8B", "#8E8E93"],
}


def evaluate_models_vs_vegas_season(
    *,
    fingerprints,
    matchup_builder,
    season,
    model_specs=None,
    models=None,
    output_dir=None,
    target_column="y_next_margin",
    margin_temperature=14.0,
    vegas_convention=None,
    make_plots=True,
    eval_config=None,
    eval_config_path=None,
):
    """Run the evaluate_models_vs_vegas_season step and return its normalized result."""
    eval_cfg = load_eval_config(
        eval_config=eval_config, eval_config_path=eval_config_path
    )
    margin_temperature = float(
        eval_cfg.get("probability", {}).get("margin_temperature", margin_temperature)
    )
    model_entries = normalize_model_entries(model_specs=model_specs, models=models)
    matchup_X, base_df = build_season_eval_frame(
        fingerprints=fingerprints,
        matchup_builder=matchup_builder,
        season=season,
        target_column=target_column,
        vegas_convention=vegas_convention,
    )
    predictions = build_prediction_table(
        model_entries, matchup_X, base_df, margin_temperature=margin_temperature
    )
    tables = compute_all_tables(predictions, eval_config=eval_cfg)
    tables["game_predictions"] = predictions
    if output_dir is not None:
        if ArtifactPolicy.from_config(eval_cfg).shap:
            shap_artifacts = save_shap_analysis_for_models(
                model_entries, matchup_X, output_dir, eval_config=eval_cfg
            )
            if not shap_artifacts.empty:
                tables["shap_artifacts"] = shap_artifacts
        tables_dir = save_metric_tables(tables, output_dir, eval_config=eval_cfg)
        if make_plots:
            save_evaluation_plots(tables, output_dir, eval_config=eval_cfg)
        print_evaluation_artifact_summary(
            output_dir=output_dir,
            tables_dir=tables_dir,
            eval_config=eval_cfg,
        )
    return tables


def load_eval_config(eval_config=None, eval_config_path=None):
    """Run the load_eval_config step and return its normalized result."""
    cfg = {}
    path = (
        Path(eval_config_path)
        if eval_config_path is not None
        else DEFAULT_EVAL_CONFIG_PATH
    )
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update(dict(eval_config or {}))
    return cfg


def normalize_model_entries(model_specs=None, models=None):
    """Run the normalize_model_entries step and return its normalized result."""
    entries = []
    for idx, model in enumerate(models or []):
        entries.append({"name": model_label(model, idx), "model": model})
    for spec in model_specs or []:
        spec = dict(spec)
        model = spec.get("model")
        checkpoint_path = spec.get("checkpoint_path") or spec.get("model_path")
        if model is None and checkpoint_path:
            model = load_model_checkpoint(checkpoint_path)
        if model is None:
            name = (
                spec.get("name")
                or spec.get("model_name")
                or f"model_{len(entries) + 1}"
            )
            raise ValueError(
                f"Model spec {name!r} must include either model or checkpoint_path/model_path."
            )
        name = normalize_identifier(
            spec.get("name") or spec.get("model_name")
        ) or model_label(model, len(entries))
        entries.append({"name": name, "model": model})
    if not entries:
        raise ValueError("At least one model or model spec is required.")
    return entries


def build_season_eval_frame(
    *,
    fingerprints,
    matchup_builder,
    season,
    target_column="y_next_margin",
    vegas_convention=None,
):
    """Run the build_season_eval_frame step and return its normalized result."""
    frame = fingerprints.frame(season=season)
    if target_column not in frame.columns:
        raise ValueError(
            f"Target column {target_column!r} is missing from fingerprint frame."
        )
    frame = frame.loc[
        pd.to_numeric(frame[target_column], errors="coerce").notna()
    ].copy()
    if frame.empty:
        raise ValueError(
            f"No completed games found for season={season} using target column {target_column!r}."
        )

    X_block, y_block, meta_df, market_df = fingerprints.split_frame(frame)
    matchup_X, matchup_meta, matchup_market = matchup_builder.matchups(
        X_block,
        meta_df,
        market_df=market_df,
        y=y_block,
    )[:3]
    actual_margin = (
        pd.to_numeric(matchup_meta.pop("y"), errors="coerce")
        if "y" in matchup_meta.columns
        else pd.Series(dtype=float)
    )

    base_df = pd.concat(
        [matchup_meta.reset_index(drop=True), matchup_market.reset_index(drop=True)],
        axis=1,
    )
    base_df = base_df.loc[:, ~base_df.columns.duplicated()].copy()
    base_df["actual_margin"] = actual_margin.reset_index(drop=True)
    base_df = normalize_vegas_frame(
        base_df, vegas_convention or DEFAULT_VEGAS_CONVENTION
    )
    base_df["vegas_implied_margin"] = market_home_margin(
        base_df, vegas_convention or DEFAULT_VEGAS_CONVENTION
    )
    base_df["actual_winner"] = winner_from_margin(base_df["actual_margin"])
    base_df["vegas_winner"] = winner_from_margin(base_df["vegas_implied_margin"])
    has_vegas = base_df["vegas_implied_margin"].notna()
    base_df["actual_is_upset"] = pd.Series(pd.NA, index=base_df.index, dtype="boolean")
    base_df.loc[has_vegas, "actual_is_upset"] = (
        base_df.loc[has_vegas, "actual_winner"]
        != base_df.loc[has_vegas, "vegas_winner"]
    ).astype("boolean")
    base_df["vegas_favorite_bucket"] = bucket_abs_margin(
        base_df["vegas_implied_margin"]
    )
    base_df["week"] = first_available_column(base_df, ["next_week", "keys_week"])
    base_df = attach_prior_record_context(base_df, frame)

    keep = base_df["actual_margin"].notna()
    return matchup_X.loc[keep].reset_index(drop=True), base_df.loc[keep].reset_index(
        drop=True
    )


def build_prediction_table(model_entries, matchup_X, base_df, margin_temperature=14.0):
    """Run the build_prediction_table step and return its normalized result."""
    out = base_df.copy().reset_index(drop=True)
    blocks = []
    for entry in model_entries:
        name = entry["name"]
        model = entry["model"]
        pred_df = model.predict(matchup_X)
        if "pred_margin" not in pred_df.columns:
            raise ValueError(f"Model {name!r} predict() output is missing pred_margin.")
        margin = pd.to_numeric(pred_df["pred_margin"], errors="coerce").reset_index(
            drop=True
        )
        proba = prediction_probability(model, pred_df, margin, margin_temperature)
        winner = winner_from_margin(margin)
        called_upset = (winner != out["vegas_winner"]).astype("boolean")
        called_upset = called_upset.where(out["vegas_implied_margin"].notna(), pd.NA)
        picked_worse_record = (winner == out["worse_record_side"]).astype("boolean")
        picked_worse_record = picked_worse_record.where(
            out["worse_record_side"].isin(["home", "away"]), pd.NA
        )
        confidence = np.maximum(proba, 1.0 - proba)
        blocks.append(
            pd.DataFrame(
                {
                    f"{name}__pred_margin": margin,
                    f"{name}__pred_win_prob": proba,
                    f"{name}__winner": winner,
                    f"{name}__error": margin - out["actual_margin"],
                    f"{name}__correct": winner == out["actual_winner"],
                    f"{name}__edge_vs_vegas": margin - out["vegas_implied_margin"],
                    f"{name}__called_upset": called_upset,
                    f"{name}__correct_upset": boolean_mask(out["actual_is_upset"])
                    & boolean_mask(called_upset),
                    f"{name}__picked_worse_record": picked_worse_record,
                    f"{name}__favorite_bucket": bucket_abs_margin(margin),
                    f"{name}__confidence": confidence,
                    f"{name}__confidence_bucket": pd.cut(
                        confidence,
                        bins=CONFIDENCE_BUCKET_BINS,
                        labels=CONFIDENCE_BUCKET_LABELS,
                        include_lowest=True,
                    ),
                },
                index=out.index,
            )
        )
    if "vegas_implied_margin" in out.columns:
        vegas_picked_worse_record = (
            out["vegas_winner"] == out["worse_record_side"]
        ).astype("boolean")
        vegas_picked_worse_record = vegas_picked_worse_record.where(
            out["worse_record_side"].isin(["home", "away"]), pd.NA
        )
        blocks.append(
            pd.DataFrame(
                {
                    "Vegas__pred_margin": out["vegas_implied_margin"],
                    "Vegas__winner": out["vegas_winner"],
                    "Vegas__error": out["vegas_implied_margin"] - out["actual_margin"],
                    "Vegas__correct": out["vegas_winner"] == out["actual_winner"],
                    "Vegas__picked_worse_record": vegas_picked_worse_record,
                },
                index=out.index,
            )
        )
    if not blocks:
        return out.copy()
    return pd.concat([out, *blocks], axis=1).copy()


def compute_all_tables(predictions, eval_config=None):
    """Run the compute_all_tables step and return its normalized result."""
    policy = ArtifactPolicy.from_config(eval_config)
    sources = prediction_sources(predictions)
    model_sources = [s for s in sources if s != "Vegas"]
    weights = dict((eval_config or {}).get("scoring_weights", {}))
    tables = {
        "model_score_matrix": model_score_matrix(predictions, sources, weights=weights),
        "overall_margin_metrics": overall_margin_metrics(predictions, sources),
        "margin_diagnostics": margin_diagnostics(predictions, sources),
        "overall_winner_metrics": overall_winner_metrics(predictions, sources),
        "overall_vegas_alignment_metrics": vegas_alignment_metrics(
            predictions, model_sources
        ),
        "winner_breakdown_counts": winner_breakdown_counts(predictions, sources),
    }
    if policy.weekly_tables:
        tables.update(
            {
                "weekly_mae": grouped_metric_table(
                    predictions, sources, "week", margin_mae
                ),
                "weekly_rmse": grouped_metric_table(
                    predictions, sources, "week", margin_rmse
                ),
                "weekly_winner_accuracy": grouped_metric_table(
                    predictions, sources, "week", winner_accuracy
                ),
                "weekly_disagreement_accuracy": grouped_metric_table(
                    predictions, model_sources, "week", disagreement_accuracy
                ),
                "weekly_vegas_alignment_accuracy": grouped_vegas_alignment_table(
                    predictions, model_sources, "week"
                ),
                "weekly_against_vegas_edge_3_plus_accuracy": grouped_metric_table(
                    predictions,
                    model_sources,
                    "week",
                    lambda d, s: against_vegas_edge_accuracy(d, s, threshold=3.0),
                ),
                "weekly_record_upset_recall": grouped_metric_table(
                    predictions, sources, "week", record_upset_recall
                ),
                "weekly_edge_3_plus_accuracy": grouped_metric_table(
                    predictions, model_sources, "week", edge_3_plus_accuracy
                ),
            }
        )
    if policy.bucket_tables:
        tables.update(
            {
                "vegas_spread_bucket_mae": grouped_metric_table(
                    predictions, sources, "vegas_favorite_bucket", margin_mae
                ),
                "vegas_spread_bucket_rmse": grouped_metric_table(
                    predictions, sources, "vegas_favorite_bucket", margin_rmse
                ),
                "vegas_spread_bucket_winner_accuracy": grouped_metric_table(
                    predictions, sources, "vegas_favorite_bucket", winner_accuracy
                ),
                "vegas_spread_bucket_upset_recall": grouped_metric_table(
                    predictions, model_sources, "vegas_favorite_bucket", upset_recall
                ),
                "vegas_spread_bucket_disagreement_accuracy": grouped_metric_table(
                    predictions,
                    model_sources,
                    "vegas_favorite_bucket",
                    disagreement_accuracy,
                ),
                "vegas_spread_bucket_edge_3_plus_accuracy": grouped_metric_table(
                    predictions,
                    model_sources,
                    "vegas_favorite_bucket",
                    edge_3_plus_accuracy,
                ),
                "model_favorite_bucket_winner_accuracy": model_bucket_table(
                    predictions, model_sources, winner_accuracy
                ),
                "model_favorite_bucket_mae": model_bucket_table(
                    predictions, model_sources, margin_mae
                ),
            }
        )
    if policy.calibration_tables:
        tables.update(
            {
                "confidence_bucket_accuracy": confidence_bucket_table(
                    predictions, model_sources, "accuracy"
                ),
                "confidence_bucket_calibration": confidence_bucket_table(
                    predictions, model_sources, "calibration"
                ),
            }
        )
    if policy.ats_tables:
        tables.update(
            {
                "ats_summary": ats_summary(predictions, model_sources),
                "ats_by_edge_bucket": ats_by_edge_bucket(predictions, model_sources),
            }
        )
    if policy.prediction_sanity:
        tables["prediction_sanity"] = prediction_sanity(predictions, model_sources)
    return tables


def model_score_matrix(df, sources, weights=None):
    """Run the model_score_matrix step and return its normalized result."""
    weights = dict(
        {
            "winner_accuracy": 0.45,
            "chalk_accuracy": 0.25,
            "upset_recall": 0.25,
            "disagreement_accuracy": 0.0,
            "edge_3_plus_accuracy": 0.0,
            "record_upset_recall": 0.0,
            "margin_score": 0.05,
        },
        **dict(weights or {}),
    )
    winner = overall_winner_metrics(df, sources).set_index("metric")
    margin = overall_margin_metrics(df, sources).set_index("metric")
    rows = []

    mae_values = {source: safe_lookup(margin, "mae", source) for source in sources}
    rmse_values = {source: safe_lookup(margin, "rmse", source) for source in sources}
    mae_scores = inverse_minmax_scores(mae_values)
    rmse_scores = inverse_minmax_scores(rmse_values)

    for source in sources:
        if source == "Vegas":
            source_type = "market"
        else:
            source_type = "model"
        winner_accuracy_value = safe_lookup(winner, "winner_accuracy", source)
        chalk_accuracy_value = safe_lookup(winner, "chalk_accuracy", source)
        upset_recall_value = safe_lookup(winner, "upset_recall", source)
        disagreement_accuracy_value = safe_lookup(
            winner, "disagreement_accuracy", source
        )
        edge_3_plus_accuracy_value = safe_lookup(winner, "edge_3_plus_accuracy", source)
        record_upset_recall_value = safe_lookup(winner, "record_upset_recall", source)
        margin_score_value = np.nanmean(
            [mae_scores.get(source, np.nan), rmse_scores.get(source, np.nan)]
        )

        total_score = (
            weights["winner_accuracy"] * fill_score(winner_accuracy_value)
            + weights["chalk_accuracy"] * fill_score(chalk_accuracy_value)
            + weights["upset_recall"] * fill_score(upset_recall_value)
            + weights["disagreement_accuracy"] * fill_score(disagreement_accuracy_value)
            + weights["edge_3_plus_accuracy"] * fill_score(edge_3_plus_accuracy_value)
            + weights["record_upset_recall"] * fill_score(record_upset_recall_value)
            + weights["margin_score"] * fill_score(margin_score_value)
        )
        rows.append(
            {
                "source": source,
                "source_type": source_type,
                "total_score": float(total_score),
                "winner_accuracy": winner_accuracy_value,
                "chalk_accuracy": chalk_accuracy_value,
                "upset_recall": upset_recall_value,
                "disagreement_accuracy": disagreement_accuracy_value,
                "edge_3_plus_accuracy": edge_3_plus_accuracy_value,
                "record_upset_recall": record_upset_recall_value,
                "mae": mae_values.get(source),
                "rmse": rmse_values.get(source),
                "mae_score": mae_scores.get(source),
                "rmse_score": rmse_scores.get(source),
                "margin_score": margin_score_value,
                "winner_weight": weights["winner_accuracy"],
                "chalk_weight": weights["chalk_accuracy"],
                "upset_recall_weight": weights["upset_recall"],
                "disagreement_weight": weights["disagreement_accuracy"],
                "edge_3_plus_weight": weights["edge_3_plus_accuracy"],
                "record_upset_recall_weight": weights["record_upset_recall"],
                "margin_weight": weights["margin_score"],
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values("total_score", ascending=False)
        .reset_index(drop=True)
    )
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def overall_margin_metrics(df, sources):
    """Run the overall_margin_metrics step and return its normalized result."""
    metrics = {
        "n": lambda d, s: valid_source_frame(d, s).shape[0],
        "mae": margin_mae,
        "rmse": margin_rmse,
        "median_absolute_error": median_absolute_error,
        "abs_error_p75": lambda d, s: abs_error_percentile(d, s, 75),
        "abs_error_p90": lambda d, s: abs_error_percentile(d, s, 90),
        "max_absolute_error": max_absolute_error,
        "bias": margin_bias,
        "error_std": error_std,
        "correlation": margin_correlation,
        "spearman_correlation": margin_spearman_correlation,
        "r2": margin_r2,
        "calibration_slope": margin_calibration_slope,
        "calibration_intercept": margin_calibration_intercept,
        "within_3": lambda d, s: within_margin(d, s, 3.0),
        "within_7": lambda d, s: within_margin(d, s, 7.0),
        "within_10": lambda d, s: within_margin(d, s, 10.0),
        "within_14": lambda d, s: within_margin(d, s, 14.0),
    }
    return metric_rows(df, sources, metrics)


def margin_diagnostics(df, sources):
    """Run the margin_diagnostics step and return its normalized result."""
    rows = []
    for source in sources:
        rows.append(
            {
                "source": source,
                "n": valid_source_frame(df, source).shape[0],
                "mae": margin_mae(df, source),
                "rmse": margin_rmse(df, source),
                "median_absolute_error": median_absolute_error(df, source),
                "abs_error_p75": abs_error_percentile(df, source, 75),
                "abs_error_p90": abs_error_percentile(df, source, 90),
                "max_absolute_error": max_absolute_error(df, source),
                "bias": margin_bias(df, source),
                "error_std": error_std(df, source),
                "correlation": margin_correlation(df, source),
                "spearman_correlation": margin_spearman_correlation(df, source),
                "r2": margin_r2(df, source),
                "calibration_slope": margin_calibration_slope(df, source),
                "calibration_intercept": margin_calibration_intercept(df, source),
                "within_3": within_margin(df, source, 3.0),
                "within_7": within_margin(df, source, 7.0),
                "within_10": within_margin(df, source, 10.0),
                "within_14": within_margin(df, source, 14.0),
            }
        )
    return pd.DataFrame(rows)


def overall_winner_metrics(df, sources):
    """Run the overall_winner_metrics step and return its normalized result."""
    metrics = {
        "winner_accuracy": winner_accuracy,
        "chalk_accuracy": chalk_accuracy,
        "upset_recall": upset_recall,
        "upset_precision": upset_precision,
        "disagreement_accuracy": disagreement_accuracy,
        "disagreement_count": disagreement_count,
        "edge_1_plus_accuracy": edge_1_plus_accuracy,
        "edge_3_plus_accuracy": edge_3_plus_accuracy,
        "edge_7_plus_accuracy": edge_7_plus_accuracy,
        "margin_edge_1_plus_winner_accuracy": lambda d, s: margin_edge_winner_accuracy(
            d, s, threshold=1.0
        ),
        "margin_edge_1_plus_count": lambda d, s: margin_edge_count(d, s, threshold=1.0),
        "contrarian_edge_1_plus_accuracy": lambda d, s: contrarian_edge_accuracy(
            d, s, threshold=1.0
        ),
        "contrarian_edge_1_plus_count": lambda d, s: contrarian_edge_count(
            d, s, threshold=1.0
        ),
        "contrarian_edge_1_plus_vegas_accuracy": lambda d, s: contrarian_edge_vegas_accuracy(
            d, s, threshold=1.0
        ),
        "contrarian_edge_1_plus_model_minus_vegas": lambda d, s: contrarian_edge_model_minus_vegas(
            d, s, threshold=1.0
        ),
        "margin_edge_3_plus_winner_accuracy": lambda d, s: margin_edge_winner_accuracy(
            d, s, threshold=3.0
        ),
        "margin_edge_3_plus_count": lambda d, s: margin_edge_count(d, s, threshold=3.0),
        "contrarian_edge_3_plus_accuracy": lambda d, s: contrarian_edge_accuracy(
            d, s, threshold=3.0
        ),
        "contrarian_edge_3_plus_count": lambda d, s: contrarian_edge_count(
            d, s, threshold=3.0
        ),
        "contrarian_edge_3_plus_vegas_accuracy": lambda d, s: contrarian_edge_vegas_accuracy(
            d, s, threshold=3.0
        ),
        "contrarian_edge_3_plus_model_minus_vegas": lambda d, s: contrarian_edge_model_minus_vegas(
            d, s, threshold=3.0
        ),
        "margin_edge_7_plus_winner_accuracy": lambda d, s: margin_edge_winner_accuracy(
            d, s, threshold=7.0
        ),
        "margin_edge_7_plus_count": lambda d, s: margin_edge_count(d, s, threshold=7.0),
        "contrarian_edge_7_plus_accuracy": lambda d, s: contrarian_edge_accuracy(
            d, s, threshold=7.0
        ),
        "contrarian_edge_7_plus_count": lambda d, s: contrarian_edge_count(
            d, s, threshold=7.0
        ),
        "contrarian_edge_7_plus_vegas_accuracy": lambda d, s: contrarian_edge_vegas_accuracy(
            d, s, threshold=7.0
        ),
        "contrarian_edge_7_plus_model_minus_vegas": lambda d, s: contrarian_edge_model_minus_vegas(
            d, s, threshold=7.0
        ),
        "record_upset_recall": record_upset_recall,
        "record_underdog_accuracy": record_underdog_accuracy,
        "rank_upset_recall": rank_upset_recall,
        "brier_score": brier_score,
        "log_loss": log_loss,
    }
    return metric_rows(df, sources, metrics)


def winner_breakdown_counts(df, sources):
    """Run the winner_breakdown_counts step and return its normalized result."""
    metrics = {
        "games_with_vegas": lambda d, s: int(
            valid_source_frame(d, s, require_vegas=True).shape[0]
        ),
        "correct_winners": lambda d, s: int(
            boolean_mask(source_series(d, s, "correct")).sum()
        ),
        "actual_upsets": lambda d, s: int(
            boolean_mask(
                valid_source_frame(d, s, require_vegas=True)["actual_is_upset"]
            ).sum()
        ),
        "called_upsets": lambda d, s: (
            int(
                boolean_mask(
                    source_series(
                        valid_source_frame(d, s, require_vegas=True), s, "called_upset"
                    )
                ).sum()
            )
            if s != "Vegas"
            else np.nan
        ),
        "correct_upsets": lambda d, s: (
            int(
                boolean_mask(
                    source_series(
                        valid_source_frame(d, s, require_vegas=True), s, "correct_upset"
                    )
                ).sum()
            )
            if s != "Vegas"
            else np.nan
        ),
        "missed_upsets": missed_upsets_count,
        "false_upset_calls": false_upsets_count,
    }
    return metric_rows(df, sources, metrics)


def metric_rows(df, sources, metric_funcs):
    """Run the metric_rows step and return its normalized result."""
    rows = []
    for metric, func in metric_funcs.items():
        row = {"metric": metric}
        for source in sources:
            try:
                row[source] = func(df, source)
            except Exception:
                row[source] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def safe_lookup(table, metric, source):
    """Run the safe_lookup step and return its normalized result."""
    if metric not in table.index or source not in table.columns:
        return np.nan
    value = pd.to_numeric(pd.Series([table.loc[metric, source]]), errors="coerce").iloc[
        0
    ]
    return float(value) if pd.notna(value) else np.nan


def fill_score(value):
    """Run the fill_score step and return its normalized result."""
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def inverse_minmax_scores(values):
    """Run the inverse_minmax_scores step and return its normalized result."""
    finite = {
        key: float(value)
        for key, value in values.items()
        if pd.notna(value) and np.isfinite(value)
    }
    if not finite:
        return {key: np.nan for key in values}
    low = min(finite.values())
    high = max(finite.values())
    if np.isclose(low, high):
        return {key: 1.0 if key in finite else np.nan for key in values}
    return {
        key: (high - finite[key]) / (high - low) if key in finite else np.nan
        for key in values
    }


def grouped_metric_table(df, sources, group_col, metric_func):
    """Run the grouped_metric_table step and return its normalized result."""
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for group_value, group in df.groupby(group_col, observed=False, dropna=False):
        row = {group_col: group_value}
        for source in sources:
            row[source] = metric_func(group, source)
        rows.append(row)
    return pd.DataFrame(rows)


def model_bucket_table(df, model_sources, metric_func):
    """Run the model_bucket_table step and return its normalized result."""
    rows = []
    for source in model_sources:
        bucket_col = f"{source}__favorite_bucket"
        if bucket_col not in df.columns:
            continue
        for bucket, group in df.groupby(bucket_col, observed=False):
            rows.append(
                {
                    "favorite_bucket": bucket,
                    "model": source,
                    "value": metric_func(group, source),
                }
            )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df
    return long_df.pivot_table(
        index="favorite_bucket", columns="model", values="value", aggfunc="first"
    ).reset_index()


def confidence_bucket_table(df, model_sources, metric):
    """Run the confidence_bucket_table step and return its normalized result."""
    rows = []
    for source in model_sources:
        bucket_col = f"{source}__confidence_bucket"
        prob_col = f"{source}__pred_win_prob"
        if bucket_col not in df.columns or prob_col not in df.columns:
            continue
        for bucket, group in df.groupby(bucket_col, observed=False):
            if metric == "accuracy":
                value = winner_accuracy(group, source)
            else:
                actual = (group["actual_margin"] > 0).astype(float)
                diff = pd.to_numeric(group[prob_col], errors="coerce") - actual
                value = float(diff.mean()) if diff.notna().any() else np.nan
            rows.append({"confidence_bucket": bucket, "model": source, "value": value})
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df
    return long_df.pivot_table(
        index="confidence_bucket", columns="model", values="value", aggfunc="first"
    ).reset_index()


def ats_summary(df, model_sources):
    """Run the ats_summary step and return its normalized result."""
    metrics = {
        "ats_accuracy": ats_accuracy,
        "push_rate": ats_push_rate,
        "mean_edge": mean_edge,
    }
    return metric_rows(df, model_sources, metrics)


def ats_by_edge_bucket(df, model_sources):
    """Run the ats_by_edge_bucket step and return its normalized result."""
    rows = []
    for source in model_sources:
        if "vegas_implied_margin" not in df.columns:
            continue
        edge = source_series(df, source, "edge_vs_vegas").abs()
        bucket = pd.cut(edge, bins=EDGE_BUCKET_BINS, labels=EDGE_BUCKET_LABELS)
        temp = df.copy()
        temp["edge_bucket"] = bucket
        for bucket_value, group in temp.groupby("edge_bucket", observed=False):
            rows.append(
                {
                    "edge_bucket": bucket_value,
                    "model": source,
                    "ats_accuracy": ats_accuracy(group, source),
                }
            )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df
    return long_df.pivot_table(
        index="edge_bucket", columns="model", values="ats_accuracy", aggfunc="first"
    ).reset_index()


def prediction_sanity(df, model_sources):
    """Run the prediction_sanity step and return its normalized result."""
    rows = []
    for source in model_sources:
        pred = source_series(df, source, "pred_margin").dropna()
        rows.append(
            {
                "metric": source,
                "n": int(len(pred)),
                "min_margin": float(pred.min()) if len(pred) else np.nan,
                "max_margin": float(pred.max()) if len(pred) else np.nan,
                "mean_margin": float(pred.mean()) if len(pred) else np.nan,
                "std_margin": float(pred.std(ddof=1)) if len(pred) > 1 else np.nan,
                "unique_predictions": int(pred.nunique()) if len(pred) else 0,
                "positive_margins": int((pred > 0).sum()) if len(pred) else 0,
                "negative_margins": int((pred < 0).sum()) if len(pred) else 0,
            }
        )
    return pd.DataFrame(rows)


def margin_mae(df, source):
    """Run the margin_mae step and return its normalized result."""
    d = valid_source_frame(df, source)
    return (
        float(np.nanmean(np.abs(source_series(d, source, "error"))))
        if len(d)
        else np.nan
    )


def margin_rmse(df, source):
    """Run the margin_rmse step and return its normalized result."""
    d = valid_source_frame(df, source)
    err = source_series(d, source, "error")
    return float(np.sqrt(np.nanmean(err**2))) if len(d) else np.nan


def median_absolute_error(df, source):
    """Run the median_absolute_error step and return its normalized result."""
    d = valid_source_frame(df, source)
    return (
        float(np.nanmedian(np.abs(source_series(d, source, "error"))))
        if len(d)
        else np.nan
    )


def abs_error_percentile(df, source, percentile):
    """Run the abs_error_percentile step and return its normalized result."""
    d = valid_source_frame(df, source)
    err = np.abs(source_series(d, source, "error").dropna())
    return float(np.nanpercentile(err, percentile)) if len(err) else np.nan


def max_absolute_error(df, source):
    """Run the max_absolute_error step and return its normalized result."""
    d = valid_source_frame(df, source)
    err = np.abs(source_series(d, source, "error").dropna())
    return float(np.nanmax(err)) if len(err) else np.nan


def margin_bias(df, source):
    """Run the margin_bias step and return its normalized result."""
    d = valid_source_frame(df, source)
    return float(np.nanmean(source_series(d, source, "error"))) if len(d) else np.nan


def error_std(df, source):
    """Run the error_std step and return its normalized result."""
    d = valid_source_frame(df, source)
    err = source_series(d, source, "error")
    return float(np.nanstd(err, ddof=1)) if len(err.dropna()) > 1 else np.nan


def margin_correlation(df, source):
    """Run the margin_correlation step and return its normalized result."""
    d = valid_source_frame(df, source)
    return (
        float(
            source_series(d, source, "pred_margin").corr(
                pd.to_numeric(d["actual_margin"], errors="coerce")
            )
        )
        if len(d)
        else np.nan
    )


def margin_spearman_correlation(df, source):
    """Run the margin_spearman_correlation step and return its normalized result."""
    d = valid_source_frame(df, source)
    if len(d) < 2:
        return np.nan
    return float(
        source_series(d, source, "pred_margin").corr(
            pd.to_numeric(d["actual_margin"], errors="coerce"), method="spearman"
        )
    )


def margin_r2(df, source):
    """Run the margin_r2 step and return its normalized result."""
    d = valid_source_frame(df, source)
    if len(d) < 2:
        return np.nan
    actual = pd.to_numeric(d["actual_margin"], errors="coerce").to_numpy(dtype=float)
    pred = source_series(d, source, "pred_margin").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[valid]
    pred = pred[valid]
    if len(actual) < 2:
        return np.nan
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan


def margin_calibration_slope(df, source):
    """Run the margin_calibration_slope step and return its normalized result."""
    fit = margin_calibration_fit(df, source)
    return fit["slope"]


def margin_calibration_intercept(df, source):
    """Run the margin_calibration_intercept step and return its normalized result."""
    fit = margin_calibration_fit(df, source)
    return fit["intercept"]


def margin_calibration_fit(df, source):
    """Run the margin_calibration_fit step and return its normalized result."""
    d = valid_source_frame(df, source)
    if len(d) < 2:
        return {"slope": np.nan, "intercept": np.nan}
    actual = pd.to_numeric(d["actual_margin"], errors="coerce").to_numpy(dtype=float)
    pred = source_series(d, source, "pred_margin").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[valid]
    pred = pred[valid]
    if len(actual) < 2 or np.nanstd(pred) <= 1e-12:
        return {"slope": np.nan, "intercept": np.nan}
    slope, intercept = np.polyfit(pred, actual, deg=1)
    return {"slope": float(slope), "intercept": float(intercept)}


def within_margin(df, source, threshold):
    """Run the within_margin step and return its normalized result."""
    d = valid_source_frame(df, source)
    err = np.abs(source_series(d, source, "error").dropna())
    return float((err <= float(threshold)).mean()) if len(err) else np.nan


def winner_accuracy(df, source):
    """Run the winner_accuracy step and return its normalized result."""
    d = valid_source_frame(df, source)
    correct = source_series(d, source, "correct")
    return float(correct.mean()) if len(correct.dropna()) else np.nan


def chalk_accuracy(df, source):
    """Run the chalk_accuracy step and return its normalized result."""
    d = valid_source_frame(df, source, require_vegas=True)
    if d.empty or "actual_is_upset" not in d.columns:
        return np.nan
    chalk = d["actual_is_upset"] == False
    if not chalk.any():
        return np.nan
    return float(source_series(d.loc[chalk], source, "correct").mean())


def upset_recall(df, source):
    """Run the upset_recall step and return its normalized result."""
    if source == "Vegas":
        return 0.0
    d = valid_source_frame(df, source, require_vegas=True)
    actual_upset = d["actual_is_upset"] == True
    if not actual_upset.any():
        return np.nan
    return float(source_series(d.loc[actual_upset], source, "called_upset").mean())


def upset_precision(df, source):
    """Run the upset_precision step and return its normalized result."""
    if source == "Vegas":
        return np.nan
    d = valid_source_frame(df, source, require_vegas=True)
    called = source_series(d, source, "called_upset") == True
    if not called.any():
        return np.nan
    return float((d.loc[called, "actual_is_upset"] == True).mean())


def disagreement_accuracy(df, source):
    """Run the disagreement_accuracy step and return its normalized result."""
    if source == "Vegas":
        return np.nan
    d = valid_source_frame(df, source, require_vegas=True)
    if d.empty:
        return np.nan
    disagreed = source_series(d, source, "winner") != d["vegas_winner"]
    disagreed = disagreed & d["vegas_winner"].notna()
    if not disagreed.any():
        return np.nan
    return float(source_series(d.loc[disagreed], source, "correct").mean())


def disagreement_count(df, source):
    """Run the disagreement_count step and return its normalized result."""
    if source == "Vegas":
        return np.nan
    d = valid_source_frame(df, source, require_vegas=True)
    if d.empty:
        return 0
    disagreed = source_series(d, source, "winner") != d["vegas_winner"]
    disagreed = disagreed & d["vegas_winner"].notna()
    return int(disagreed.sum())


def edge_1_plus_accuracy(df, source):
    """Run the edge_1_plus_accuracy step and return its normalized result."""
    return edge_accuracy(df, source, threshold=1.0)


def edge_3_plus_accuracy(df, source):
    """Run the edge_3_plus_accuracy step and return its normalized result."""
    return edge_accuracy(df, source, threshold=3.0)


def edge_7_plus_accuracy(df, source):
    """Run the edge_7_plus_accuracy step and return its normalized result."""
    return edge_accuracy(df, source, threshold=7.0)


def edge_accuracy(df, source, threshold):
    """Run the edge_accuracy step and return its normalized result."""
    return margin_edge_winner_accuracy(df, source, threshold)


def margin_edge_winner_accuracy(df, source, threshold):
    """Run the margin_edge_winner_accuracy step and return its normalized result."""
    if source == "Vegas":
        return np.nan
    d = valid_source_frame(df, source, require_vegas=True)
    if d.empty:
        return np.nan
    edge = source_series(d, source, "edge_vs_vegas").abs()
    mask = edge >= float(threshold)
    if not mask.any():
        return np.nan
    return float(source_series(d.loc[mask], source, "correct").mean())


def margin_edge_count(df, source, threshold):
    """Run the margin_edge_count step and return its normalized result."""
    if source == "Vegas":
        return np.nan
    d = valid_source_frame(df, source, require_vegas=True)
    if d.empty:
        return 0
    edge = source_series(d, source, "edge_vs_vegas").abs()
    return int((edge >= float(threshold)).sum())


def contrarian_edge_mask(df, source, threshold):
    """Run the contrarian_edge_mask step and return its normalized result."""
    d = valid_source_frame(df, source, require_vegas=True)
    if d.empty or source == "Vegas":
        return d, pd.Series(False, index=d.index)
    edge = source_series(d, source, "edge_vs_vegas").abs()
    disagreed = source_series(d, source, "winner") != d["vegas_winner"]
    disagreed = disagreed & d["vegas_winner"].notna()
    return d, (edge >= float(threshold)) & disagreed


def contrarian_edge_accuracy(df, source, threshold):
    """Run the contrarian_edge_accuracy step and return its normalized result."""
    d, mask = contrarian_edge_mask(df, source, threshold)
    if not mask.any():
        return np.nan
    return float(source_series(d.loc[mask], source, "correct").mean())


def contrarian_edge_count(df, source, threshold):
    """Run the contrarian_edge_count step and return its normalized result."""
    d, mask = contrarian_edge_mask(df, source, threshold)
    if d.empty and source == "Vegas":
        return np.nan
    return int(mask.sum())


def contrarian_edge_vegas_accuracy(df, source, threshold):
    """Run the contrarian_edge_vegas_accuracy step and return its normalized result."""
    d, mask = contrarian_edge_mask(df, source, threshold)
    if not mask.any():
        return np.nan
    return float((d.loc[mask, "vegas_winner"] == d.loc[mask, "actual_winner"]).mean())


def contrarian_edge_model_minus_vegas(df, source, threshold):
    """Run the contrarian_edge_model_minus_vegas step and return its normalized result."""
    model_acc = contrarian_edge_accuracy(df, source, threshold)
    vegas_acc = contrarian_edge_vegas_accuracy(df, source, threshold)
    if pd.isna(model_acc) or pd.isna(vegas_acc):
        return np.nan
    return float(model_acc - vegas_acc)


def vegas_alignment_metrics(df, model_sources):
    """Run the vegas_alignment_metrics step and return its normalized result."""
    rows = []
    for source in model_sources:
        rows.append(
            {
                "model": source,
                "with_vegas_accuracy": with_vegas_accuracy(df, source),
                "with_vegas_count": with_vegas_count(df, source),
                "against_vegas_accuracy": against_vegas_accuracy(df, source),
                "against_vegas_count": against_vegas_count(df, source),
                "against_vegas_vegas_accuracy": against_vegas_vegas_accuracy(
                    df, source
                ),
                "against_vegas_model_minus_vegas": against_vegas_model_minus_vegas(
                    df, source
                ),
                "with_vegas_edge_3_plus_accuracy": with_vegas_edge_accuracy(
                    df, source, threshold=3.0
                ),
                "with_vegas_edge_3_plus_count": with_vegas_edge_count(
                    df, source, threshold=3.0
                ),
                "against_vegas_edge_3_plus_accuracy": against_vegas_edge_accuracy(
                    df, source, threshold=3.0
                ),
                "against_vegas_edge_3_plus_count": against_vegas_edge_count(
                    df, source, threshold=3.0
                ),
                "against_vegas_edge_3_plus_vegas_accuracy": against_vegas_edge_vegas_accuracy(
                    df, source, threshold=3.0
                ),
                "against_vegas_edge_3_plus_model_minus_vegas": against_vegas_edge_model_minus_vegas(
                    df, source, threshold=3.0
                ),
            }
        )
    return pd.DataFrame(rows)


def grouped_vegas_alignment_table(df, model_sources, group_col):
    """Run the grouped_vegas_alignment_table step and return its normalized result."""
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for group_value, group in df.groupby(group_col, observed=False, dropna=False):
        for source in model_sources:
            rows.append(
                {
                    group_col: group_value,
                    "model": source,
                    "with_vegas_accuracy": with_vegas_accuracy(group, source),
                    "with_vegas_count": with_vegas_count(group, source),
                    "against_vegas_accuracy": against_vegas_accuracy(group, source),
                    "against_vegas_count": against_vegas_count(group, source),
                    "against_vegas_edge_3_plus_accuracy": against_vegas_edge_accuracy(
                        group, source, threshold=3.0
                    ),
                    "against_vegas_edge_3_plus_count": against_vegas_edge_count(
                        group, source, threshold=3.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def vegas_alignment_mask(df, source, *, agree, threshold=None):
    """Run the vegas_alignment_mask step and return its normalized result."""
    d = valid_source_frame(df, source, require_vegas=True)
    if d.empty or source == "Vegas":
        return d, pd.Series(False, index=d.index)
    same = source_series(d, source, "winner") == d["vegas_winner"]
    same = same & d["vegas_winner"].notna()
    mask = same if agree else ~same & d["vegas_winner"].notna()
    if threshold is not None:
        edge = source_series(d, source, "edge_vs_vegas").abs()
        mask = mask & (edge >= float(threshold))
    return d, mask


def with_vegas_accuracy(df, source):
    """Run the with_vegas_accuracy step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=True)
    if not mask.any():
        return np.nan
    return float(source_series(d.loc[mask], source, "correct").mean())


def with_vegas_count(df, source):
    """Run the with_vegas_count step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=True)
    if d.empty and source == "Vegas":
        return np.nan
    return int(mask.sum())


def against_vegas_accuracy(df, source):
    """Run the against_vegas_accuracy step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=False)
    if not mask.any():
        return np.nan
    return float(source_series(d.loc[mask], source, "correct").mean())


def against_vegas_count(df, source):
    """Run the against_vegas_count step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=False)
    if d.empty and source == "Vegas":
        return np.nan
    return int(mask.sum())


def against_vegas_vegas_accuracy(df, source):
    """Run the against_vegas_vegas_accuracy step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=False)
    if not mask.any():
        return np.nan
    return float((d.loc[mask, "vegas_winner"] == d.loc[mask, "actual_winner"]).mean())


def against_vegas_model_minus_vegas(df, source):
    """Run the against_vegas_model_minus_vegas step and return its normalized result."""
    model_acc = against_vegas_accuracy(df, source)
    vegas_acc = against_vegas_vegas_accuracy(df, source)
    if pd.isna(model_acc) or pd.isna(vegas_acc):
        return np.nan
    return float(model_acc - vegas_acc)


def with_vegas_edge_accuracy(df, source, threshold):
    """Run the with_vegas_edge_accuracy step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=True, threshold=threshold)
    if not mask.any():
        return np.nan
    return float(source_series(d.loc[mask], source, "correct").mean())


def with_vegas_edge_count(df, source, threshold):
    """Run the with_vegas_edge_count step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=True, threshold=threshold)
    if d.empty and source == "Vegas":
        return np.nan
    return int(mask.sum())


def against_vegas_edge_accuracy(df, source, threshold):
    """Run the against_vegas_edge_accuracy step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=False, threshold=threshold)
    if not mask.any():
        return np.nan
    return float(source_series(d.loc[mask], source, "correct").mean())


def against_vegas_edge_count(df, source, threshold):
    """Run the against_vegas_edge_count step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=False, threshold=threshold)
    if d.empty and source == "Vegas":
        return np.nan
    return int(mask.sum())


def against_vegas_edge_vegas_accuracy(df, source, threshold):
    """Run the against_vegas_edge_vegas_accuracy step and return its normalized result."""
    d, mask = vegas_alignment_mask(df, source, agree=False, threshold=threshold)
    if not mask.any():
        return np.nan
    return float((d.loc[mask, "vegas_winner"] == d.loc[mask, "actual_winner"]).mean())


def against_vegas_edge_model_minus_vegas(df, source, threshold):
    """Run the against_vegas_edge_model_minus_vegas step and return its normalized result."""
    model_acc = against_vegas_edge_accuracy(df, source, threshold)
    vegas_acc = against_vegas_edge_vegas_accuracy(df, source, threshold)
    if pd.isna(model_acc) or pd.isna(vegas_acc):
        return np.nan
    return float(model_acc - vegas_acc)


def record_upset_recall(df, source):
    """Run the record_upset_recall step and return its normalized result."""
    d = valid_source_frame(df, source)
    if d.empty or "worse_record_side" not in d.columns:
        return np.nan
    actual_record_upset = d["actual_winner"] == d["worse_record_side"]
    actual_record_upset = actual_record_upset & d["worse_record_side"].isin(
        ["home", "away"]
    )
    if not actual_record_upset.any():
        return np.nan
    return float(
        (
            source_series(d.loc[actual_record_upset], source, "winner")
            == d.loc[actual_record_upset, "worse_record_side"]
        ).mean()
    )


def record_underdog_accuracy(df, source):
    """Run the record_underdog_accuracy step and return its normalized result."""
    d = valid_source_frame(df, source)
    if d.empty or "worse_record_side" not in d.columns:
        return np.nan
    picked_worse = source_series(d, source, "winner") == d["worse_record_side"]
    picked_worse = picked_worse & d["worse_record_side"].isin(["home", "away"])
    if not picked_worse.any():
        return np.nan
    return float(
        (
            d.loc[picked_worse, "actual_winner"]
            == d.loc[picked_worse, "worse_record_side"]
        ).mean()
    )


def rank_upset_recall(df, source):
    """Run the rank_upset_recall step and return its normalized result."""
    d = valid_source_frame(df, source)
    home_rank_col = first_present_column(
        d, ["home_rank", "home_ap_rank", "ap_rank_home", "home_tdnet_rank"]
    )
    away_rank_col = first_present_column(
        d, ["away_rank", "away_ap_rank", "ap_rank_away", "away_tdnet_rank"]
    )
    if d.empty or home_rank_col is None or away_rank_col is None:
        return np.nan
    home_rank = pd.to_numeric(d[home_rank_col], errors="coerce").fillna(26)
    away_rank = pd.to_numeric(d[away_rank_col], errors="coerce").fillna(26)
    lower_ranked_side = pd.Series(pd.NA, index=d.index, dtype="object")
    lower_ranked_side.loc[home_rank > away_rank] = "home"
    lower_ranked_side.loc[away_rank > home_rank] = "away"
    actual_rank_upset = (
        d["actual_winner"] == lower_ranked_side
    ) & lower_ranked_side.isin(["home", "away"])
    if not actual_rank_upset.any():
        return np.nan
    return float(
        (
            source_series(d.loc[actual_rank_upset], source, "winner")
            == lower_ranked_side.loc[actual_rank_upset]
        ).mean()
    )


def brier_score(df, source):
    """Run the brier_score step and return its normalized result."""
    prob_col = f"{source}__pred_win_prob"
    if prob_col not in df.columns:
        return np.nan
    actual = (df["actual_margin"] > 0).astype(float)
    prob = pd.to_numeric(df[prob_col], errors="coerce")
    mask = prob.notna() & actual.notna()
    return float(np.nanmean((prob[mask] - actual[mask]) ** 2)) if mask.any() else np.nan


def log_loss(df, source):
    """Run the log_loss step and return its normalized result."""
    prob_col = f"{source}__pred_win_prob"
    if prob_col not in df.columns:
        return np.nan
    actual = (df["actual_margin"] > 0).astype(float)
    prob = np.clip(pd.to_numeric(df[prob_col], errors="coerce"), 1e-8, 1.0 - 1e-8)
    mask = pd.Series(prob).notna() & actual.notna()
    if not mask.any():
        return np.nan
    y = actual[mask].to_numpy(dtype=float)
    p = np.asarray(prob[mask], dtype=float)
    return float(np.nanmean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def ats_accuracy(df, source):
    """Run the ats_accuracy step and return its normalized result."""
    d = valid_source_frame(df, source, require_vegas=True)
    if d.empty or source == "Vegas":
        return np.nan
    edge = source_series(d, source, "edge_vs_vegas")
    pick = np.where(edge > 0, "home", "away")
    ats_margin = d["actual_margin"] - d["vegas_implied_margin"]
    result = np.where(
        np.isclose(ats_margin, 0.0), "push", np.where(ats_margin > 0, "home", "away")
    )
    mask = result != "push"
    return float((pick[mask] == result[mask]).mean()) if mask.any() else np.nan


def mean_edge(df, source):
    """Run the mean_edge step and return its normalized result."""
    edge = source_series(df, source, "edge_vs_vegas").dropna()
    return float(edge.mean()) if len(edge) else np.nan


def ats_push_rate(df, source):
    """Run the ats_push_rate step and return its normalized result."""
    d = valid_source_frame(df, source, require_vegas=True)
    if d.empty or source == "Vegas":
        return np.nan
    ats_margin = d["actual_margin"] - d["vegas_implied_margin"]
    return float(np.isclose(ats_margin, 0.0).mean())


def missed_upsets_count(df, source):
    """Run the missed_upsets_count step and return its normalized result."""
    if source == "Vegas":
        return np.nan
    d = valid_source_frame(df, source, require_vegas=True)
    actual = d["actual_is_upset"] == True
    called = source_series(d, source, "called_upset") == True
    return int((actual & ~called).sum())


def false_upsets_count(df, source):
    """Run the false_upsets_count step and return its normalized result."""
    if source == "Vegas":
        return np.nan
    d = valid_source_frame(df, source, require_vegas=True)
    actual = d["actual_is_upset"] == True
    called = source_series(d, source, "called_upset") == True
    return int((~actual & called).sum())


def boolean_mask(series):
    """Run the boolean_mask step and return its normalized result."""
    return (
        pd.Series(series, index=getattr(series, "index", None))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )


def valid_source_frame(df, source, require_vegas=False):
    """Run the valid_source_frame step and return its normalized result."""
    pred = source_series(df, source, "pred_margin")
    mask = pred.notna() & pd.to_numeric(df["actual_margin"], errors="coerce").notna()
    if require_vegas:
        if "vegas_implied_margin" not in df.columns:
            return df.iloc[0:0].copy()
        mask = mask & pd.to_numeric(df["vegas_implied_margin"], errors="coerce").notna()
    return df.loc[mask].copy()


def prediction_sources(df):
    """Run the prediction_sources step and return its normalized result."""
    sources = []
    for col in df.columns:
        if col.endswith("__pred_margin"):
            sources.append(col.split("__", 1)[0])
    return [s for s in sources if s == "Vegas"] + [s for s in sources if s != "Vegas"]


def source_series(df, source, field):
    """Run the source_series step and return its normalized result."""
    col = f"{source}__{field}"
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    series = df[col]
    try:
        return pd.to_numeric(series)
    except (TypeError, ValueError):
        return series


def prediction_probability(model, pred_df, margin, fallback_temperature):
    """Run the prediction_probability step and return its normalized result."""
    if "pred_proba_home_win" in pred_df.columns:
        return pd.to_numeric(
            pred_df["pred_proba_home_win"], errors="coerce"
        ).reset_index(drop=True)
    if hasattr(model, "margin_to_probability"):
        return pd.Series(model.margin_to_probability(margin), index=margin.index)
    temperature = getattr(
        model,
        "margin_temperature",
        getattr(model, "margin_scale", fallback_temperature),
    )
    logits = np.clip(margin.astype(float) / max(float(temperature), 1e-8), -60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-logits)), index=margin.index)


def winner_from_margin(margin):
    """Run the winner_from_margin step and return its normalized result."""
    margin = pd.to_numeric(pd.Series(margin), errors="coerce")
    return pd.Series(np.where(margin > 0, "home", "away"), index=margin.index).where(
        margin.notna()
    )


def bucket_abs_margin(values):
    """Run the bucket_abs_margin step and return its normalized result."""
    return pd.cut(
        pd.to_numeric(values, errors="coerce").abs(),
        bins=FAVORITE_BUCKET_BINS,
        labels=FAVORITE_BUCKET_LABELS,
    )


def attach_prior_record_context(base_df, fingerprint_frame):
    """Run the attach_prior_record_context step and return its normalized result."""
    out = base_df.copy()
    required = {"keys_season", "keys_team", "keys_week", "y_margin_this_week"}
    if not required.issubset(fingerprint_frame.columns):
        return out

    records = fingerprint_frame.loc[:, list(required)].copy()
    records["keys_week"] = pd.to_numeric(records["keys_week"], errors="coerce")
    records["y_margin_this_week"] = pd.to_numeric(
        records["y_margin_this_week"], errors="coerce"
    )
    records = records.sort_values(["keys_season", "keys_team", "keys_week"])
    records["win_this_week"] = (
        (records["y_margin_this_week"] > 0)
        .where(records["y_margin_this_week"].notna(), 0)
        .astype(int)
    )
    records["loss_this_week"] = (
        (records["y_margin_this_week"] < 0)
        .where(records["y_margin_this_week"].notna(), 0)
        .astype(int)
    )
    grouped = records.groupby(["keys_season", "keys_team"], observed=True, sort=False)
    records["prior_wins"] = grouped["win_this_week"].cumsum() - records["win_this_week"]
    records["prior_losses"] = (
        grouped["loss_this_week"].cumsum() - records["loss_this_week"]
    )
    records["prior_games"] = records["prior_wins"] + records["prior_losses"]
    records["prior_win_pct"] = np.where(
        records["prior_games"] > 0,
        records["prior_wins"] / records["prior_games"],
        np.nan,
    )

    lookup = records.loc[
        :,
        [
            "keys_season",
            "keys_team",
            "keys_week",
            "prior_wins",
            "prior_losses",
            "prior_games",
            "prior_win_pct",
        ],
    ].drop_duplicates(
        ["keys_season", "keys_team", "keys_week"],
        keep="last",
    )
    game_week = pd.to_numeric(out["week"], errors="coerce")

    for side in ["home", "away"]:
        team_col = f"keys_team_{side}"
        if team_col not in out.columns:
            continue
        left = pd.DataFrame(
            {
                "__idx": out.index,
                "keys_season": out["keys_season"],
                "keys_team": out[team_col],
                "keys_week": game_week,
            }
        )
        merged = left.merge(
            lookup, on=["keys_season", "keys_team", "keys_week"], how="left"
        ).set_index("__idx")
        for col in ["prior_wins", "prior_losses", "prior_games", "prior_win_pct"]:
            out[f"{side}_{col}"] = merged[col].reindex(out.index).to_numpy()

    home_pct = numeric_column(out, "home_prior_win_pct")
    away_pct = numeric_column(out, "away_prior_win_pct")
    home_games = numeric_column(out, "home_prior_games")
    away_games = numeric_column(out, "away_prior_games")
    out["worse_record_side"] = pd.Series(pd.NA, index=out.index, dtype="object")
    comparable = (
        home_pct.notna() & away_pct.notna() & (home_games > 0) & (away_games > 0)
    )
    out.loc[comparable & (home_pct < away_pct), "worse_record_side"] = "home"
    out.loc[comparable & (away_pct < home_pct), "worse_record_side"] = "away"
    out.loc[comparable & np.isclose(home_pct, away_pct), "worse_record_side"] = "tie"
    return out


def first_available_column(df, columns):
    """Run the first_available_column step and return its normalized result."""
    for col in columns:
        if col in df.columns:
            return df[col]
    return pd.Series(np.nan, index=df.index)


def first_present_column(df, columns):
    """Run the first_present_column step and return its normalized result."""
    for col in columns:
        if col in df.columns:
            return col
    return None


def numeric_column(df, column):
    """Run the numeric_column step and return its normalized result."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def save_metric_tables(tables, output_dir, eval_config=None):
    """Run the save_metric_tables step and return its normalized result."""
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in filter_tables_for_policy(tables, eval_config).items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    return tables_dir


def save_evaluation_plots(tables, output_dir, eval_config=None):
    """Run the save_evaluation_plots step and return its normalized result."""
    import matplotlib.pyplot as plt

    if eval_config is None:
        resolved_config = load_eval_config()
    elif isinstance(eval_config, (str, Path)):
        resolved_config = load_eval_config(eval_config_path=eval_config)
    else:
        resolved_config = dict(eval_config)
    policy = ArtifactPolicy.from_config(resolved_config)
    plots_dir = Path(output_dir) / "plots"
    if not policy.png_plots:
        return plots_dir

    tables = filter_tables_for_policy(tables, resolved_config)
    plot_cfg = dict(resolved_config.get("plotting", {}))
    color_map = load_plot_color_map(plot_cfg)
    dpi = int(plot_cfg.get("dpi", 150))
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_score_matrix(
        tables.get("model_score_matrix"),
        plots_dir / "model_score_matrix.png",
        plt,
        color_map,
        plot_cfg,
        dpi,
    )
    plot_margin_parity(
        tables.get("game_predictions"),
        plots_dir / "predicted_vs_actual_margin.png",
        plt,
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_individual_margin_parity(
        tables.get("game_predictions"),
        plots_dir / "parity",
        plt,
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_margin_fit_diagnostics(
        tables.get("margin_diagnostics"),
        plots_dir / "margin_fit_diagnostics.png",
        plt,
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_margin_error_quantiles(
        tables.get("margin_diagnostics"),
        plots_dir / "margin_error_quantiles.png",
        plt,
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_metric_comparison(
        tables.get("overall_margin_metrics"),
        ["mae", "rmse"],
        plots_dir / "overall_margin_mae_rmse.png",
        plt,
        title="Margin Error",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_metric_comparison(
        tables.get("overall_winner_metrics"),
        ["winner_accuracy", "chalk_accuracy", "upset_recall"],
        plots_dir / "overall_winner_chalk_upset.png",
        plt,
        title="Winner, Chalk, Upset",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_metric_comparison(
        tables.get("overall_winner_metrics"),
        [
            "disagreement_accuracy",
            "margin_edge_3_plus_winner_accuracy",
            "contrarian_edge_3_plus_accuracy",
            "record_upset_recall",
            "record_underdog_accuracy",
        ],
        plots_dir / "overall_contrarian_winner_metrics.png",
        plt,
        title="Contrarian Winner Metrics",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_vegas_alignment_accuracy(
        tables.get("overall_vegas_alignment_metrics"),
        plots_dir / "vegas_alignment_accuracy_by_model.png",
        plt,
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
        vegas_accuracy=overall_vegas_accuracy(tables.get("overall_winner_metrics")),
    )
    plot_contrarian_edge_advantage(
        tables.get("overall_winner_metrics"),
        plots_dir / "contrarian_edge_3_plus_model_minus_vegas.png",
        plt,
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("weekly_winner_accuracy"),
        "week",
        plots_dir / "weekly_winner_accuracy.png",
        plt,
        ylabel="Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("weekly_disagreement_accuracy"),
        "week",
        plots_dir / "weekly_disagreement_accuracy.png",
        plt,
        ylabel="Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("weekly_record_upset_recall"),
        "week",
        plots_dir / "weekly_record_upset_recall.png",
        plt,
        ylabel="Recall",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("weekly_mae"),
        "week",
        plots_dir / "weekly_mae.png",
        plt,
        ylabel="MAE",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("weekly_rmse"),
        "week",
        plots_dir / "weekly_rmse.png",
        plt,
        ylabel="RMSE",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("weekly_edge_3_plus_accuracy"),
        "week",
        plots_dir / "weekly_edge_3_plus_accuracy.png",
        plt,
        ylabel="Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("weekly_against_vegas_edge_3_plus_accuracy"),
        "week",
        plots_dir / "weekly_against_vegas_edge_3_plus_accuracy.png",
        plt,
        ylabel="Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("vegas_spread_bucket_winner_accuracy"),
        "vegas_favorite_bucket",
        plots_dir / "vegas_bucket_winner_accuracy.png",
        plt,
        ylabel="Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("vegas_spread_bucket_mae"),
        "vegas_favorite_bucket",
        plots_dir / "vegas_bucket_mae.png",
        plt,
        ylabel="MAE",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("vegas_spread_bucket_rmse"),
        "vegas_favorite_bucket",
        plots_dir / "vegas_bucket_rmse.png",
        plt,
        ylabel="RMSE",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("vegas_spread_bucket_upset_recall"),
        "vegas_favorite_bucket",
        plots_dir / "vegas_bucket_upset_recall.png",
        plt,
        ylabel="Upset Recall",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("vegas_spread_bucket_disagreement_accuracy"),
        "vegas_favorite_bucket",
        plots_dir / "vegas_bucket_disagreement_accuracy.png",
        plt,
        ylabel="Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("vegas_spread_bucket_edge_3_plus_accuracy"),
        "vegas_favorite_bucket",
        plots_dir / "vegas_bucket_edge_3_plus_accuracy.png",
        plt,
        ylabel="Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("model_favorite_bucket_winner_accuracy"),
        "favorite_bucket",
        plots_dir / "model_favorite_bucket_winner_accuracy.png",
        plt,
        ylabel="Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("model_favorite_bucket_mae"),
        "favorite_bucket",
        plots_dir / "model_favorite_bucket_mae.png",
        plt,
        ylabel="MAE",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("confidence_bucket_accuracy"),
        "confidence_bucket",
        plots_dir / "confidence_bucket_accuracy.png",
        plt,
        ylabel="Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("confidence_bucket_calibration"),
        "confidence_bucket",
        plots_dir / "confidence_bucket_calibration.png",
        plt,
        ylabel="Mean Predicted Probability Minus Actual",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_grouped_metric(
        tables.get("ats_by_edge_bucket"),
        "edge_bucket",
        plots_dir / "ats_by_edge_bucket.png",
        plt,
        ylabel="ATS Accuracy",
        color_map=color_map,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    plot_all_table_heatmaps(
        tables,
        plots_dir / "table_heatmaps",
        plt,
        plot_cfg=plot_cfg,
        dpi=dpi,
    )
    return plots_dir


def load_plot_color_map(plot_cfg):
    """Run the load_plot_color_map step and return its normalized result."""
    color_map = {}
    model_colors_path = resolve_repo_path(plot_cfg.get("model_colors_path"))
    if model_colors_path is not None and model_colors_path.exists():
        frame = pd.read_csv(model_colors_path)
        if {"model", "hex"}.issubset(frame.columns):
            color_map.update(
                dict(zip(frame["model"].astype(str), frame["hex"].astype(str)))
            )

    wheel_path = resolve_repo_path(plot_cfg.get("color_wheel_path"))
    if wheel_path is not None and wheel_path.exists():
        wheel = pd.read_csv(wheel_path)
        if {"name", "hex"}.issubset(wheel.columns):
            for name, hex_code in zip(
                wheel["name"].astype(str), wheel["hex"].astype(str)
            ):
                color_map.setdefault(name, hex_code)

    if not color_map:
        color_map.update(
            {
                "Vegas": "#11214F",
                "default_0": "#1EA7FF",
                "default_1": "#5FF2D2",
                "default_2": "#00C853",
                "default_3": "#6A37C8",
                "default_4": "#D4B56E",
                "default_5": "#FF5FA2",
            }
        )
    return color_map


def resolve_repo_path(path):
    """Run the resolve_repo_path step and return its normalized result."""
    if not path:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def color_for_source(source, color_map, idx=0):
    """Run the color_for_source step and return its normalized result."""
    if source in color_map:
        return color_map[source]
    lower_map = {str(key).lower(): value for key, value in color_map.items()}
    if str(source).lower() in lower_map:
        return lower_map[str(source).lower()]
    family = model_family_for_source(source)
    family_wheel = FAMILY_COLOR_WHEELS.get(family)
    if family_wheel:
        return family_wheel[stable_source_index(source, idx) % len(family_wheel)]
    wheel = [
        value for key, value in color_map.items() if str(key).startswith("default_")
    ]
    wheel += [
        color_map[key]
        for key in [
            "Ion Blue",
            "Neon Mint",
            "Electric Emerald",
            "Gridiron Violet",
            "Brass",
            "Edge Pink",
        ]
        if key in color_map
    ]
    if not wheel:
        wheel = ["#1EA7FF", "#5FF2D2", "#00C853", "#6A37C8", "#D4B56E", "#FF5FA2"]
    return wheel[idx % len(wheel)]


def model_family_for_source(source):
    """Infer a broad model family from a plotted source label."""
    text = str(source).strip().lower()
    if text == "vegas":
        return "vegas"
    if (
        text.startswith("stat")
        or "z_index" in text
        or "percentile" in text
        or "robust" in text
        or "weighted" in text
    ):
        return "stat"
    if any(
        token in text
        for token in [
            "forest",
            "tree",
            "boost",
            "xgb",
            "lgbm",
            "catboost",
            "extra_trees",
            "gradient_boosted",
        ]
    ):
        return "tree"
    if any(
        token in text
        for token in [
            "linear",
            "ols",
            "ridge",
            "lasso",
            "elastic",
            "huber",
            "bayesian",
            "ard",
            "ransac",
            "orthogonal",
            "sgd",
            "passive",
        ]
    ):
        return "linear"
    return "model"


def source_style(source, color_map, idx=0):
    """Return family-aware color, marker, and line style for a model source."""
    family = model_family_for_source(source)
    style = dict(FAMILY_STYLE.get(family, {"marker": "o", "linestyle": "-"}))
    style["family"] = family
    style["color"] = color_for_source(source, color_map, idx)
    return style


def stable_source_index(source, fallback=0):
    """Return a deterministic palette index for a source label."""
    text = str(source)
    if not text:
        return int(fallback)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def plot_score_matrix(table, path, plt, color_map, plot_cfg, dpi):
    """Run the plot_score_matrix step and return its normalized result."""
    if (
        table is None
        or table.empty
        or "source" not in table.columns
        or "total_score" not in table.columns
    ):
        return
    plot_df = table.sort_values("total_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(plot_df))))
    bars = []
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        bars.extend(
            ax.barh(
                row["source"],
                row["total_score"],
                color=color_for_source(row["source"], color_map, idx),
                label=row["source"],
            )
        )
    ax.set_xlabel("Composite score")
    ax.set_title("Model Score Matrix")
    ax.set_xlim(0, max(1.0, float(plot_df["total_score"].max()) * 1.1))
    if bool(plot_cfg.get("show_bar_values", True)):
        annotate_bar_values(ax, bars, horizontal=True)
    move_legend_outside(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_margin_parity(df, path, plt, color_map, plot_cfg, dpi):
    """Run the plot_margin_parity step and return its normalized result."""
    if df is None or df.empty or "actual_margin" not in df.columns:
        return
    sources = prediction_sources(df)
    if not sources:
        return

    n_sources = len(sources)
    ncols = min(3, n_sources)
    nrows = int(np.ceil(n_sources / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 4.4 * nrows), squeeze=False
    )
    limits = parity_limits(df, sources)

    for idx, source in enumerate(sources):
        ax = axes[idx // ncols][idx % ncols]
        draw_margin_parity_axis(
            ax,
            df,
            source,
            limits,
            color_for_source(source, color_map, idx),
            color_map,
            plot_cfg,
        )

    for idx in range(n_sources, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Predicted vs Actual Margin", y=1.01)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_individual_margin_parity(df, output_dir, plt, color_map, plot_cfg, dpi):
    """Run the plot_individual_margin_parity step and return its normalized result."""
    if df is None or df.empty or "actual_margin" not in df.columns:
        return
    sources = prediction_sources(df)
    if not sources:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = parity_limits(df, sources)

    for idx, source in enumerate(sources):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        draw_margin_parity_axis(
            ax,
            df,
            source,
            limits,
            color_for_source(source, color_map, idx),
            color_map,
            plot_cfg,
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"predicted_vs_actual_margin_{safe_filename(source)}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def draw_margin_parity_axis(ax, df, source, limits, color, color_map, plot_cfg):
    """Run the draw_margin_parity_axis step and return its normalized result."""
    actual = pd.to_numeric(df["actual_margin"], errors="coerce")
    pred = source_series(df, source, "pred_margin")
    correct = source_series(df, source, "correct")
    mask = actual.notna() & pred.notna()
    if not mask.any():
        ax.set_title(source)
        ax.axis("off")
        return

    x = actual[mask].to_numpy(dtype=float)
    y = pred[mask].to_numpy(dtype=float)
    correct_mask = (
        boolean_mask(correct[mask])
        if len(correct)
        else pd.Series(False, index=actual[mask].index)
    )
    style = source_style(source, color_map)

    ax.scatter(
        x[correct_mask.to_numpy(dtype=bool)],
        y[correct_mask.to_numpy(dtype=bool)],
        s=float(plot_cfg.get("parity_marker_size", 18)),
        alpha=float(plot_cfg.get("parity_alpha", 0.55)),
        color=style["color"] if color is None else color,
        marker=style["marker"],
        edgecolors="none",
        label="Correct winner",
    )
    missed = ~correct_mask.to_numpy(dtype=bool)
    ax.scatter(
        x[missed],
        y[missed],
        s=float(plot_cfg.get("parity_marker_size", 18)),
        alpha=float(plot_cfg.get("parity_alpha", 0.55)),
        color=color_for_source("Edge Pink", color_map, 5),
        marker="x",
        label="Wrong winner",
    )
    ax.plot(
        limits, limits, color="#222222", linestyle="--", linewidth=1.1, label="Perfect"
    )
    ax.axhline(0.0, color="#AAB2BD", linewidth=0.8)
    ax.axvline(0.0, color="#AAB2BD", linewidth=0.8)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(str(source))
    ax.set_xlabel("Actual margin")
    ax.set_ylabel("Predicted margin")
    ax.grid(True, alpha=0.22)
    ax.text(
        0.03,
        0.97,
        parity_stats_text(df, source),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#DDE2E7",
            "alpha": 0.88,
        },
    )


def parity_stats_text(df, source):
    """Run the parity_stats_text step and return its normalized result."""
    stats = {
        "n": valid_source_frame(df, source).shape[0],
        "MAE": margin_mae(df, source),
        "RMSE": margin_rmse(df, source),
        "Bias": margin_bias(df, source),
        "R2": margin_r2(df, source),
    }
    lines = [f"n={int(stats['n'])}"]
    for label in ["MAE", "RMSE", "Bias", "R2"]:
        value = stats[label]
        if pd.notna(value):
            lines.append(f"{label}={value:.2f}")
    return "\n".join(lines)


def parity_limits(df, sources):
    """Run the parity_limits step and return its normalized result."""
    values = [pd.to_numeric(df["actual_margin"], errors="coerce")]
    for source in sources:
        values.append(source_series(df, source, "pred_margin"))
    series = pd.concat(values, ignore_index=True)
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return [-35.0, 35.0]
    low = float(series.min())
    high = float(series.max())
    pad = max(3.0, 0.05 * (high - low))
    low = np.floor((low - pad) / 7.0) * 7.0
    high = np.ceil((high + pad) / 7.0) * 7.0
    if np.isclose(low, high):
        low -= 7.0
        high += 7.0
    return [low, high]


def plot_margin_fit_diagnostics(table, path, plt, color_map, plot_cfg, dpi):
    """Run the plot_margin_fit_diagnostics step and return its normalized result."""
    if table is None or table.empty or "source" not in table.columns:
        return
    metrics = [
        ("correlation", "Corr"),
        ("r2", "R2"),
        ("within_7", "Within 7"),
        ("within_14", "Within 14"),
    ]
    plot_df = table.copy()
    if not any(col in plot_df.columns for col, _ in metrics):
        return
    x = np.arange(len(plot_df))
    width = min(0.8 / len(metrics), 0.18)
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(plot_df)), 4.8))
    for idx, (col, label) in enumerate(metrics):
        if col not in plot_df.columns:
            continue
        offset = (idx - (len(metrics) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            pd.to_numeric(plot_df[col], errors="coerce").to_numpy(dtype=float),
            width=width,
            color=color_for_source(label, color_map, idx),
            label=label,
        )
        if bool(plot_cfg.get("show_bar_values", True)):
            annotate_bar_values(ax, bars, horizontal=False)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_title("Margin Fit Diagnostics")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["source"].astype(str), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    move_legend_outside(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_margin_error_quantiles(table, path, plt, color_map, plot_cfg, dpi):
    """Run the plot_margin_error_quantiles step and return its normalized result."""
    if table is None or table.empty or "source" not in table.columns:
        return
    metrics = [
        ("median_absolute_error", "P50"),
        ("abs_error_p75", "P75"),
        ("abs_error_p90", "P90"),
    ]
    plot_df = table.copy()
    if not any(col in plot_df.columns for col, _ in metrics):
        return
    x = np.arange(len(plot_df))
    width = min(0.8 / len(metrics), 0.22)
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(plot_df)), 4.8))
    for idx, (col, label) in enumerate(metrics):
        if col not in plot_df.columns:
            continue
        offset = (idx - (len(metrics) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            pd.to_numeric(plot_df[col], errors="coerce").to_numpy(dtype=float),
            width=width,
            color=color_for_source(label, color_map, idx),
            label=label,
        )
        if bool(plot_cfg.get("show_bar_values", True)):
            annotate_bar_values(ax, bars, horizontal=False)
    ax.set_title("Absolute Margin Error Quantiles")
    ax.set_ylabel("Points")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["source"].astype(str), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    move_legend_outside(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(table, metrics, path, plt, title, color_map, plot_cfg, dpi):
    """Run the plot_metric_comparison step and return its normalized result."""
    if table is None or table.empty or "metric" not in table.columns:
        return
    plot_df = table.loc[table["metric"].isin(metrics)].set_index("metric")
    if plot_df.empty:
        return
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
    source_cols = list(plot_df.columns)
    x = np.arange(len(plot_df.index))
    width = min(0.8 / max(len(source_cols), 1), 0.18)
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, source in enumerate(source_cols):
        offset = (idx - (len(source_cols) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            plot_df[source].to_numpy(dtype=float),
            width=width,
            color=color_for_source(source, color_map, idx),
            label=source,
        )
        if bool(plot_cfg.get("show_bar_values", True)):
            annotate_bar_values(ax, bars, horizontal=False)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index.astype(str), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    move_legend_outside(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_metric(table, group_col, path, plt, ylabel, color_map, plot_cfg, dpi):
    """Run the plot_grouped_metric step and return its normalized result."""
    if table is None or table.empty or group_col not in table.columns:
        return
    plot_df = table.copy()
    source_cols = [col for col in plot_df.columns if col != group_col]
    if not source_cols:
        return
    for col in source_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(plot_df))
    for idx, col in enumerate(source_cols):
        style = source_style(col, color_map, idx)
        ax.plot(
            x,
            plot_df[col],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=col,
            color=style["color"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[group_col].astype(str), rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    move_legend_outside(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_all_table_heatmaps(tables, output_dir, plt, plot_cfg, dpi):
    """Run the plot_all_table_heatmaps step and return its normalized result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if (
            name in {"game_predictions", "shap_artifacts"}
            or not isinstance(table, pd.DataFrame)
            or table.empty
        ):
            continue
        plot_table_heatmap(
            table,
            output_dir / f"{safe_filename(name)}.png",
            plt,
            title=f"{name.replace('_', ' ').title()} Heatmap",
            plot_cfg=plot_cfg,
            dpi=dpi,
        )


def plot_table_heatmap(table, path, plt, title, plot_cfg, dpi):
    """Run the plot_table_heatmap step and return its normalized result."""
    heatmap = table_heatmap_frame(table)
    if heatmap is None or heatmap.empty:
        return

    labels = heatmap.index.astype(str).tolist()
    numeric = heatmap.apply(pd.to_numeric, errors="coerce")
    matrix = column_normalized_matrix(numeric)
    if matrix.size == 0 or not np.isfinite(matrix).any():
        return

    fig_width = max(7.5, 0.75 * len(numeric.columns) + 2.5)
    fig_height = max(4.0, 0.34 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=str(plot_cfg.get("heatmap_cmap", "viridis")),
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(numeric.columns)))
    ax.set_xticklabels(numeric.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Metric / source")
    ax.set_ylabel("")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Column-normalized value")

    annotate_heatmap = bool(plot_cfg.get("annotate_table_heatmaps", True))
    if annotate_heatmap and numeric.shape[0] * numeric.shape[1] <= int(
        plot_cfg.get("heatmap_annotation_limit", 360)
    ):
        for i in range(numeric.shape[0]):
            for j in range(numeric.shape[1]):
                value = numeric.iloc[i, j]
                if pd.isna(value):
                    continue
                color = (
                    "white" if matrix[i, j] < 0.25 or matrix[i, j] > 0.75 else "#111111"
                )
                ax.text(
                    j,
                    i,
                    compact_number(value),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=color,
                )

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def table_heatmap_frame(table):
    """Run the table_heatmap_frame step and return its normalized result."""
    frame = pd.DataFrame(table).copy()
    label_cols = [
        col
        for col in [
            "metric",
            "source",
            "model",
            "week",
            "vegas_favorite_bucket",
            "favorite_bucket",
            "confidence_bucket",
            "edge_bucket",
        ]
        if col in frame.columns
    ]
    numeric_cols = [
        col
        for col in frame.columns
        if col not in label_cols
        and pd.to_numeric(frame[col], errors="coerce").notna().any()
    ]
    if not numeric_cols:
        return None
    if label_cols:
        labels = frame.loc[:, label_cols].astype(str).agg(" | ".join, axis=1)
    else:
        labels = frame.index.astype(str)
    out = frame.loc[:, numeric_cols].copy()
    out.index = labels
    return out


def row_normalized_matrix(frame):
    """Normalize each row of a numeric frame to 0-1 color values."""
    values = frame.to_numpy(dtype=float)
    matrix = np.full(values.shape, np.nan, dtype=float)
    for i in range(values.shape[0]):
        row = values[i]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        low = np.nanmin(row[finite])
        high = np.nanmax(row[finite])
        if np.isclose(low, high):
            matrix[i, finite] = 0.5
        else:
            matrix[i, finite] = (row[finite] - low) / (high - low)
    return matrix


def column_normalized_matrix(frame):
    """Normalize each column of a numeric frame to 0-1 color values."""
    values = frame.to_numpy(dtype=float)
    matrix = np.full(values.shape, np.nan, dtype=float)
    for j in range(values.shape[1]):
        col = values[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        low = np.nanmin(col[finite])
        high = np.nanmax(col[finite])
        if np.isclose(low, high):
            matrix[finite, j] = 0.5
        else:
            matrix[finite, j] = (col[finite] - low) / (high - low)
    return matrix


def compact_number(value):
    """Run the compact_number step and return its normalized result."""
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:.2g}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def safe_filename(value):
    """Run the safe_filename step and return its normalized result."""
    text = str(value).strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return text or "plot"


def plot_vegas_alignment_accuracy(
    table, path, plt, color_map, plot_cfg, dpi, vegas_accuracy=None
):
    """Run the plot_vegas_alignment_accuracy step and return its normalized result."""
    if table is None or table.empty or "model" not in table.columns:
        return
    metrics = [
        ("with_vegas_accuracy", "with_vegas_count", "With Vegas"),
        ("against_vegas_accuracy", "against_vegas_count", "Against Vegas"),
        (
            "against_vegas_edge_3_plus_accuracy",
            "against_vegas_edge_3_plus_count",
            "Against Vegas + 3pt Edge",
        ),
    ]
    plot_df = table.copy()
    x = np.arange(len(plot_df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(plot_df)), 4.8))
    for idx, (value_col, count_col, label) in enumerate(metrics):
        values = pd.to_numeric(plot_df[value_col], errors="coerce").to_numpy(
            dtype=float
        )
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=color_for_source(label, color_map, idx),
        )
        for bar, value, count in zip(bars, values, plot_df[count_col]):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}\nn={int(count) if pd.notna(count) else 0}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    if vegas_accuracy is not None and pd.notna(vegas_accuracy):
        ax.axhline(
            float(vegas_accuracy),
            color=color_for_source("Vegas", color_map),
            linestyle="--",
            linewidth=1.2,
        )
        ax.text(
            len(plot_df) - 0.5,
            float(vegas_accuracy),
            f"Vegas {float(vegas_accuracy):.3f}",
            va="bottom",
            ha="right",
            fontsize=8,
        )
    ax.set_title("Winner Accuracy: With Vegas vs Against Vegas")
    ax.set_ylabel("Winner Accuracy")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"].astype(str), rotation=30, ha="right")
    ax.set_ylim(
        0,
        min(
            1.0,
            max(
                0.65,
                np.nanmax(plot_df[[m[0] for m in metrics]].to_numpy(dtype=float))
                + 0.15,
            ),
        ),
    )
    ax.grid(axis="y", alpha=0.25)
    move_legend_outside(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_contrarian_edge_advantage(table, path, plt, color_map, plot_cfg, dpi):
    """Run the plot_contrarian_edge_advantage step and return its normalized result."""
    if table is None or table.empty or "metric" not in table.columns:
        return
    metric = "contrarian_edge_3_plus_model_minus_vegas"
    rows = table.loc[table["metric"] == metric]
    if rows.empty:
        return
    series = (
        rows.drop(columns=["metric"]).iloc[0].drop(labels=["Vegas"], errors="ignore")
    )
    series = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(series))))
    bars = []
    for idx, (model, value) in enumerate(series.items()):
        bars.extend(
            ax.barh(model, value, color=color_for_source(model, color_map, idx))
        )
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_title("3+ Point Contrarian Edge: Model Minus Vegas")
    ax.set_xlabel("Accuracy advantage")
    ax.grid(axis="x", alpha=0.25)
    if bool(plot_cfg.get("show_bar_values", True)):
        annotate_bar_values(ax, bars, horizontal=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def overall_vegas_accuracy(table):
    """Run the overall_vegas_accuracy step and return its normalized result."""
    if (
        table is None
        or table.empty
        or "metric" not in table.columns
        or "Vegas" not in table.columns
    ):
        return None
    row = table.loc[table["metric"] == "winner_accuracy", "Vegas"]
    if row.empty:
        return None
    value = pd.to_numeric(row, errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def annotate_bar_values(ax, bars, horizontal=False):
    """Run the annotate_bar_values step and return its normalized result."""
    for bar in bars:
        if horizontal:
            value = bar.get_width()
            if not np.isfinite(value):
                continue
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:.3g}",
                va="center",
                ha="left",
                fontsize=8,
            )
        else:
            value = bar.get_height()
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3g}",
                va="bottom",
                ha="center",
                rotation=90,
                fontsize=8,
            )


def move_legend_outside(ax):
    """Run the move_legend_outside step and return its normalized result."""
    handles, labels = ax.get_legend_handles_labels()
    labeled = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not str(label).startswith("_")
    ]
    if not labeled:
        return
    handles, labels = zip(*labeled)
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=False,
        title="Legend",
    )
