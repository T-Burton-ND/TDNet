"""TDNet run orchestration, evaluation, reporting, and matchup helpers."""

from .market import (
    DEFAULT_VEGAS_CONVENTION,
    VegasConvention,
    market_home_margin,
    normalize_vegas_frame,
)
from .matchups import DEFAULT_ZERO_OFFSET, MatchupBuilder
from .generate_figures import regenerate_comparison_figures, regenerate_poll_figures
from .shap_analysis import save_shap_analysis_for_models
from .data_points import (
    DataPoint,
    data_point_options,
    load_data_point_catalog,
    plot_data_point_logo_scatter,
    print_data_point_options,
    resolve_data_point,
)
from .evaluator import TDEval
from .td_run import TDRun, result_table
from .training import (
    DEFAULT_MODEL_SPECS,
    ModelRunSpec,
    TrainingResult,
    TrainingRun,
    build_eval_config,
    checkpoint_path,
    clear_model_run_dir,
    filter_model_specs,
    model_run_dir,
    train_model_specs,
)
from .weekly_report import WeeklyReportBuilder, discover_latest_checkpoints

__all__ = [
    "DEFAULT_VEGAS_CONVENTION",
    "DEFAULT_MODEL_SPECS",
    "DEFAULT_ZERO_OFFSET",
    "DataPoint",
    "MatchupBuilder",
    "ModelRunSpec",
    "TDRun",
    "TDEval",
    "TrainingResult",
    "TrainingRun",
    "VegasConvention",
    "WeeklyReportBuilder",
    "build_eval_config",
    "checkpoint_path",
    "clear_model_run_dir",
    "data_point_options",
    "discover_latest_checkpoints",
    "filter_model_specs",
    "market_home_margin",
    "model_run_dir",
    "normalize_vegas_frame",
    "load_data_point_catalog",
    "plot_data_point_logo_scatter",
    "print_data_point_options",
    "regenerate_comparison_figures",
    "regenerate_poll_figures",
    "resolve_data_point",
    "result_table",
    "save_shap_analysis_for_models",
    "train_model_specs",
]
