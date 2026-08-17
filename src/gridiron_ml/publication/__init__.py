"""Publication experiments, freezes, predictions, and reporting for TDNet.

Scientific models come from the unified :mod:`gridiron_ml.models` registry;
this package owns publication-specific orchestration and artifact contracts.
"""

from gridiron_ml.models import (
    TDBoosted,
    TDEnsemble,
    TDLinear,
    TDMLP,
    TDNaive,
    TDSpline,
    TDStructuredMLP,
    TDTree,
    build_model_from_config,
)
from .bundles import (
    build_prediction_bundle,
    prepare_public_prediction_table,
    score_prediction_bundle,
    verify_prediction_bundle,
)
from .weekly import build_weekly_blog_package
from .freeze import (
    build_model_cards,
    build_preseason_freeze,
    render_freeze_readme,
    verify_preseason_freeze,
    write_sha256sums,
)
from .preseason_states import build_preseason_state_frame, materialize_preseason_state
from .polls import load_ap_top25
from .manual_poll import (
    ballot_store_path,
    find_latest_model_poll,
    load_saved_ballot,
    make_drag_poll_editor,
    run_manual_poll,
    save_manual_ballot,
    validate_ballot,
)
from .recaps import (
    build_individual_model_recaps,
    build_retrospective_consensus,
    build_season_sunday_recaps,
    model_chalk_upset_matrix,
    model_vegas_correctness_matrix,
    plot_objective_weekly_comparison,
    plot_sunday_recap_table,
    write_model_vegas_confusion_artifacts,
)
from .poll_recaps import build_season_poll_recaps, model_consensus_disagreement
from .selection import select_confirmatory_roster, write_confirmatory_roster
from .roster_poll import build_frozen_roster_poll
from .preseason_rankings import build_preseason_performance_rankings, load_preseason_performance_rankings
from .validation_figures import normalize_cv_metrics, plot_cv_metric_boxplots
from .figures import PublicationFigureBuilder
from .tables import build_publication_tables
from .weekly_protocol import (
    build_snapshot_completeness,
    inspect_endpoint,
    validate_deadline_utc,
)
from .baselines import fit_baseline, fit_predict_baseline, predict_baseline
from .consensus import build_equal_weight_consensus, leave_one_out_audit, select_compact_components
from .imputation import TemporalDonorImputer
from .locked_bundle import validate_canonical_2026_inventory

__all__ = [
    "TDBoosted",
    "TDEnsemble",
    "TDLinear",
    "TDMLP",
    "TDNaive",
    "TDSpline",
    "TDStructuredMLP",
    "TDTree",
    "build_model_from_config",
    "build_prediction_bundle",
    "prepare_public_prediction_table",
    "build_weekly_blog_package",
    "build_model_cards",
    "build_preseason_freeze",
    "render_freeze_readme",
    "verify_preseason_freeze",
    "write_sha256sums",
    "build_preseason_state_frame",
    "materialize_preseason_state",
    "load_ap_top25",
    "ballot_store_path",
    "find_latest_model_poll",
    "load_saved_ballot",
    "make_drag_poll_editor",
    "run_manual_poll",
    "save_manual_ballot",
    "validate_ballot",
    "build_season_sunday_recaps",
    "build_retrospective_consensus",
    "plot_sunday_recap_table",
    "plot_objective_weekly_comparison",
    "build_individual_model_recaps",
    "model_chalk_upset_matrix",
    "model_vegas_correctness_matrix",
    "write_model_vegas_confusion_artifacts",
    "build_season_poll_recaps",
    "model_consensus_disagreement",
    "select_confirmatory_roster",
    "write_confirmatory_roster",
    "build_frozen_roster_poll",
    "build_preseason_performance_rankings",
    "load_preseason_performance_rankings",
    "normalize_cv_metrics",
    "plot_cv_metric_boxplots",
    "build_publication_tables",
    "PublicationFigureBuilder",
    "score_prediction_bundle",
    "verify_prediction_bundle",
    "build_snapshot_completeness",
    "inspect_endpoint",
    "validate_deadline_utc",
    "fit_baseline",
    "fit_predict_baseline",
    "predict_baseline",
    "build_equal_weight_consensus",
    "leave_one_out_audit",
    "select_compact_components",
    "TemporalDonorImputer",
    "validate_canonical_2026_inventory",
]
