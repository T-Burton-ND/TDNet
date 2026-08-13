"""Experiment helpers for TDNet research workflows."""

from .opponent_adjusted import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_VERSION_SPECS,
    OpponentAdjustedVersionSpec,
    StaticFrameFingerprints,
    build_opponent_adjusted_experiment_frames,
    run_opponent_adjusted_sweep,
)
from .opponent_ablation import (
    DEFAULT_ABLATION_EXPERIMENT_NAME,
    DEFAULT_ABLATION_SPECS,
    AblationSpec,
    apply_ablation_view,
    build_ablation_job_manifest,
    merge_ablation_outputs,
    run_manifest_job,
)

__all__ = [
    "AblationSpec",
    "DEFAULT_ABLATION_EXPERIMENT_NAME",
    "DEFAULT_ABLATION_SPECS",
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_VERSION_SPECS",
    "OpponentAdjustedVersionSpec",
    "StaticFrameFingerprints",
    "apply_ablation_view",
    "build_ablation_job_manifest",
    "build_opponent_adjusted_experiment_frames",
    "merge_ablation_outputs",
    "run_manifest_job",
    "run_opponent_adjusted_sweep",
    "build_experiment_manifest",
    "filter_frame_for_feature_config",
    "merge_experiment_chunks",
    "run_experiment_chunk",
    "run_experiment_trial",
    "select_finalists",
    "validate_experiment_output",
]

from .publication import (
    build_experiment_manifest,
    filter_frame_for_feature_config,
    merge_experiment_chunks,
    run_experiment_chunk,
    run_experiment_trial,
    select_finalists,
    validate_experiment_output,
)
