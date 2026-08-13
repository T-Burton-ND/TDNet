# TDNet Contracts And Artifacts

TDNet keeps shared column, feature, artifact, metadata, and row-semantics rules in `src/gridiron_ml/contracts/`.

## Row Semantics

v0 fingerprint rows are `state_after_week`: `keys_week=N` means the row includes games completed through week `N`. Week `0` is the preseason/bootstrap state.

`y_next_margin` is the default training target because it is shifted from the next scheduled game onto the prior team state. `y_margin_this_week` is same-row completed-game information and is unsafe with postgame/current-week features.

## Feature, Label, And Market Separation

`fingerprints.features.split_frame` separates a canonical fingerprint frame into:

- `x_df`: numeric/bool model features.
- `y`: the configured target, defaulting to `y_next_margin`.
- `meta_df`: row metadata and labels not used as model features.
- `market_df`: `market_*` context for evaluation.

`market_*` columns are evaluation-only by default. Training paths reject them unless `allow_market_features_for_training=True` is explicitly configured.

Known leaky coach fields are centralized in `contracts.features` and enforced through schema validators. Current blocked patterns include `coach_season_*`, `coach_*postseason*`, and `coach_career_mean_postseason_rank_points`.

## Artifact Naming

Artifact names and path helpers live in `contracts.artifacts`.

Current v0 paths are:

- `data/fingerprints/v0/canonical_fingerprint.parquet`
- `data/fingerprints/v0/v0_gridiron_ml_fingerprints.parquet`
- `data/fingerprints/v0/v0_gridiron_ml_labels.parquet`
- `data/fingerprints/v0/<season>/team_week_fingerprints_<season>.parquet`
- `data/fingerprints/v0/<season>/team_week_labels_<season>.parquet`
- `metadata.json` sidecars at the version and season levels

v0 feature artifacts must not contain `market_*`; canonical fingerprints may retain market context so evaluation can build `market_df`.

## Metadata

Metadata sidecar keys and values live in `contracts.metadata`. v0 historical fingerprint metadata records:

- `fingerprint_version`
- `row_semantics: state_after_week`
- `default_target: y_next_margin`
- `unsafe_same_row_target: y_margin_this_week`
- `market_policy: evaluation_only`
- `artifact_kind: historical_fingerprint`
- optional `season`

## Rebuild Cleanup

`contracts.artifacts.cleanup_fingerprint_artifacts` removes generated v0 fingerprint parquet and metadata files before an overwrite rebuild. It intentionally leaves debug files, notebooks, model outputs, and unrelated directories alone.

`BaseFingerprintBuilder.cleanup_artifacts_for_rebuild` exposes that behavior to builders. `V0FingerprintBuilder.build(overwrite=True)` runs the cleanup before rebuilding from team-game tables.

## Schema And Training Safety

Required dataframe column groups live in `contracts.columns` and are consumed by `schemas.validators`.

Training safety is enforced in:

- fingerprint splitting
- fingerprint training blocks
- matchup/eval training frames
- direct model training

All of those paths reject unsafe market and coach columns through shared validators/contracts.

## v1 Guidance

Future v1 opponent-adjusted fingerprints should import these contracts instead of redefining labels, path names, metadata fields, row semantics, or leakage rules. New v1-specific residual columns should add their own constants only when the v1 schema exists; this sprint intentionally does not implement opponent adjustment.
