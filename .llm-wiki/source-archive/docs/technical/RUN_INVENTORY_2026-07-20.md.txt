# TDNet run inventory after `/groups` filled — 2026-07-20

Read-only audit performed after the TDNet SGE chain stopped appearing in `qstat`.
No new jobs were submitted.

## Storage / queue state

- `/groups` reported as effectively full: `20T` size, `20T` used, `119G` available, `100%` use.
- No TDNet jobs appeared in `qstat` during the audit.
- `qacct` did not yet return accounting records for the recent TDNet array IDs (`1208975`-`1208985`), so final SGE exit reasons were not available from accounting at audit time.
- Several per-task `status.json` files still say `running`; because no matching jobs were visible in `qstat`, treat those as stale/incomplete unless a valid completion artifact exists.

## Completion inventory by array

| Array | Expected tasks | Completed | Remaining | Notes |
|---|---:|---:|---:|---|
| `publication_matrix` | 6000 | 2544 | 3456 | Completed rows are one-line `result.parquet` summaries. Includes 1333 winner and 1211 margin completions. |
| `hps_spline` | 3200 | 31 | 3169 | Only early margin spline tasks completed before the chain stopped. |
| `hps_hist_gradient_boosted` | 5120 | 0 | 5120 | Not started/completed. |
| `hps_mlp` | 3840 | 0 | 3840 | Not started/completed. |
| `hps_structured_mlp` | 640 | 0 | 640 | Not started/completed. |
| `hps_kernel` | 7920 | 1 | 7919 | Smoke/early task only. |
| `hps_temporal` | 9120 | 1 | 9119 | Smoke/early task only. |
| `legacy_balanced_recovery` | 22 | 0 | 22 | Intentionally dropped from scope per owner decision; do not rerun unless scope changes. |
| `legacy_margin_recovery` | 270 | 0 | 270 | Still needed for winner/margin-only scope. |
| `legacy_ablation_recovery` | 5 | 5 | 0 | Complete. |
| `final_model_diagnostics` | 51 | 51 | 0 | Complete; outputs are `plots/` and `tables/`, not `result.parquet`. Includes 17 winner, 17 balanced, 17 margin legacy diagnostics. |
| `preseason_saved_states` | 7 | 0 | 7 | Still needed. |

## Completed publication-matrix coverage

Completed `publication_matrix` tasks by objective:

- winner: 1333
- margin: 1211

Completed `publication_matrix` tasks by model family:

- linear: 516
- spline: 516
- tree: 503
- boosted: 510
- neural: 499

## Important stale failures

There are 14 stale failed `status.json` files in `publication_matrix` from the aborted pre-fix run (`1208957`), all with:

```text
meta_df must contain either current-game pairing fields or next-game pairing fields.
```

Those belong to the killed attempt before the F7/F8 market-feature fix. The corrected run had already passed that old failure zone before `/groups` became the limiting issue.

## What remains before model/publication freeze

Remaining model-search / training work, after excluding balanced:

- `publication_matrix`: 3456 tasks
- `hps_spline`: 3169 tasks
- `hps_hist_gradient_boosted`: 5120 tasks
- `hps_mlp`: 3840 tasks
- `hps_structured_mlp`: 640 tasks
- `hps_kernel`: 7919 tasks
- `hps_temporal`: 9119 tasks
- `legacy_margin_recovery`: 270 tasks
- `preseason_saved_states`: 7 tasks

Total remaining, excluding balanced: **33,540 tasks**.

Publication/analysis work still depends on those outputs:

- merge incomplete search fragments;
- rank best model per family/type using 2024/2025 criteria;
- select top-1, top-3, and consensus rosters;
- regenerate 2025 dry-run weekly predictions/recaps with all included models;
- regenerate per-model weekly folders and cumulative accuracy tracks;
- regenerate weekly TDNet top-25 poll, AP comparison, model disagreement, and full-season poll grid;
- regenerate closest-10 weekly matchup table/figure;
- build preseason saved-state artifacts and freeze metadata/hashes.

## Storage note

The current per-training-run output is summary-only:

- `status.json`
- one-row `result.parquet` with metrics/config metadata

Sampled `result.parquet` files had shape `(1, 57)`. No model/checkpoint files were sampled in `publication_artifacts`.
