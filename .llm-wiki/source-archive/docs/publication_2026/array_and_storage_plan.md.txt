# Publication compute and storage plan

Prepared: 2026-07-16. No SGE jobs were submitted.

The canonical catalog is
`EXTERNAL_DURABLE_ROOT/publication_artifacts/suite_manifests/array_catalog.csv`.
Every SGE task selects exactly one manifest row and performs at most one model
fit. Workers request one slot and force BLAS/OpenMP/PyTorch internal threads to
one. Default task concurrency is deliberately limited to 5 or 10.

| Array | Tasks | `-tc` | Purpose |
|---|---:|---:|---|
| `publication_matrix` | 6,000 | 10 | Controlled feature-tier × model-family comparison |
| `hps_spline` | 3,200 | 10 | 40 deterministic setpoints × 4 tiers × 2 objectives × 10 rolling folds |
| `hps_hist_gradient_boosted` | 5,120 | 10 | 64 setpoints × 4 tiers × 2 objectives × 10 folds |
| `hps_mlp` | 3,840 | 5 | 48 bounded setpoints × 4 tiers × 2 objectives × 10 folds |
| `hps_structured_mlp` | 640 | 5 | 32 setpoints × F6 × 2 objectives × 10 folds |
| `legacy_balanced_recovery` | 22 | 5 | Missing retained balanced-study rows only |
| `legacy_margin_recovery` | 270 | 5 | Corrected OMP/RANSAC failed rows only |
| `legacy_ablation_recovery` | 5 | 5 | Failed numerical ablation rows only |
| `final_model_diagnostics` | 51 | 5 | Outstanding diagnostics for retained finalists |
| `preseason_saved_states` | 7 | 5 | One saved 2026 Week-0 state per v1.1–v1.7 fingerprint |
| **Total** | **19,155** |  | **10 arrays** |

The new-family searches are bounded deterministic random searches, as required
by the publication plan; they are not impractical full Cartesian products.
Naive baselines have no useful hyperparameter search. Ensemble membership and
weights must wait for historical out-of-fold finalist predictions and must not
use 2026 outcomes.

The currently enumerable compact-output estimate is about 5.35 GiB. An 8 GiB
reserve for later finalist predictions, checkpoints, calibration objects, and
diagnostics puts the projected program at roughly 13.35 GiB, above the declared
10 GiB repository-local threshold. All suite outputs are therefore directed to
`CLUSTER_STORAGE_ROOT`. Repository
storage keeps only code, configs, compact public tables, figures, and manifests.

All four new model families completed a one-row local smoke fit in temporary
output roots. Their group manifests remain locked:
`submit_publication_suite_array.sh --submit` refuses to run until the
corresponding manifest directory contains an explicitly created
`smoke_test.ok`. The controlled matrix alone retains its earlier successful
group smoke marker. Dry-run inspection example:

```bash
scripts/sge/submit_publication_suite_array.sh --array-id hps_mlp
```

No wrapper calls `qsub` without `--submit`.
