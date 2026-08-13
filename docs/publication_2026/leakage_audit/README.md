# TDNet adversarial leakage audit

Created: 2026-08-13T15:37:40.394793+00:00

| Test | Status | Expected | Observed |
|---|---|---|---|
| market_boundary | pass | market feature rejected by default | ValueError: Market/Vegas/betting-derived columns are eval-only by default and must not be included in model training features. Remove market_* columns from X, or set allow_market_features_for_training=True only for an explicit market-feature experiment. Offending columns: market_spread_close |
| season_split_overlap | pass | overlap rejected | ValueError: Training/evaluation leakage risk: train and val years overlap: 2023. |
| future_target_knn | pass | earlier OOF predictions invariant; target games absent from neighbors | {'predictions': 2, 'audited_neighbors': 4} |
| future_opponent_adjustment | pass | prior-week adjusted contributions invariant | {'week_checked': 1, 'method': 'opponent_ridge'} |
| opponent_reorder | pass | opponent-adjusted contributions invariant to input order | {'rows': 6, 'shuffled_seed': 7} |
| duplicate_game_detection | pass | duplicate season/week/game keys explicitly flagged | {'duplicate_rows_flagged': 2, 'key': 'season/week/game_id'} |
| missing_team_history | pass | declared finite fallback rather than future data | {'rows': 2, 'week': 1, 'fallback': 'global_history_or_zero'} |
| failed_model_consensus | pass | no imputation; effective membership is recorded per game | {'effective_model_counts': {'g1': 2, 'g2': 1}, 'membership_rows': 3} |
| deadline_contract | pass | Thursday New York deadline maps to exact UTC | {'deadline_local': '2026-07-30T23:59:00-04:00', 'deadline_timezone': 'America/New_York', 'deadline_utc': '2026-07-31T03:59:00Z'} |
| calibration_bounds | pass | fold-separated calibrator is monotone and bounded | {'fit_rows': 6, 'probability_min': 3.670953304960871e-08, 'probability_max': 0.999999963290467} |
| bundle_immutability | pass | prediction bytes and manifest remain verifiable | {'bundle_verified': True, 'prediction_bytes_unchanged': True} |

## Limitations

- This does not replace the full historical F0-F8 replay.
- CFBD line timestamp/provider completeness remains a prospective-data blocker.
- Neutral-site, canceled/postponed, and late-field fixtures still require explicit historical replay coverage.
