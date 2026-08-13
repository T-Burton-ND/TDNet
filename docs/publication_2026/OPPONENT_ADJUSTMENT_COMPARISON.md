# Opponent-adjustment comparison and selection status

The repository contains seven opponent-adjustment families and an existing
126-run comparison. The existing summary below is useful descriptive evidence,
but it evaluates the consumed 2025 holdout (`train=2010–2023`, `validation=2024`,
`test=2025`). It must not be used as the 2026 primary-method selection rule.

| Version | Method | 2025 MAE | 2025 RMSE | Winner accuracy | Upset correct | Composite score |
|---|---|---:|---:|---:|---:|---:|
| v1.1 | opponent context | 14.5678 | 18.4267 | 0.6720 | 0.2753 | 0.5162 |
| v1.2 | opponent ridge | 14.5050 | 18.3944 | 0.6742 | 0.2848 | 0.5202 |
| v1.3 | joint ridge | 14.7202 | 18.6276 | 0.6709 | 0.2924 | 0.5197 |
| v1.4 | Elo context | 14.3523 | 18.1521 | 0.6837 | 0.2856 | 0.5315 |
| v1.5 | graph context | 14.4898 | 18.3165 | 0.6742 | 0.2817 | 0.5202 |
| v1.6 | dynamic ridge | 14.5283 | 18.4224 | 0.6736 | 0.2949 | 0.5226 |
| v1.7 | ensemble average | 14.5017 | 18.2912 | 0.6750 | 0.2738 | 0.5205 |

## Development-only selection

The matched selection run used ridge models across all seven fingerprint
families: train 2010–2021, validate 2022, and evaluate 2023–2024. 2025 and
2026 were excluded. The superseded development-output tree was removed after
selection. The current realized-ladder provenance and schema hashes are in
`docs/publication_2026/feature_manifests/selected_source_provenance.json`.

The averaged v1.7 family had the lowest raw MAE (12.9699), but the protocol
forbids an averaged primary adjustment. Among eligible non-averaged methods,
v1.4 Elo context was selected (MAE 12.9802, RMSE 16.4442, winner accuracy
0.7043, 1,488 games across two seasons). The difference from v1.7 is only
0.0103 points and v1.7 remains a supplemental comparator, not the operational
primary method.

The temporal perturbation regression in
`tests/test_opponent_adjusted_experiment.py` shows that changing week-2 game
contributions does not change week-1 adjusted contributions. The realized
canonical frame is stored outside Git at the path in the provenance manifest.
