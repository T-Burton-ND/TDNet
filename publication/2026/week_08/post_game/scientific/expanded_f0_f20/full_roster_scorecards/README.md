# 2026 scientific F0–F20 scorecards through Week 8

Each CSV has 155 rows: 126 individual F×M models, 21 fingerprint consensuses, six architecture consensuses, one available-cell roster consensus, and one Vegas reference. M1–M5 and M10 are the six architectures. Fingerprint and architecture consensuses average the constituent home-win probabilities and predicted home margins **per game**, then score those predictions; their metrics are not averages of model metrics.

| Table | Model coverage | Use |
| --- | --- | --- |
| `week08_latest_available.csv` | F0–F19: 53 games; F20: 49 | All saved Week 8 predictions. |
| `week08_matched_49.csv` | Every cell: 49 games | Direct Week 8 model comparison. |
| `season_latest_available.csv` | F0–F19: 433 games; F20: 369; union: 441 | All available 2026 games through Week 8. |
| `season_matched_361.csv` | Every cell: 361 games | Direct 126-model season comparison. |

The latest-available architecture and whole-roster consensuses have 1, 20, or 21 fingerprints (6, 120, or 126 models) per game: 361 shared games, 72 F0–F19-only games, and eight Week 1 F20-only games. F20 is absent in Week 7 and four Week 8 games. The matched tables use all 126 cells on every game. `models_per_game_min` and `models_per_game_max` make that explicit. A separate previously published 353-game comparison additionally requires a Vegas probability for every game; the 361-game matched table keeps eight games without an archived Vegas probability, so its Vegas Brier has 353 observations. The latest-available season Vegas row has 432 spread lines and 425 probability quotes out of the 441-game union. Vegas SU and margin use the archived spread when present; Vegas Brier uses the archived probability. Vegas has no ATS pick against its own line, so its ATS cells are N/A. ATS percentages exclude pushes, missing lines, and exact no-picks; counts retain each category.

The model rows were recomputed from saved game-level forecasts and certified results. All 126 weekly matched and latest-available season model rows reconcile to the previously published scorecards; a 353-game subset also reconciles to the previously published paired scorecard. F20 Weeks 1–6 are retrospective replay on an assumed weekly expert archive. Week 8 F20 probabilities were late pregame research revisions, while its margins were applied after the games using models fitted before kickoff. Many F0–F19 early-week scores are retrospective backfills; treat season rankings as descriptive.

The two PNGs summarize the Vegas, roster, fingerprint, and architecture rows. The CSVs retain every individual model and exact denominator.
