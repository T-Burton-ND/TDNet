# 2026 Week 8 pregame forecasts

The packages below cover the same 53 FBS games that had not started at the October 21 freeze. Three Week 8 games had already started and have no forecasts here. `forecast_roster_coverage.json` checks game IDs, model counts, duplicate cells, valid predictions, and pre-kickoff publication times.

| Roster | Model cells per game | Forecast package |
|---|---:|---|
| Operational margin | 33 | `frozen_bundle/public/predictions.parquet` |
| Original market-free scientific F0–F6 | 42 | `scientific/scientific_all_game_predictions.csv` |
| Full scientific F0–F8 comparison | 54 | `scientific/full_f0_f8/scientific_all_game_predictions.csv` |
| Expanded scientific F0–F19 research | 120 | `scientific/full_f0_f19/scientific_all_game_predictions.csv` and `scientific/full_f0_f19/scientific_consensus_game_predictions.csv` |

The margin, F0–F6, and F0–F8 packages retain their original freeze manifests. The expanded research package was completed at 15:47 UTC on October 21 and has a separate `scientific/full_f0_f19/cutoff_proof.json`, manifest, 20 fingerprint-level consensuses, and a 53-game consensus figure. Its latest documented game-derived source was available October 20 03:00 UTC after the required 48-hour lag; its frozen market summary was captured October 21 14:08 UTC; the first remaining kickoff is October 22 00:00 UTC. No Week 8 target scores entered the feature state or model fit. Provider field-level historical publication times and historical quote-level market times remain unverified, and the expanded roster is exploratory after prior 2026 research.

The online contest submission used only the operational margin consensus and is recorded under `tools/cfbd_pickem/audit/2026_week08/`. Current cumulative performance charts end with the Week 7 retrospective replay at `publication/2026/week_07/post_game/retrospective_replay/cumulative/`; Week 8 cannot be scored until certified final results are available.
