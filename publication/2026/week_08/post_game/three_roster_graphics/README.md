# Week 8 postgame · all three roster graphic types

`coverage_matrix.csv` maps 27 graphic types to every one of the three roster views. Entries marked `existing_week08` point to verified graphics already in the Week 8 package; new files here fill the other cells. The rejected `lost_graphics/` review styles are not included. All new figures are built from saved scored forecasts and polls, without refitting models.

- Margin-wide: 33 models, 53 frozen Week 8 games, 378 games from Weeks 0–6 and 8.
- Original scientific F0–F6: 42 models, 53 frozen Week 8 games, 378 games from Weeks 0–6 and 8. The saved Week 6 F0–F8 component supplies its 42 F0–F6 cells; all checkpoint hashes match the frozen Week 8 roster. This fills a gap in the prior 320-game F0–F6 cumulative figure without changing that earlier artifact.
- Expanded scientific F0–F20: 126 models on 49 Week 8 games; cumulative matched graphics use 353 shared games from Weeks 1–6 and 8. This snapshot excludes Week 7. Its earlier expert-input replay is retrospective; Week 8 F20 margins were applied after games to archived pregame features. Its power rankings are a descriptive 353-game prediction-network fit, not a live Week 9 state poll. A later [retrospective Week 7 backfill](../../../week_07/post_game/retrospective_replay/scientific/expanded_f0_f20/README.md) adds all 59 Week 7 games and updates the paired season cohort to 420 separately from this 353-game view.

The six betting graphic types use the same 49 Week 8 games for all three rosters. ATS assumes -110; moneyline uses saved pregame odds. Flat stakes risk $10 per eligible game. Scaled stakes use $0 at 49.9% confidence and $25 at 100%; ATS confidence is the fraction of models backing the consensus spread side, while moneyline confidence is the consensus picked-team probability. Threshold sweep points require at least ten eligible games and are descriptive in-sample results. Vegas has no ATS record against its own line.

`manifest.json` records source and generated-image SHA-256 hashes. The regular weekly renderer's files were left intact; this completeness pass adds the missing views for the latest postgame package.
