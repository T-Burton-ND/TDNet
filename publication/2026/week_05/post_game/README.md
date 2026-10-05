# Week 5 postgame results

This package scores the frozen Week 5 predictions against the completed CFBD results. The results check passed for all 56 games with no missing game IDs, duplicate IDs, or missing final scores. The 33-model operational scorecard and the 42-model scientific F0–F6 scorecard are included; a separate F0–F8 scientific cohort is retained under `scientific/full_f0_f8/`.

The published Week 5 prediction CSV was restored byte-for-byte from the local archived release (SHA-256 `dd2ad005dbb9d9353088a0b1e4df5716ab4bb34115fffc305ff927e594863543`). The restored scoring bundle verified with 1,848 prediction rows. The completed results file used for scoring has SHA-256 `7935b9af5294f09db93d33521825f45b1a2e1a413d31e9e37fd46c4b88da45cb`.

The best F0–F6 scientific margin MAE was 13.116 points (`scientific_F2_M2`); the best straight-up accuracy was 78.6% (`scientific_F6_M3`). These are single-week descriptive results over 56 games.
