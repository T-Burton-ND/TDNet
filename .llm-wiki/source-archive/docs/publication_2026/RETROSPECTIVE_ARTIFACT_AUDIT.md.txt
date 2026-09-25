# Retrospective artifact audit

The former `data/comparisons/season_2026_full_roster_vs_vegas` tables were
workflow diagnostics, not leakage-safe prospective backtests. They were fit
through 2025 and then applied to 2023–2025 historical seasons, so they were
deleted rather than retained beside claimable evidence.

The apparent perfect KNN rows in
`season_2025/tables/model_score_matrix.csv` (MAE 1.491 and accuracy 1.0) were a
warning sign consistent with that training/evaluation contamination. They are
deleted and excluded from publication claims.

The valid retrospective evidence is the externally archived four-way 2025
regeneration. Its holdout packages use checkpoints trained only through 2024;
the through-2025 companions are labeled pipeline dry runs and are not used for
performance claims.

This audit is intentionally conservative: it prevents an attractive but
invalid artifact from being promoted into the current F0–F8 publication matrix.
