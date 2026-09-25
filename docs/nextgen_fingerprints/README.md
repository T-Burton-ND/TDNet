# Next-generation fingerprints: exploratory setup contract

Status: **setup only**. Do not run bulk CFBD acquisition, model arrays, F09–F12 training, or diagnostic production as part of this pass. This extension does not change the frozen 2026 confirmatory study or its artifacts.

## Scientific invariant and time boundary

The unit of prediction is the **next game**. Every dynamic team-week fingerprint value comes from games completed before its target game; a same-game relationship is descriptive and cannot establish predictive value. The central outcome is that team's next-game margin. FBS-versus-FBS regular-season games are targets. FCS games may inform an FBS team's prior state, while postseason performance is excluded from numeric historical state. Pre-2010 acquisition is targeted only to initialize coach, player, or program history.

2010–2023 is base development, 2024 is validation/design evidence, 2025 is late-development/retrospective evidence, and **2026 is untouched prospective evidence**. All feature discovery, equations, fitted weights, scaling, missingness, correlation, pruning, SHAP, hyperparameters, lineage, recommendation, and design evaluation must operate on seasons through 2025. A 2026 cache may exist, but it is quarantined. Later workers must call `assert_design_operation_frame` at each design operation's input boundary and record the maximum season actually used. Do not describe 2025 as untouched for this extension.

The new F09–F12 fingerprints are market-free and pregame-win-probability-free. Betting lines, totals, moneyline, market probabilities, and CFBD pregame win probability are excluded from both model inputs and feature engineering. Market data belong in a separate evaluation sidecar for ATS, chalk, and upset metrics only.

## Generation graph and naming

The public historical ladder retains F00–F06 (market-free), F07 (market-only), and F08 (F06 plus market). The research lineage deliberately skips F07/F08: **F06 → F09 → F10 → F11 → F12**. Its two-digit identifiers use `F<generation>_<lineage>_<design>`.

| Generation | New information | Full | Reduced |
|---|---|---|---|
| F06 | Canonical 227-source-feature market-free F6 | `F06_F_a/b/c` | independent `F06_R_a/b/c` |
| F09 | Structured game microstructure | `F09_F_a/b/c` | `F09_LR_a/b/c`, `F09_PR_a/b/c` |
| F10 | Week-0 roster, recruiting, transfers, player use/production | `F10_F_a/b/c` | `F10_LR_a/b/c`, `F10_PR_a/b/c` |
| F11 | New derived coaching history | `F11_F_a/b/c` | `F11_LR_a/b/c`, `F11_PR_a/b/c` |
| F12 | Team-driven OL/QB/RB/receiving/defensive/special-teams units | `F12_F_a/b/c` | `F12_LR_a/b/c`, `F12_PR_a/b/c` |

Design `a` is canonical/atomic, choosing one primary temporal aggregation and avoiding duplicates. Design `b` adds interpretable football-informed transformations and may replace redundant `a` columns. Design `c` composes compact, film-intuitive concepts with an exact Excel-readable equation and **at most five source inputs**. Fold-safe fitted weights/scales are permitted if documented. Neither PCA nor opaque miniature models belong here. A generation number means a new information source; representation changes belong in suffixes.

`F` inherits the prior full version of the **same design** and adds the complete current family. `LR` prunes the full current-generation representation, never inheriting prior LR. `PR` starts at `F06_R` of the same design, adds all current-family information, prunes, and then inherits the previous PR in later generations. Designs never crossbreed. The earlier F6-C/C25 study supplies evidence only; it is not a seed.

F06_F_a must match the canonical F6 source list and hash in `docs/publication_2026/feature_manifests/F6.json`. F06_F_b/c and all reduced/new manifests must be materialized and checked before training. The setup manifest has only the mandatory F09 rushing pair; it is not a claim that the other families have already been engineered.

## Family design

- **F09:** Numeric structured plays, drives, halves, quarters, whole game, and the middle eight (last 4:00 Q2 plus first 4:00 Q3). Explore field position, pace, scoring opportunities, red zone, goal-to-go, short-yardage, one-score, backed-up, and two-minute situations with explicit support thresholds. Free-text play parsing and win-probability-defined leverage are forbidden. Include both `offense_rush_ypa_q4_minus_q1` and its defensive counterpart; test quarter-to-quarter rushing slopes and next-game relationships.
- **F10:** Freeze roster composition at Week 0. Prefer documented player use, then prior production, structured depth only if reliable, and recruiting as context/fallback. Match IDs first, exact normalized names second, constrained fuzzy matches last; leave uncertainty unmatched. Do not assign historical production to new players, infer injuries from nonparticipation, scrape news, or scrape combine data. Position-appropriate physical summaries are allowed where structured data are reliable. Transfer-era absence is structural missingness, not zero transfers.
- **F11:** Extend F06 coaching context with genuinely new tenure, career trajectory, prior-team history, change/interim indicators where structured data support them. Do not memorize literal identity or invent coordinator records. A weak or thin generation is a valid research result.
- **F12:** Derive ordinary numeric team-week unit states for OL pass protection/run blocking, QB, RB, receivers/TE, defensive line, linebackers, secondary, and special teams. They remain opponent-independent until `MatchupBuilder` compares home offense to away defense and vice versa.

Every source feature declares a counterpart and matchup formula. A self-pair is valid for global context. Pruning removes a pair only when **both** members qualify. Floor counts use concrete source features individually. Every final selected team-week fingerprint is universal for M1/M2/M3/M4/M5/M10; model-specific feature sets are forbidden.

## Game-state, eligibility, and aggregation

Use structured pre-play score. Starting garbage-time thresholds are absolute lead >28 in Q1, >24 in Q2, and >21 in Q3. Q4 is garbage **only if every qualifying Q4 play** has absolute lead >16; any qualifying play at 16 or below makes the whole quarter nongarbage under this rule. Garbage status supersedes time-based leverage. Location/red-zone concepts may be separately retained; `b` may compare clearly labeled all-play alternatives. Never use win probability for the classification.

Require metric-specific observation support. A candidate should generally have around 100 qualifying observations across a typical FBS season and enough team-game or quarter-level density for its use. Do not mechanically discard established base events such as sacks. A chooses one temporal representation; b adds justified windows; c compresses temporal behavior. Static preseason values are calculated once per season where possible.

The average-team reference is frozen annually for each exact fingerprint. For a season Y, use only completed team-week states through Y-1. First average rows within each FBS team-season, then average teams within each season, then average seasons equally. Store participating seasons, teams, feature hash, and reference vector. This avoids overweighting teams that played more games.

## Reduction and evaluation

Use Pearson/Spearman clustering as appropriate. Absolute correlation ≥0.995 identifies duplicate candidates; 0.98–0.995 requires representative validation; 0.90–0.98 is reported without automatic deletion. Prefer less missingness, simpler equations, stronger joint M2/M4 contribution, then earlier generation. Maintain pair closure.

Screen with M2 spline ridge and M4 histogram gradient boosting using the same ten frozen sparse setpoints per architecture and one seed. The setpoints are fixed in `nextgen_screening_setpoints_v1.json` before F09 results. Run comparable end-to-end permutation SHAP for all ten setpoints, approximately 256 background games and 512 explanation games per relevant cell; attribute home and away effects back to source features and normalize importance within architecture before voting. A/B pruning candidates must be weak/redundant under both; C may use average M2/M4 tradeoffs. The target reduced size is the smallest representation within +0.25 MAE points of its full reference, subject to floors.

F06 contributes at least 60 concrete surviving source features; each added generation contributes at least 10. If a generation has fewer than 50 survivors, retain ten with pairwise |r| <0.90 when ten legitimate independent signals exist. If fewer exist, document the shortfall and **do not fabricate junk**; whether the hard ten-feature ancestry floor can then be relaxed needs explicit disposition before reduction is finalized. A thin F11 is the likely conflict to watch.

For each generation/design, recommend one F/LR/PR lineage (F/R for F06). Report 2024 and 2025 separately. Within 0.5 MAE points of best, break ties by Brier, ATS, upset, chalk, and feature count. Retain every run, including failures, in one ultra-wide Parquet with run and fingerprint×architecture summary rows. Discard checkpoints and routine logs after extracting compact diagnostics.

## Operations and artifacts

Large artifacts live under `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/`, with a ~100 GB soft budget. Check group free space and delete temporary SHAP/scratch/log data. The acquisition plan is in `nextgen_acquisition_v1.json`; existing CFBD partitions are reused only after completeness checks. A schema/content incompatibility gets a new cache version. Do not cache a response that hits an endpoint cap as complete. Use at most three request attempts, polite backoff, and record unresolved gaps.

**CFBD preferred target: at most 20,000 new calls; hard experiment ceiling: 24,000 outbound attempts; minimum account reserve: 6,000 calls.** The live `/info` check confirmed a 30,000-call monthly account capacity on 2026-09-25. Every next-generation CFBD client shares `results/cfbd_api_call_budget.json` under the artifact root and reserves before each HTTP attempt, including retries and quota checks. `scripts/nextgen_preflight.py` writes a deduplicated request manifest and per-request ledger under `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/`. Future stage execution uses `scripts/nextgen_cfbd_acquire.py` and checks live quota before network work. Direct nextgen use of the old single-year fetch loop is disabled. The ignored repository `.env` holds `CFBD_API_KEY` with owner-only permissions; never print or commit it.

The [acquisition inventory](acquisition_inventory.md) lists current cached families, provider-era coverage, minimum legal partitions, call ranges, and fetcher fixes required before the separate launch.

SGE/UGE may have at most 50 simultaneously running jobs for this experiment, including non-array jobs. Train one generation at a time; materialization of the next may overlap. If at least three of ten configurations succeed, proceed with honest coverage. Retry only below three, at most three retry attempts, then mark incomplete and continue.

Diagnostic selection is capped at 1,000 advanced features **total across generations** by consensus architecture-normalized SHAP. Plot a duplicate formula once unless it materially differs. Retain PNG, compact Markdown, and a lightweight index under `feature_diagnostics/`, not separate per-feature observation tables. Plot a pre-target-game feature against **next-game margin** by default and next-game empirical win frequency, points for/against, or time of possession where meaningful. Same-game association is descriptive only. Spearman is primary; show Pearson where helpful. The rushing Q4-minus-Q1 pair gets a polished next-game margin/win-frequency diagnostic in the later launch. Diagnostic correlation is not the pruning criterion.

F13–F15 are design-only placeholders after F12 evaluation; their information families are not assigned here. They must add new market-free, pregame, team-week information. PCA is deferred to a derived representation suffix.

## Reuse and current gaps

Reuse `cfbd_fetch_v2.py`, `configs/features/feature_registry.yaml`, `configs/features/feature_ladders.yaml`, `MatchupBuilder` and reviewed unit pairings, `publication/scientific_shap_study.py`, the F6 compression study as evidence, SGE array and atomic Parquet fragment patterns. The nextgen manifest executor wraps the existing CFBD client and call counter: it uses legal endpoint partitions, a maximum of three attempts, atomic Parquet, content/schema hashes, and a persistent request ledger. The legacy single-year loop remains for other workflows and is explicitly blocked for the nextgen fetch config; do not route this program through the frozen confirmatory config.

Some current CFBD endpoints have provider-defined metrics whose retrospective snapshots may use future-season knowledge. Use only point-in-time reconstructible inputs or verified as-of snapshots for fingerprint values. Opaque ratings (including proprietary CORE/SP+/FPI) are not reproducible feature equations and should not be treated as newly derived football concepts.

CFBD's September 2026 coverage table says player usage begins in 2013, returning production in 2014, talent in 2015, and the portal in 2021. Treat earlier years as structurally unavailable and record coverage explicitly. Season-aggregated `/player/usage` and `/stats/player/season` cannot be used as the current season's pregame state without rebuilding through the prior completed week; they are useful directly only as prior-season history. Enriched passing/rushing endpoints begin in 2025 and cannot define an all-years baseline.

Sources: [CFBD API reference](https://api.collegefootballdata.com/api/plays), [CFBD availability](https://api.collegefootballdata.com/data-availability), [CFBD player endpoints](https://api.collegefootballdata.com/api/players), [Football Study Hall advanced stats glossary](https://www.footballstudyhall.com/2018/2/2/16963820/college-football-advanced-stats-glossary), [Football Study Hall IsoPPP discussion](https://www.footballstudyhall.com/2014/1/27/5349762/five-factors-college-football-efficiency-explosiveness-isoppp).
