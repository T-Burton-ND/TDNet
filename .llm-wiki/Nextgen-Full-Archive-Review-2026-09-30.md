---
type: analysis
up: "[[Next-Generation-Fingerprint-Experiment]]"
derived_from: ["[[Nextgen-Historical-Archive-Completion]]", "[[Nextgen-Empirical-Lessons-2026-09-28]]"]
tags: [archive, audit, coverage, source-semantics, nextgen]
---

# Nextgen Full Archive Review — 2026-09-30 UTC

The archive supports five concrete research directions; the review identified and locally repaired a play-taxonomy omission, whose effect must be separated from future feature gains.

## Question

Do we have the data for the next four or five generations of team-week fingerprints, and what does the full archive actually support?

## Context

The user requested a full review and public Markdown plans, with no archive upload. This is an offline review of the historical nextgen archive through 2025 after [best-effort recovery](Nextgen-Historical-Archive-Completion). No requests, acquisitions, training, canonical rebuilds, or scheduler changes occurred. Review date is September 30 UTC / September 29 US Eastern. The scientific goal remains paused. The [F13–F17 roadmap](Nextgen-F13-F17-Roadmap) contains proposals, not accepted generations or measured gains.

## Analysis

### Integrity and the meaning of completeness

The scan enumerated **14,669 ledger records across 70 endpoint paths**, then read and hashed every distinct SHA-bearing cache path: **14,102 files, 410,170,510 bytes**, with **zero hash, Parquet-footer, or recorded-row-count discrepancies**. All **14,086 Parquet files under `raw_cache/`** have a ledger reference; every complete record has a SHA. The larger hashed-file count also includes referenced caches outside `raw_cache/`.

Ledger statuses are 14,085 `success_complete`, 240 `skipped_existing_complete`, 170 `needs_review`, 71 `failed_final`, 102 `planned`, and one `success_suspected_partial`. These are request records, not distinct games, API calls, or unique observations. The 170 review records include the already documented empty probes; planned/failed auxiliary requests are not evidence that all provider routes were exhausted. The capped attribution partition remains excluded from complete derivation. Multiple conference and game records represent overlapping events; never sum their response rows as unique observations.

The approved attribution scope finished with **10,639 / 10,704 games (99.39%)** cached for 2013–2025, **65 provider-empty games**, and no remaining retry failures within that game scope. Of those 65, 16 have ordinary plays and 49 have earlier empty targeted play queries. The all-endpoint ledger still has auxiliary failures. The cumulative acquisition budget remains **5,629 / 20,000 attempts**; this review spent zero. Integrity means the saved source is unchanged, not that the source is complete or historically available as of each game.

### Structured plays and drives: exhaustive field review

Across the authoritative 2010–2025 regular-season FBS-involved schedule, the 245 weekly partitions contain **2,351,115 plays in 12,964 / 13,059 expected games**, with 95 missing games. All 2,351,115 play `(game_id, drive_id)` keys join to the **327,915** schedule-filtered drive rows. There are zero duplicate `(game_id, play_id)` keys, zero missing drive/play sequence positions, and zero duplicate drive identity keys. There are **28 repeated `(game_id, drive_number, play_number)` positions**: one in 2021 and 27 in 2025. A stable ID sort is not proof of event order; sequence-derived features must resolve or exclude ambiguous sequences.

Under the **original saved F09 classifier**, 1,640,231 rows qualify as rush/dropback plays. Of those, **1,635,016 (99.68%)** have regulation period, nonnegative distance, field position in [0,100], valid minute/second ranges, and a reconstructable pre-play score. None lack clock values; 150 lack reconstructed pre-play score. The 5,215 rows outside the combined condition include overtime and invalid/unknown context, not just missing values. Existing time/garbage filters retain 1,392,803 plays. These counts describe the original classifier and must not be represented as all football scrimmage plays.

**New material finding: historical play types are omitted by that classifier.** The archive contains **118,823 `Pass Completion` rows** (118,041 in 2010–2013 and 782 in 2025) and **5,816 `Pass Interception` rows** (2010–2013). Neither literal occurred in the original saved `PASS_TYPES` allowlist. Thus those rows cannot contribute as qualifying dropbacks to the original F09 sufficient statistics. This is confirmed against code SHA `eb051a8c0d05e6f7f3f0ecd16cfbfe5cddda088c0aba6bab1878fa77b269c98e`, also recorded by the saved F09 partition metadata. The 2013 archive has only 36 rows explicitly labeled `Sack`; that is a taxonomy/source warning, not evidence of an exceptionally low football sack rate. The review does not infer where differently classified sacks went. These counts and the table below describe the original saved classifier, before the authorized repair.

This finding limits interpretation of historical pass-related F09 features and descendants. It does **not** invalidate the fact that the recorded pipelines produced the saved scores, quantify the effect on those scores, or explain the observed gain. Do not silently replace existing artifacts: retain their source version and compare a corrected rebuild separately before assessing a new family. Inspect structured type definitions and reconciliation with box scores; do not assume label equivalence solely from names. The source scores are post-play; current code already reconstructs pre-play scores by chronological shift. Preserve that correction and explicitly audit ambiguous sequences before context conditioning.

| Season | Observed / expected games | Original qualifying plays | Complete regulation context | Repeated sequence positions |
|---|---:|---:|---:|---:|
| 2010 | 751 / 773 | 73,288 | 73,002 | 0 |
| 2011 | 768 / 777 | 75,796 | 75,511 | 0 |
| 2012 | 790 / 805 | 79,657 | 79,282 | 0 |
| 2013 | 813 / 813 | 80,176 | 79,793 | 0 |
| 2014 | 812 / 829 | 114,035 | 113,690 | 0 |
| 2015 | 822 / 829 | 114,842 | 114,410 | 0 |
| 2016 | 816 / 832 | 114,144 | 113,698 | 0 |
| 2017 | 829 / 834 | 113,853 | 113,438 | 0 |
| 2018 | 842 / 845 | 116,010 | 115,671 | 0 |
| 2019 | 847 / 848 | 115,179 | 114,804 | 0 |
| 2020 | 542 / 542 | 74,312 | 74,070 | 0 |
| 2021 | 849 / 849 | 113,726 | 113,539 | 1 |
| 2022 | 854 / 854 | 113,927 | 113,617 | 0 |
| 2023 | 868 / 868 | 112,764 | 112,582 | 0 |
| 2024 | 873 / 873 | 113,112 | 112,852 | 0 |
| 2025 | 888 / 888 | 115,410 | 115,057 | 27 |

### Player attribution: useful roles, incomplete participation

The final successful-game audit includes two sparse 2012 probes in addition to the approved bulk era: **10,641 games, 2,161,110 actor/stat rows, 1,436,178 distinct attributed plays out of 1,925,676 canonical plays**. Not every canonical event should have an actor association; this fraction is not a snap-participation recall measure. There are zero unmatched attributed rows or missing athlete IDs, but **33,260 missing expected rush associations, 13,341 rushing-yardage disagreements, 14 rushing-team disagreements, and 40 duplicate actor/stat rows**. Treat canonical play context/outcome and athlete attribution as separate sources; quarantine conflicting joins rather than silently choosing a convenient value.

The supported bulk era is 2013–2025. The two 2012 successes contain just four field-goal association rows; the earlier sampling decision excluded bulk 2012 rather than proving universal source absence. Within successful 2013 caches, 55,204 of 60,366 expected rushes have matching attribution (91.45%); in 2021 the figure is 51,639 / 57,740 (89.43%); in 2025 it is 57,723 / 59,236 (97.45%). These denominators exclude provider-empty game caches and expose year-dependent selection.

Across successful game caches, stat labels include 721,314 Rush rows, 389,687 Completion, 383,493 Reception, 136,888 Target, 38,676 Sack Taken, and 30,071 Sack. **Target is not established as a complete attempted-target denominator:** for 2024 there are only 3,132 Target rows versus 31,139 Reception rows; 2025 has 17,267 versus 32,912. These labels could partition events rather than nest; validate their joint semantics before defining target share. Only 1,110 Tackle and 1,083 QB Hurry rows exist in the entire successful-game audit. Consequently the archive cannot establish complete defensive participation, offensive-line blocking, formations, routes, snap counts, or injury status. F15 should begin with observed rushing/reception roles and explicitly measured association support.

### Other families and temporal limits

- **Schedule and environment:** all 13,059 authoritative games have kickoff timestamps. Venue IDs are missing for 115 games; 12,943 join to a venue with latitude/longitude. Weather has temperature and wind for 12,735 distinct scheduled games. These are retrospective actuals, not archived pregame forecasts. Venue metadata also does not prove historical surface/renovation state. Rest intervals can be derived, but future travel/environment features must add something beyond F06's existing `travel_distance_diff` and `travel_tz_diff`. Environment is a reserve direction, not required for F13–F17.
- **Roster and recruiting:** 2010–2025 caches exist. The existing roster audit finds `year` mixes requested season and class-like values; it is not a reliable universal experience field. Current roster snapshots do not certify Week-0 historical membership. Recruiting identities and prior observed appearances need cutoff-safe joins. Player-PPA binding previously matched 210,820 rows after removing 3,002 exact duplicates from 258,553 source rows; that is not proof all archive players have game identities.
- **Transfers and returning production:** portal caches span 2021–2025 and the inspected schema has names but no athlete ID; returning-production caches span 2014–2025. They cannot be treated as uniformly observed 2010–2025 player movement or current-season lineup state. Transfer dates and identity resolution require separate validation.
- **Coaches and units:** annual coaches/season summaries are available, but `/coaches/tenures` has no successful response. End-of-season summaries do not prove pregame assignments. Existing F11/F12 coverage and negative comparisons remain relevant; more cached columns do not establish added predictive value.
- **Ratings, PPA, WEPA, season summaries:** many are cached broadly, but present-day retrospective estimates and full-season aggregates require historical as-of and estimator-training validation. No provider pregame win probability, market line, target outcome, or 2026 observation becomes eligible merely because it is cached. The passing/rushing endpoint extensions represented here cover 2025 only.
- **Existing canonical families:** the saved family-coverage audit has no entirely missing training feature in its listed families, yet early observed continuity can be 100% missing in a season and recruit-history cell missingness reaches roughly 64–66% in its worst listed seasons. This argues for explicit era/support controls, not treating missingness as a football trait.

### What the completed models actually teach

The [paired empirical lessons](Nextgen-Empirical-Lessons-2026-09-28) justify testing play-derived information: in the frozen 241-success snapshot, F09 M4 improved all ten shared configurations in 2024 in each A/B/C design, but less consistently in 2025. F10 A/M2 improved nine of ten in 2024 and eight of ten in 2025. F11 A/M2 worsened on common evaluation games; F12 A/M2 worsened all ten shared configurations in both years versus F11. These are observed pipeline comparisons, not proof of a mechanism or uselessness of whole data families. The taxonomy finding is an additional confounder to investigate, not a post-hoc explanation.

The completed 360-configuration table contains 356 successes and four failures. F06/F09/F10 evaluated 746/757 development games; F11/F12 evaluated 626/553. Raw medians across these cohorts do not rank generations fairly. Existing paired results align evaluation games but do not control training cohorts. Future comparisons must align both. 2024/2025 are design-informed; 2026 remains quarantined. See the existing completed-results page through the experiment contract for the full table; this review did not rescore predictions.

### Complete ledger endpoint inventory

The following is an exhaustive inventory of endpoint paths represented in the ledger, not all possible provider endpoints. “Complete” includes reused complete records; a complete request need not contain all desired events. Years are request years with complete records; “unkeyed” means the request has no year key. Other statuses use R=needs_review, F=failed_final, P=planned, S=suspected_partial. Response rows are deliberately omitted because their scopes overlap.

| Endpoint | Complete records | Complete request years | Other statuses |
|---|---:|---|---|
| `/calendar` | 16 | 2010–2025 | — |
| `/coaches` | 16 | 2010–2025 | — |
| `/coaches/seasons` | 16 | 2010–2025 | — |
| `/coaches/tenures` | 0 | none | F=1, P=15 |
| `/conferences` | 16 | 2010–2025 | — |
| `/conferences/affiliations` | 16 | 2010–2025 | — |
| `/conferences/changes` | 16 | 2010–2025 | — |
| `/draft/picks` | 16 | 2010–2025 | — |
| `/draft/positions` | 1 | unkeyed | — |
| `/draft/teams` | 1 | unkeyed | — |
| `/drives` | 16 | 2010–2025 | — |
| `/games` | 16 | 2010–2025 | — |
| `/games/media` | 16 | 2010–2025 | — |
| `/games/players` | 245 | 2010–2025 | — |
| `/games/teams` | 245 | 2010–2025 | R=9 |
| `/games/weather` | 16 | 2010–2025 | — |
| `/lines` | 13 | 2013–2025 | P=3 |
| `/metrics/fg/ep` | 1 | unkeyed | — |
| `/metrics/wp/pregame` | 13 | 2013–2025 | P=3 |
| `/passing/players/games` | 16 | 2025 | F=1 |
| `/passing/players/season` | 1 | 2025 | — |
| `/passing/plays` | 16 | 2025 | — |
| `/passing/teams/games` | 16 | 2025 | F=1 |
| `/passing/teams/season` | 1 | 2025 | — |
| `/player/portal` | 5 | 2021–2025 | — |
| `/player/returning` | 12 | 2014–2025 | — |
| `/player/usage` | 13 | 2013–2025 | — |
| `/plays` | 245 | 2010–2025 | R=95 |
| `/plays/stats` | 12,432 | 2012–2025 | R=66, S=1 |
| `/plays/stats/types` | 1 | unkeyed | — |
| `/plays/types` | 1 | unkeyed | — |
| `/ppa/games` | 16 | 2010–2025 | — |
| `/ppa/players/games` | 200 | 2013–2025 | F=46, P=15 |
| `/ppa/players/season` | 13 | 2013–2025 | F=3 |
| `/ppa/teams` | 16 | 2010–2025 | — |
| `/rankings` | 16 | 2010–2025 | — |
| `/ratings/core` | 10 | 2016–2025 | P=6 |
| `/ratings/elo` | 16 | 2010–2025 | — |
| `/ratings/fpi` | 16 | 2010–2025 | — |
| `/ratings/sp` | 16 | 2010–2025 | — |
| `/ratings/sp/conferences` | 16 | 2010–2025 | — |
| `/ratings/srs` | 16 | 2010–2025 | — |
| `/ratings/srs/expanded` | 15 | 2010–2019, 2021–2025 | P=1 |
| `/records` | 16 | 2010–2025 | — |
| `/recruiting/groups` | 1 | unkeyed | — |
| `/recruiting/players` | 16 | 2010–2025 | — |
| `/recruiting/teams` | 16 | 2010–2025 | — |
| `/roster` | 16 | 2010–2025 | — |
| `/rushing/players/games` | 16 | 2025 | F=1 |
| `/rushing/players/season` | 1 | 2025 | — |
| `/rushing/plays` | 16 | 2025 | — |
| `/rushing/teams/games` | 16 | 2025 | F=1 |
| `/rushing/teams/season` | 1 | 2025 | — |
| `/stats/categories` | 1 | unkeyed | — |
| `/stats/game/advanced` | 16 | 2010–2025 | — |
| `/stats/game/havoc` | 16 | 2010–2025 | — |
| `/stats/player/season` | 16 | 2010–2025 | — |
| `/stats/player/success` | 13 | 2013–2025 | F=1, P=2 |
| `/stats/player/success/game` | 200 | 2013–2025 | F=15, P=30 |
| `/stats/season` | 16 | 2010–2025 | — |
| `/stats/season/advanced` | 16 | 2010–2025 | — |
| `/talent` | 11 | 2015–2025 | — |
| `/teams` | 16 | 2010–2025 | — |
| `/teams/ats` | 0 | none | F=1, P=15 |
| `/teams/fbs` | 16 | 2010–2025 | — |
| `/venues` | 1 | unkeyed | — |
| `/wepa/players/kicking` | 10 | 2016–2025 | P=6 |
| `/wepa/players/passing` | 13 | 2013–2025 | P=3 |
| `/wepa/players/rushing` | 13 | 2013–2025 | P=3 |
| `/wepa/team/season` | 16 | 2010–2025 | — |

### Authorized local repair and measured validation

After the initial review, the user authorized necessary repairs/additions. The archived `/plays/types` catalog confirms both legacy labels (IDs 4 and 6). The local `nextgen_microstructure.py` now includes them without a year restriction, orders plays by numeric period before drive/play position, and marks pre-play score unknown for unresolved same-period sequence ties and their immediate successors. It does not infer ambiguous play kinds from prose or reconstruct absent sack labels. Existing raw data, canonical fingerprints, predictions and model results were preserved.

Twelve focused tests in `tests/test_nextgen_microstructure.py` and `tests/test_nextgen_f09.py` pass, including regression cases for old/new pass vocabulary, reciprocal defensive denominators, reused overtime positions and ambiguous-score exclusion. A second offline pass over all 2,351,115 archived plays completed successfully:

| Classifier diagnostic, same archive | Original | Repaired |
|---|---:|---:|
| Qualifying rush/dropback plays | 1,640,231 | 1,764,870 |
| Qualifying plays with complete regulation context | 1,635,016 | 1,759,243 |
| Qualifying plays lacking pre-play score | 150 | 153 |
| Time/garbage-eligible plays | 1,392,803 | 1,502,070 |

The repair recognizes **124,639 additional dropbacks**. The increase in unknown-score rows is a conservative handling of ambiguous chronology, not invented context. The repair is committed locally as `3cf8648` on `experiment/nextgen-fingerprints-v1`. The repaired source SHA is `c1fbfa1981103e8f198ee958f2e416c75299e80c4cf34de21ec176e3932b02ed`. Private `archive_review_20260930/repair/` contains `field_profile.json`, `repair_receipt.json`, `profile.py` and its log. This is a classifier/source repair and validation, not a new F13 artifact, canonical rebuild or model experiment. Existing F09 cache bindings include the microstructure code hash, so a future rebuild must not reuse old sufficient statistics as if they were repaired. Predictive effects remain unmeasured. The public push contains Markdown only; repair source/tests remain local.

## Conclusion

**Yes: the saved plays, drives, schedule and actor associations support a concrete F13–F17 research plan without further acquisition.** Readiness means enough source material to implement and audit candidates, not feature validity or a prediction gain. F13/F14 have the strongest raw foundation after taxonomy/sequence validation; F15 is restricted to observed roles in supported eras; F16/F17 depend on building reliable preceding states. Full defensive participation and historically certain current lineups remain unsupported. More API calls are not the immediate bottleneck.

## Open follow-ups

1. The two confirmed label omissions and chronology handling are repaired locally; complete the broader historical taxonomy reconciliation before building new families, preserve existing results and compare the correction alone.
2. Carry the repaired chronology rules into any future series parser; validate within-game missing-event consistency beyond cross-table identity joins and the original 28 repeated positions.
3. Validate actor stat-label meaning and year/team/context support before role-share denominators or current-player assertions.
4. Produce cutoff-safe feature coverage and common training/evaluation cohorts before any new model run. These are planned steps, not executed here.

## Reproduction and evidence

Private root: `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/`. Review outputs and scripts are preserved in `results/preflight/archive_review_20260930/`: `inventory.py` hashes ledger caches and checks footers; `profile.py` filters to the authoritative schedule and measures play/drive/context/weather fields using existing `play_flags`; `semantics.py` counts literal play/actor stat labels. `ledger_snapshot.json`, `verified_files.json`, and `file_reconciliation.json` preserve the inventory boundary. Scripts read sources and write review outputs only. The scripts' output-directory constant points to `/tmp/tdnet_archive_review`; adjust that path for a later reproduction without overwriting this snapshot. Source-code HEAD at initial review was `97a4b25`; the subsequent authorized local repair is documented above. Reproducing the pre-repair counts requires that original code version; the saved repair profile uses the repaired code.

Additional source reports: `results/f09_play_coverage.json`, `results/roster_semantics_audit.json`, `results/player_ppa_game_binding.json`, `results/canonical_family_coverage.json`, and `results/preflight/archive_completion_20260928/attribution_coverage.parquet` and `attribution_audit.json`. The review is exhaustive over ledger integrity and ordinary-play fields, plus the existing exhaustive actor audit; it is not a cell-by-cell semantic certification of every nested auxiliary endpoint or a new audit of model training.

| Private review artifact | SHA-256 |
|---|---|
| `inventory.json` | `b8c8d40cbd485dfb8f9e1ff56220ddd862da4c0dcd7807045689fb1157f091bd` |
| `field_profile.json` | `fe0b946cf7a0e414ac770483da846d14930ede367149b2676a6ee2915390ad6f` |
| `play_type_profile.json` | `c8de7177a425ce82511eb3ad5a509b1b4a08e77c7822040da8a7501f6442b5ae` |
| `actor_type_profile.json` | `e56eb2509b07cf029bbe4126c24618f61051558ed10df9227ac8fd63d2127708` |

Only Markdown findings and plans are published for this review; these private artifacts and raw data are not uploaded.

See also: [F13–F17 roadmap](Nextgen-F13-F17-Roadmap) · [Archive completion](Nextgen-Historical-Archive-Completion) · [Empirical lessons](Nextgen-Empirical-Lessons-2026-09-28) · [Experiment contract](Next-Generation-Fingerprint-Experiment).
