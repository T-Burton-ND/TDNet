---
type: entity
up: "[[Next-Generation-Fingerprint-Experiment]]"
related: "[[Fingerprint-F13-Proposal]]"
tags: [acquisition, archive, budget, attribution]
---

# Nextgen Historical Archive Completion

The user approved completing useful historical data on 2026-09-28, reusing valid caches and retaining the original 20,000-attempt budget; model experiments remain paused.

## Completed pass and audit — 2026-09-29

The acquisition pass finished at 21:01 UTC and its audit/quota readback finished at 21:10 UTC (5:10 p.m. America/New_York). This supersedes the historical interim operational statements below. The background executor has exited. The pass is complete **with unresolved games**, not a claim of a complete source archive.

For the approved 10,704 regular-season FBS-involved games in 2013–2025, 10,637 have successful caches (99.4%): 8,873 derived from complete conference partitions and 1,764 directly cached responses. There are 67 unresolved games: 54 empty conference-partition unions, 11 empty direct responses and two retained earlier retry failures. These have not been converted to zeros or asserted permanently unavailable. The two failures retain three throttled attempts and two interrupted attempts respectively; targeted recovery and empty-union review remain open.

The resumed plan fetched/reused 1,697 conference partitions, of which 1,696 were complete and one hit the response cap; the capped partition was excluded from game reconstruction. The direct/fallback phase contained 1,690 games. The final verification checked 1,789 distinct source partitions used by derived caches, including partitions retained from earlier plans.

The original shared budget finished at **5,571 / 20,000 attempts**, including retries and quota reads (3,713 additional reservations since the pre-acquisition count of 1,858). Provider quota readback reported 24,594 calls remaining, preserving the 10,000-call reserve. Local reservation and provider counters are distinct. No new model experiment or canonical rebuild was performed by acquisition.

The final offline audit includes all 10,637 successful approved games plus the two sparse 2012 probe caches: **10,639 cached game responses, 2,160,775 attribution rows and 1,435,920 distinct attributed plays** against 1,925,317 canonical plays in those games. There were zero unmatched attributed rows, zero missing athlete IDs and zero rushing-period disagreements. It also found **33,254 missing canonical rush attributions**, 9,327 extra rush-attributed play IDs outside the canonical rush-type set, **13,330 row-level rushing-yardage disagreements**, 14 rushing-team disagreements and 40 duplicate actor/stat rows. These are source-quality findings, not predictive model results. Clean play-ID joins therefore do not establish complete event coverage or universally consistent attribution; all source values and missingness are retained.

[Final-pass evidence](evidence/Archive-Final-Pass-Audit-2026-09-29.json) includes exact year/status counts, the completion receipt, unresolved game identities and the per-game audit hash. The audit JSON and per-game Parquet hashes were checked again when filing this result. The separate 104 empty play/box gap probes and bulk-2012 exclusion remain as described below. Further recovery of the 67 attribution gaps has not been executed at this cutoff.

## Authorization and source boundary

This direct instruction supersedes the earlier autonomous Stage E **skip** decision for raw archive acquisition. It does not assert that the old F10/F12 feature-adoption requirements have been met. The new archive can support future observed player-event and contextual analyses; it cannot establish complete participation, roster positions, or historical Week-0 availability. No canonical training input, trained model or prediction is changed.

Preserve the original `results/cfbd_api_call_budget.json` ledger under `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen`; it held 1,858 reservations before this work. Retries and quota checks count. The 10,000-call provider reserve and storage guard remain. 2026 and postseason data are excluded from this acquisition.

## Completed reconciliation and samples

- The base acquisition was already cached. Five weekly team-box partitions remained coverage-incomplete after two previous fetches each. Their nine missing game IDs were queried directly instead of repeating the weekly calls.
- Ninety-five missing play-game IDs were investigated with documented team/year/week queries. All 95 play queries and all nine box queries returned empty HTTP 200 responses. Their records retain `needs_review`; no zero data or permanent-absence inference was created.
- The game-level attribution inventory contains 11,509 regular FBS-involved games for 2012–2025. Three 2012 probes returned one empty response and two two-row, single-field-goal responses. Bulk 2012 acquisition was therefore excluded for low demonstrated value: 805 games in that era, including the three retained probes. This does not prove that every 2012 game lacks attribution.
- The approved bulk scope is the remaining 10,704 games in 2013–2025. The original four sample games were reused. New era probes had clean play-ID joins and athlete IDs; existing sample yardage disagreements and incomplete rushing attribution remain source limitations.

Evidence directory: `results/preflight/archive_completion_20260928/`. `plan.json` preserves the initial game-level request plan; `era_samples.json`, `era_sample_audit.json` and `reviewed_gate.json` record actual responses and the acquisition decision. The earlier `plays_stats_sample_audit.json` and `plays_stats_approval.json` are preserved.

## Verified batching reduces unnecessary calls

Three conference/week responses were compared against existing game responses. All 22 conference-specific subsets across 17 games matched exactly. The API filter selects the **attributed team's conference**, so complete game reconstruction requires both conferences' complete partitions. Evidence: `batch_samples.json` and `batch_validation.json`.

The optimized initial plan has 1,790 conference/week requests, 1,680 direct game requests for small groups/FCS opponents, and 87 existing verified game caches. Capped/unavailable batches require individual-game fallbacks, so 3,470 is a first-attempt plan, not a final consumption figure. Exact 2,000-row responses remain suspected partial and are never used to reconstruct complete games.

A derived game cache binds source partition hashes and records zero HTTP attempts for derivation. Prior failed-attempt history is retained. Existing successful game caches are verified and preserved; source discrepancies and duplicate events are not silently repaired. No data is invented for an empty partition union.

## Provider throttling and operational state

Initial four-per-second acquisition hit HTTP 429 and stopped. Four concurrent conference queries also throttled; reducing concurrency to two still produced intermittent throttles. The initial adaptive executor started batch attempts at least five seconds apart with at most two concurrent requests. On a 429, it waits at least 65 seconds (longer for a numeric Retry-After) and doubles the interval, capped at sixty seconds. These are our adaptive settings, not a claim about a documented provider rate limit. Small game queries begin at a 1.1-second interval and adapt separately.

**Interim state at 2026-09-28 17:14 UTC:** 51 complete conference batches were cached, containing 43,302 raw attributed-event rows across those batches; 2,132 total experiment attempts were reserved. Acquisition and the final full-archive audit remain in progress. Do not report the whole archive complete from these interim counts.

`batch_status.json` contains the latest executor checkpoint. A finished dispatcher is not proof every expected source exists. `batch_reconstruction.json` records reconstructed and fallback game counts. Acquisition is reversible/resumable at verified request boundaries; it holds an exclusive executor lock and never resets the budget.

## Verification and remaining work

### Interim measured coverage, 17:18 UTC

The offline audit covered 494 successful game caches: 475 from 2013, six from 2014, one each from 2015–2025, and two sparse 2012 probes. This early, chronologically concentrated subset is not representative evidence for the full archive. Its 85,573 attribution rows joined to 91,776 canonical plays, covering 64,481 distinct play IDs. There were zero unmatched attribution rows, missing athlete IDs, duplicate actor/stat rows, rushing-team mismatches or rushing-period mismatches in this subset.

The same subset contains 36,471 expected canonical rushing plays, of which 33,161 have rushing attribution and 3,310 do not. There are 16 additional rushing-attributed play IDs outside the canonical rush-type set and 3,118 row-level rushing-yardage disagreements. Missing rushing attribution occurs in 415 audited games and yardage disagreements in 479. These are measured source limitations, not imputed nonparticipation or model-performance results. Future contextual features must preserve event support/missingness and use the original structured-play outcome when joining yardage; player-event attribution alone does not reproduce all rushing outcomes.

[Saved numerical evidence](evidence/Archive-Interim-Audit-2026-09-28.json) records the exact cutoff, year/status counts and evidence hashes. The full original audit and per-game Parquet are preserved under `results/preflight/archive_completion_20260928/interim_audit_171827/`. The final audit was pending at that interim cutoff; see the completed-pass section above.

The acquisition tools and guards passed 22 tests before the final adaptive-pacing change; ten acquisition/batching tests subsequently passed after that change. Tests cover avoiding duplicate outbound queries, preserving the shared budget under concurrency, cache corruption rejection, refusing capped reconstruction, preserving missing/disputed events and increasing cooldown without spending a call.

The offline audit in `scripts/nextgen_archive_audit.py` verifies each cached response and original play source, then records per-game missing events, unmatched plays, missing athlete IDs, duplicate attributions and yardage/team/period disagreements. Its outputs are `attribution_coverage.parquet` and `attribution_audit.json` in the evidence directory. Final verification, unresolved-source dispositions and final quota readback are outstanding at the interim cutoff.

Code/runbook: repository `scripts/nextgen_archive_complete.py`, `nextgen_archive_batch.py`, `nextgen_archive_audit.py`, and `docs/nextgen_fingerprints/archive_completion.md`; initial implementation commit `de6dc3b`, adaptive changes through `8531903`.

See also: [Experiment](Next-Generation-Fingerprint-Experiment) · [F13 proposal](Fingerprint-F13-Proposal) · [Measured lessons](Nextgen-Empirical-Lessons-2026-09-28).
