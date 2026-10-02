# Historical archive completion — 2026-09-28

The user explicitly approved completing useful historical data while avoiding
unnecessary repeat API calls, with the existing 20,000-attempt experiment cap.
This authorizes raw acquisition, not new model experiments or canonical rebuilds.
The provider's 10,000-call reserve and the 100 GiB storage guard remain in force.

Artifacts live under
`/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/results/preflight/archive_completion_20260928/`.
The attempt counter remains the original `results/cfbd_api_call_budget.json`;
it had 1,858 reservations before this work. It must never be reset.

## Scope and evidence

- The default acquisition manifest already has cached responses. Five weekly
  team-box partitions are still coverage-incomplete despite two earlier fetches.
  Instead of repeating them, this run queried their nine missing game IDs once.
- Ninety-five missing play-game IDs were investigated through documented
  team/year/week queries. All 95 play and nine box queries returned empty 200
  responses. Their ledgers retain `needs_review`; no zero observations or
  universal structural-absence claim was created.
- The main addition is player-play event attribution for 10,704 regular-season
  FBS-involved games in 2013–2025. Original samples are reused.
- Three 2012 probes gave one empty result and two two-row, single-field-goal
  results. The 805-game bulk 2012 expansion was excluded for low demonstrated
  incremental value. This is not a claim that all 2012 attribution is absent.
- New probes in the other unsampled years joined cleanly to play IDs and had
  athlete IDs. Existing source-yardage disagreements and incomplete event
  coverage remain limitations. Attribution is not full snap participation.

`plan.json` preserves the original game-level request plan. `reviewed_gate.json`
records the direct user authorization and sample decision, separately from the
earlier Stage E skip decision for adopting F10/F12 features. We do not fabricate
that older gate's feature-completeness conditions to authorize an archive.

## Avoiding redundant requests

Conference/week batches were compared against cached game responses: 22
conference-specific subsets across 17 games matched exactly. The provider's
conference filter restricts the attributed **team**, not both sides of a game.
Thus a reconstructed game requires complete partitions for both conferences.

The initial optimized plan has 1,790 conference/week requests, 1,680 individual
game requests for small groups/FCS opponents, and 87 verified existing game
caches. Capped or unavailable batches fall back to individual-game requests;
the first-attempt total is therefore not a final consumption claim.

Every response at the 2,000-row cap remains incomplete. It is never used to
create a complete game cache. Derived caches bind their actual source partition
hashes, record zero HTTP attempts for the derivation, and preserve prior attempts
if a previous request failed. They retain source duplicates/discrepancies.
Existing successful game caches are verified and never overwritten. Terminal
empty/failed queries are retained rather than automatically resent.

An initial four-per-second batch hit HTTP 429 and stopped. The revised executor
uses at most two concurrent requests, with batch attempts initially five
seconds apart and game requests initially 1.1 seconds apart. A 429 causes a
shared cooldown of at least 65 seconds, extended by a longer numeric Retry-After
header, and doubles the request interval up to sixty seconds. The initial provider
responses did not expose rate-limit limit/remaining headers.

## Commands and completion checks

The following commands are for an authorized continuation only; do not infer
permission to train models from this document:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /users/tburton2/.conda/envs/gridiron/bin/python \
  scripts/nextgen_archive_batch.py --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /users/tburton2/.conda/envs/gridiron/bin/python \
  scripts/nextgen_archive_audit.py
```

The executor holds an exclusive run lock. It uses the shared atomic budget on
every attempt, a fresh provider quota check, cache hashes, and storage guards.
`batch_status.json` records progress and terminal conditions; a finished
dispatcher does not imply all expected data exists. `batch_reconstruction.json`
separates reconstructed games from individual-game fallbacks.

The offline audit verifies attribution caches, game/year/week scope and original
play source hashes, then records per-game missing events, unmatched plays,
athlete-ID gaps, duplicate attributions and yardage/team/period disagreements.
It writes `attribution_coverage.parquet` and `attribution_audit.json` without
altering raw data, training inputs or predictions. Final completion requires
reading those results and explicitly reporting unresolved requests and the
final budget; file presence or HTTP 200 alone is insufficient.

`scripts/nextgen_archive_finish.py --expected-start <started_utc>` can wait on
the executor lock and perform that offline audit when acquisition exits. It
verifies game caches and their source-partition hashes, saves unresolved game
identities in `completion_receipt.json`, and makes one counted quota readback.
The receipt distinguishes complete query coverage, unresolved games and failed
verification; it never equates query coverage with complete participation.

## Reviewed gap recovery

The user subsequently requested best-effort completion of the remaining archive.
`scripts/nextgen_archive_recover.py` selects first-time direct game queries for
empty conference unions and one manually reviewed attempt for interrupted or
throttled queries. It does not repeat successful queries or known-empty direct
responses. The reviewed retry includes the previously exhausted throttled query;
its cumulative history is preserved, and automatic retries are disabled.

The first recovery pass selected 56 queries: both interrupted games returned data,
while all 54 previously unqueried direct game requests returned empty. Alongside
the 11 retained empty direct responses, 65 attribution gaps remain. Of these,
16 have ordinary play data and 49 also lack cached ordinary plays; all 49 already
had empty targeted play probes. There is no evidence that repeating those calls
would fill the gaps. Preserve absence and source discrepancies without imputation.

`recovery_pass_1/` retains the review, original audit/receipt, per-request results,
and `unresolved_games.csv` with exact IDs, teams, source checks and dispositions.
The original shared budget remains authoritative. A second dry run selects zero
requests after the pass; executing an existing pass again is refused.
