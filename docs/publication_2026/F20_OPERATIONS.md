# F20 2026 capture and forecast operations

## Current status, October 23, 2026

- Private live Prediction Tracker HTML was captured at 16:28:41 UTC; source
  page said it was last updated at 15:11:05 UTC. SHA-256:
  `8ef0867f67fd8995fa46cf33df46cd683ca13dbbdf4d1d906e1982cebe177df6`.
- Of 56 unique live source matchups, 55 mapped to the local CFBD 2026 schedule.
  Forty-nine had not kicked off at capture. The unmapped Appalachian State–
  James Madison row has no matching local schedule game and was not guessed.
- Forty-nine Week 8 F20-A and market-control probabilities were issued before
  their kickoffs. They are **late pregame revisions**, because Week 8's official
  Thursday cutoff had already passed. No F20 Week 8 official pointer exists.
- Frozen developmental models were fit on 5,417 matched 2017–2024 regular
  FBS-vs-FBS games. 2025 and 2026 outcomes were excluded. Historical tracker
  publication times are unverified. The model files and exact private inputs
  are in the owner-only persistent directory
  `/groups/bsavoie2/tburton2/TDNet/f20_2026/` (mode 700). The initial 49
  forecasts reproduce from that copy; see the verifier below.
- An immediate SGE compute-node smoke job (1617881) fetched a fresh live
  snapshot, issued 49 more append-only Week 8 late revisions in persistent
  storage, and wrote a successful run receipt. The archive verifier reproduced
  all 98 stored forecast records from archived inputs and the frozen models;
  official pointer count remained zero.
- The 2025 paired exploratory check covered 733 matched games. Market-only
  Brier was 0.171861; F20-A Brier was 0.171440. This small observed difference
  does not establish prospective improvement. The 2025 season was already
  consumed by earlier TDNet research.

## Weekly schedule and cutoff

The existing protocol defines the official forecast cutoff as Thursday 23:59
America/New_York. A game is eligible for an official pointer only if kickoff is
after that instant. Early-week games can receive separate pregame research
revisions but cannot be called official Thursday-cutoff forecasts.

The SGE submission script requests capture/issuance on Monday, Tuesday, and
Wednesday at 12:00 Eastern, then Thursday at 21:00 and 23:30 Eastern. The
second Thursday job is a retry and can record a new source revision; the first
valid official pointer remains fixed. These times avoid the daylight-saving
transition hour. The runner calculates cutoffs with the IANA
`America/New_York` zone; Week 9's cutoff is October 30 03:59 UTC and Week 10's
is November 6 04:59 UTC.

Scheduled weeks are 9–13 and 15. The current CFBD cache shows no Week 14
regular FBS-vs-FBS games. Any schedule change must be reflected by the Monday
CFBD refresh. An official forecast refuses a schedule cache more than five
days old. Capture still archives live HTML when schedule freshness is uncertain.
There is no current `crontab` access, and GitHub Actions only runs CI. The SGE
submission ledger and run logs live under the private `prospective/` archive.

## Commands

From the repository root, use the `gridiron` environment Python:

```bash
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/capture_f20_experts.py
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/fit_f20_baselines.py
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/report_f20_sources.py
/users/tburton2/.conda/envs/gridiron/bin/python -m pytest -q tests/test_f20.py
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/verify_f20_archive.py \
  --archive /groups/bsavoie2/tburton2/TDNet/f20_2026/prospective \
  --models-root /groups/bsavoie2/tburton2/TDNet/f20_2026/models
```

The `fit_f20_baselines.py` output is immutable and intentionally refuses to
overwrite the frozen model directory. Do not rerun it into the existing path.
The historical CSVs are private; `f20_source_inventory.json` summarizes their
season/week and source-column coverage without redistributing raw rows.

To inspect a run, read `prospective/runs/*.json` and
`prospective/scheduler_logs/` in the private archive. A run report has separate
capture and issuance statuses, a count of expected official games, and missing
game IDs. A failed fetch records its own `prospective/failures/*.json`. Source
collection retries three times with backoff. Repeating the same raw/schedule
capture or same source/model forecast ID preserves the original bytes. A new
source response or explicit revision ID yields a new append-only record.

## Source status and remaining setup

The [Prediction Tracker live page](https://www.thepredictiontracker.com/predncaa.php)
is the integrated source for individual ESPN FPI, FEI, Massey Ratings, Team
Rankings, and Sagarin Pred. projected margins. These are *tracker-reported*
values, not independently verified publisher snapshots. Its
[season archive](https://www.thepredictiontracker.com/ncaaarchive.html)
supports historical development but lacks original publication timestamps.
Raw source redistribution rights have not been established, so the archive is
private. [CFBD's availability guide](https://api.collegefootballdata.com/data-availability)
notes that current SP ratings change as games are played; direct FPI/SP+
historical downloads are not accepted as point-in-time F20 observations.

The current shell has no `CFBD_API_KEY`. To keep weekly schedule times fresh,
the existing Monday refresh needs its normal credential and owner review. If
it does not run, later official F20 issuance stops on the five-day cache guard
while raw expert capture continues. The SGE runner also needs outbound access
to the public tracker on compute nodes; the first scheduled job must confirm
this. F20-B and F20-C remain untrained: the full prospective F19 representation
has a Week 8 research implementation but no reusable Week 9+ weekly artifact.
Their retrospective work must retain F19's historical market quote-time caveat.

No F20 probability has been added to the canonical operational 33-model
consensus, poll, contest submission, or public package. F20 forecasts are
registered in the private weekly run and forecast archive. Public release
requires the existing publication review and source-rights decision.
