# F20 2026 capture and forecast operations

## Current scientific F20 panel

The owner's selection of the A expert-consensus input block is now named
**F20**. Its probability panel uses M1, M2, M3, M4, M5, and M10, with the
equal mean of their six home-win probabilities as F20 consensus. All six were
trained and frozen on 5,417 matched 2017–2024 regular FBS games. Rolling
2021–2024 Brier scores on 2,847 matched games are in the frozen private
`models/f20_scientific_2017_2024_v1/manifest.json`; the pooled consensus score
is 0.176629. The archived market-only baseline scored 0.175110 on those
same 2,847 games. The earlier A/B/C comparison used a separate 2,216-game
cohort; its scores are evidence for the input choice, not scores of this panel.

The new `models/f20_scientific_2017_2024_v2_margin/` keeps those six probability
models byte-for-byte and adds one frozen home-margin regressor for each M1–M5
and M10 architecture. It uses the same expert/market feature block and 5,417
historical games; no 2025 or 2026 target scores were used for fitting. The
rolling 2021–2024, 2,847-game six-margin consensus MAE is **12.329956 points**.
This model was trained after the 2026 games graded below, so its 2026 score is
retrospective. The on-demand weekly path now uses this v2 panel and issues
both probability and predicted-margin fields, with separate hashes.

The Tuesday pipeline invokes F20 only on demand, using its frozen bundle's
exact game slate and owner-selected cutoff. Official issuance requires a fresh
capture, a complete slate, kickoffs after cutoff, and a frozen model hash. A
research run cannot create official pointers. Prior 147 Week 8 research
records remain historical F20-A experiments and are not renamed or promoted.
After complete official issuance, Tuesday materializes the six F20 probability
and margin cells and their equal-mean consensus in
`pre_game/scientific/expanded_f0_f20/f20_component/`, bound to the exact
frozen slate and official forecast record hashes.

The 2026 tracker CSV supports a **retrospective** F20 replay on CFBD Weeks
1–6. Its numeric tracker week is offset from CFBD week, so games are joined by
team, final score, and CFBD game ID. Original weekly publication timestamps
are absent; the owner accepts the weekly projections as pregame for these
charts. The six models were fitted after those games. CFBD Week 7 has no F20
archive inputs. See [the cumulative chart readme](../../publication/2026/research/scientific_f0_f20_cumulative_week01_07_v3_margin/README.md)
and [all-cell post-hoc scorecards](../../publication/2026/research/scientific_f0_f20_posthoc_scorecards_week01_07/README.md).
The [individual-source Week 7 audit](F20_WEEK07_SOURCE_AUDIT.md) records the
FEI, Massey, and TeamRankings pages found after searching outside the stale
tracker CSV, and the still-missing complete FPI/Sagarin projected margins.

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
  snapshot and issued 49 more append-only Week 8 late revisions. A subsequent
  end-to-end schedule-refresh smoke at 16:58 UTC captured 56 CFBD FBS games,
  refreshed the expert page, and issued 49 further late revisions. The archive
  verifier reproduced all 147 stored forecast records from archived inputs and
  the frozen models; official pointer count remained zero.
- A CFBD-only SGE compute-node smoke job (1617954) captured the Week 8 schedule
  at 17:02 UTC: 56 FBS-vs-FBS games, zero TBD kickoffs, and the same raw SHA as
  the local refresh. This verifies compute-node API access; the combined
  schedule-plus-expert Week 9 job has not yet run.
- The 2025 paired exploratory check covered 733 matched games. Market-only
  Brier was 0.171861; F20-A Brier was 0.171440. This small observed difference
  does not establish prospective improvement. The 2025 season was already
  consumed by earlier TDNet research.
- On the owner's October 23 correction, all 30 previously queued F20 SGE
  future-week jobs (1617888–1617918, excluding 1617892) were canceled. A
  `qstat` check returned zero matching F20 future-week jobs. The private
  `prospective/scheduler_logs/cancellation_20261023.json` preserves the IDs.
- [The matched A/B/C comparison](F20_COMPARISON.md) informed the owner's
  selection of the expert-consensus inputs now used by F20.
- A fresh on-demand **six-model F20** research capture at 18:11 UTC used the
  Week 8 page and refreshed CFBD schedule. All 49 games still pregame at
  capture received M1–M5, M10, and equal-mean F20 probabilities; none received
  an official pointer because the Week 8 cutoff had passed. Its private run
  receipt is `prospective/runs/2026_w08_six_model_week08_capture_20261023T181112743109Z.json`.
  The archive verifier reproduced all 196 records: 147 legacy A/control and
  49 six-model F20, with zero official game pointers. Week 9 and later await
  their on-demand weekly pipeline run; no future jobs were queued.

## On-demand weekly operation

The approved `run_tuesday_publish.py` command now invokes
`run_f20_weekly.py --phase canonical` **during that on-demand weekly run**, after
the canonical bundle and X draft are prepared. There is no F20 cron, SGE
future queue, or automatic recurrence. The F20 component refreshes that week's
CFBD FBS schedule with the existing ignored `.env` credential, archives the
raw response privately, captures the live expert page, and issues the frozen
six F20 scientific probabilities, six margins, and their equal means for games still pregame. A
failed source or schedule request is recorded separately. A failed F20 run
marks the Tuesday command incomplete after the existing canonical outputs
have been written; it does not rewrite those outputs. Inspect
`data/publication/2026/weekly_operations/week_XX/f20_canonical_run.json` and the
private F20 run receipt.

The same Tuesday command writes a [three-roster registry](THREE_ROSTERS.md)
under that week's `data/publication/2026/weekly_operations/` directory. The
original frozen scientific roster is the 42-cell F0–F6 cohort. F0–F8 is now
the base component of the single expanded F0–F20 research roster, not a fourth
roster. The registry identifies the F9–F19 cells that still lack a reusable
prospective weekly path and keeps F20's own margin panel separate from the
incomplete expanded-roster margin consensus.

The research phase **cannot create official F20 pointers**. Canonical issuance
uses the same owner-supplied weekly cutoff as the main publication pipeline,
including an earlier cutoff when a game starts before Thursday.
Week 8's 147 research records stay unchanged. The current CFBD cache showed
no Week 14 regular FBS-vs-FBS games on October 23, but a future on-demand run
will request its actual week directly. The old SGE submission script was
removed so it cannot requeue the canceled jobs.

## Commands

From the repository root, use the `gridiron` environment Python:

```bash
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/refresh_f20_schedule.py --week 9
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/run_f20_weekly.py \
  --week 9 --phase research \
  --archive /groups/bsavoie2/tburton2/TDNet/f20_2026/prospective \
  --models /groups/bsavoie2/tburton2/TDNet/f20_2026/models/f20_scientific_2017_2024_v2_margin
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/fit_f20_scientific_margins.py
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/fit_f20_scientific.py
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/fit_f20_baselines.py
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/compare_f20_abc.py \
  --c-grid 0.0001,0.0003,0.001,0.003,0.01,0.1,1,10,100 \
  --output data/private/f20/abc_comparison_new \
  --summary docs/publication_2026/f20_abc_comparison_new.json
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/report_f20_sources.py
/users/tburton2/.conda/envs/gridiron/bin/python -m pytest -q tests/test_f20.py
/users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/verify_f20_archive.py \
  --archive /groups/bsavoie2/tburton2/TDNet/f20_2026/prospective \
  --models-root /groups/bsavoie2/tburton2/TDNet/f20_2026/models
```

The fit outputs are immutable and intentionally refuse to overwrite frozen
model directories. Choose a new versioned output to refit them.
The historical CSVs are private; `f20_source_inventory.json` summarizes their
season/week and source-column coverage without redistributing raw rows.
The normal approved Tuesday command remains the entry point for the full
weekly pipeline; its F20 archive and model paths are configurable with
`--f20-archive` and `--f20-models`.

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

The repo's ignored `.env` credential worked for the narrow F20 Week 8 CFBD
schedule request. The separate operational Monday refresh and owner review
still govern the canonical TDNet publication; F20 does not replace them. The
SGE smoke jobs confirmed tracker and CFBD access from compute nodes before
the future jobs were canceled. F20-B and F20-C have developmental candidate
fits only: the full prospective F19 representation has a Week 8 research
implementation but no reusable future-week artifact. Their retrospective
comparison retains F19's historical market quote-time caveat.

No F20 probability has been added to the canonical operational 33-model
consensus, poll, contest submission, or public package. F20 forecasts are
registered in the private weekly run and forecast archive. Public release
requires the existing publication review and source-rights decision.
