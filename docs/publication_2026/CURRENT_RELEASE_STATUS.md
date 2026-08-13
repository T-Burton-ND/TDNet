# TDNet public-release status

Updated 2026-08-13 after the corrected-fingerprint and roster normalization.

## Canonical definitions

- F0–F6 are the nested market-free ladder.
- F7 is a market-only benchmark.
- F8 is F6 plus market information.
- The scientific roster contains every M architecture at every fingerprint:
  54 margin cells. Official predictions, consensus, and polls use only the 42
  market-free F0–F6 cells.
- The operational wide-margin roster uses corrected F6, selected by the
  prespecified margin-MAE criterion after corrected HPS.
- The prospective scope is FBS-versus-FBS regular season only.
- The immutable prediction deadline is Thursday 23:59 America/New_York.

## Verified locally

- The corrected-F6 wide bundle contains 34 learned estimators and two
  equal-weight ensembles, all trained through 2025 with 2026 excluded.
- Corrected fingerprint HPS completed 13,920/13,920 trials with no failed,
  missing, or duplicate rows; F6 improved margin MAE over F5 for all six
  scientific architecture types.
- The full F0–F8 scientific refit contains 54/54 checkpoint cells trained
  through 2025. Its market-bearing F7/F8 cells are comparative-only.
- TDNet-authored code uses Apache-2.0. CFBD rows, credentials, third-party
  marks, and separately distributed model artifacts are excluded.
- Public notebooks have saved outputs cleared. Model bytes remain ignored and
  will be released through a separate research-artifact repository.
- The repository-local `.env` is ignored, permission 0600, and successfully
  authenticates to CFBD.

## Current CFBD preseason availability

The 2026 API currently provides schedules, lines, pregame win probability,
Coaches Poll, returning production, coaches, recruiting-team context, and FBS
membership. It does not yet provide the AP Top 25 or 2026 talent. Postgame-only
team statistics, advanced statistics, havoc, and game PPA are correctly empty
before games are played.

`run_2026_preseason_release` is ready and fail-closed. It reports
`waiting_for_official_ap_top25` until an actual AP poll appears. Once available,
it generates TDNet preseason polls, AP comparisons, Week 1 predictions,
figures, failure reports, and a blog draft. The preseason-only state carries
2025 talent forward when 2026 talent remains unavailable and uses live 2026
returning production.

## Open before public release

1. Upload the two prepared, locally verified model archives to the public
   artifact host and verify fresh downloads against the published SHA-256 sums.
2. Review third-party team marks/logos included in curated figures.
3. The local full suite and release verifier pass; rerun the archive-based clean-checkout
   smoke pass on the normalized root commit.
4. Create the final annotated tag only after all release gates pass.

The requested four-way 2025 regeneration is complete and verified; see
`REGENERATION_2025.md` and `REGENERATION_2025_STATUS.json`. Every package has
17 poll weeks, 16 prediction weeks, and zero poll failures. Through-2025
outputs are pipeline dry runs only, while through-2024 outputs are the actual
2025 holdout evaluation.

Missing 2026 outcomes does not block a code-and-retrospective-figures release.
It does block describing the prospective study as completed.
