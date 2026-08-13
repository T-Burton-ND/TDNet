# 2026 preseason rehearsal data policy

## Status

Outputs under the preseason rehearsal directory are **non-public previews**.
They are not frozen predictions, prospective evidence, or release artifacts.

## Temporary data substitutions

- The CFBD Coaches Poll is used as the ranked-team reference until the AP Top
  25 becomes available. Every table, figure, and manifest must label it
  `Coaches Poll (AP surrogate; rehearsal only)`.
- If CFBD returns no 2026 talent table, each team's latest 2025 `roster_talent`
  value is carried into its Week-0 state.
- If CFBD returns no 2026 returning-production table, each team's latest 2025
  returning-production values are carried into its Week-0 state.
- These carry-forwards are derived-state assumptions. They must never be
  written into the raw 2026 CFBD cache or represented as provider observations.

## Rosters

The rehearsal is rendered twice:

1. `scientific`: the 54-model F0–F8 research bundle. Rehearsal and release
   predictions/polls use only its 42 market-free F0–F6 cells.
2. `margin_wide`: the corrected-F6 bundle with 34 learned estimators and two
   equal-weight ensembles. Three statistical models are prediction-only after
   failing the poll-ordering sanity check.

## Next CFBD check

After the rehearsal refresh, subsequent API calls must be limited to:

- `rankings`
- `talent`
- `returning`

Use `configs/fetch/preseason_2026_watch.yaml`. Expand that endpoint set only if
the schedule changes materially or a separate weekly refresh is explicitly
requested.
