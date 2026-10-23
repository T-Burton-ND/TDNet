# F20 expert-consensus research design

## Architecture observed before implementation (2026-10-23)

TDNet's canonical F0–F8 ladder and six-model scientific panel are configured in
`configs/publication/confirmatory_protocol.yaml`. Later F9–F19 research is built
by scripts in `scripts/publication/` and `scripts/constraint_free_*`; F18 is a
market-free reduced representation and F19 adds archived market information.
The 120-cell F0–F19 historical search uses the same six architectures with
chronological folds, but its market source audit says historical quote times are
unverified. The 2026 weekly workflow writes frozen bundles under
`publication/2026/week_XX/pre_game/`; the Week 8 expanded research package is
separate. `configs/publication/data_source_manifest.yaml` specifies Thursday
23:59 America/New_York as the weekly cutoff. `.github/workflows/ci.yml` runs
tests only; the cron file is a template, not an installed schedule. Existing
prospective records must remain untouched.

## Study question and information levels

F20 asks whether point-in-time independent expert projections add information
beyond the market and TDNet. The comparison unit is a game with the same
cutoff, source coverage, train/test boundary, and outcome across arms.

| Arm | Inputs | Status |
| --- | --- | --- |
| Market-only | Captured current/open market columns, explicit missingness | Control |
| F20-A | Market-only plus individually archived expert home-margin projections and coverage/dispersion | Developmental 2017–2024 model frozen; on-demand research capture active |
| F20-B | F20-A plus up to 20 TDNet columns selected inside each training fold | Historical candidate fitted; no prospective feature path |
| F20-C | Complete F19 input matrix plus the same expert block | Historical candidate fitted; no prospective feature path |

The arms are comparisons, not a single nested ladder. F20-B's selector uses
up to 20 prospective-allowed TDNet columns and only training-fold labels.
The [matched A/B/C comparison](F20_COMPARISON.md) fitted private candidate
B/C models on 2017–2024 games, but no variant has been selected canonical.
F20-C requires a complete point-in-time F19 weekly matrix, unavailable for
future weeks; B likewise needs an audited weekly TDNet feature projection.
Do not use tracker
`Prediction Avg.`, `Prediction Median`, `Home Wins`, or `Computer Adj. Line` as
an independent expert: these aggregate or transform other rows. Sagarin's four
variants are one source family, not four independent votes. ESPN FPI, FEI,
Massey Ratings, Team Rankings, and Sagarin Pred. are candidate individual
columns. Their raw values are home projected margin, not probabilities.

## Point-in-time contract

Every capture stores exact response bytes in private local storage, source URL,
collection UTC time, displayed site update time if parseable, SHA-256, parser
version, game mapping, and source values. Outcome or postgame columns are never
copied into the prospective feature projection. A capture is eligible only if
collection occurred before issuance and before kickoff. A published timestamp
is evidence but does not prove that the underlying expert model avoided
retroactive updates. The prospective evidence is the captured bytes. A
historical CSV is `archived_pregame_unverified`, never `prospective`.

Week 8's official Thursday cutoff had passed before F20 work began. Week 8
observations captured on October 23 support only clearly marked later pregame
research revisions. All future F20 collection now runs on demand when the
owner invokes the approved weekly Tuesday pipeline. No recurring jobs remain
queued. F20 research issuance cannot create an official pointer before the
owner selects a canonical A/B/C variant and its weekly feature path is ready.
The canonical weekly bundle's owner-supplied cutoff, including earlier cutoffs
for early kickoffs, will govern any later F20 canonical integration.

The collector is separable from fitting. A failed source yields a failure
receipt and no synthetic observations. A rerun with the same response hash is
idempotent; a changed response becomes a new revision. Research forecasts have
distinct immutable issuance records; official designation is disabled pending
selection.

## Analysis plan and boundaries

Primary metric is game-level Brier score. Compare arms and F18/F19 on identical
game IDs; also report full-coverage scores separately. Use chronological folds
ending no later than 2024 for design and tuning, and report 2025 as already
consumed retrospective evidence. 2026 research already influenced the broader
fingerprint program, so F20 cannot claim an untouched 2026 confirmation.
Report uncertainty over paired game or week blocks and mark broad searches
exploratory. Do not claim an irreducible error estimate.

## Source and licensing boundaries

The Prediction Tracker's public current page and season CSV archive provide
many game-specific projections. Its public pages do not establish a license for
redistributing a raw archive, so raw snapshots stay outside public Git. CFBD
provides season schedules and market lines but says current-season ratings
change as games are played; a later historical response is not evidence of its
earlier value. Individual expert sites require further terms and timestamp
review before direct scraping. The current page is the first source integrated
for private capture, with attribution in the capture receipt.
