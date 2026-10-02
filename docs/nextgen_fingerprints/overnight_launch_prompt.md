# TDNet Next-Generation Fingerprint Experiment — Overnight Launch Prompt

You are launching the full TDNet next-generation fingerprint experiment.

This prompt assumes the repository preparation and hardening work on:

`experiment/nextgen-fingerprints-v1`

has been completed and pushed.

Expected recent launch-ready head is approximately:

`cc3038f1bb293e127e69d18d00647bf6d37b1e5a`

Do not rely only on that SHA: fetch/pull the branch and inspect the actual current head before starting.

This is now an **execution prompt**, not another planning pass.

Your job is to carry the experiment through as far as possible tonight, autonomously and reproducibly, while preserving the scientific and operational contracts already encoded in the repository.

---

# 0. Primary operating rule

**Figure things out and keep going.**

Do not repeatedly stop to ask the user questions.

Use:

1. the current experiment configs,
2. existing TDNet code,
3. current TDNet caches/results,
4. the `.llm-wiki/` project knowledge,
5. current CFBD API/OpenAPI documentation,
6. reputable football analytics literature,
7. standard ML/statistical engineering practice,

to resolve implementation details.

If a newly encountered problem does **not** threaten:

- data leakage,
- scientific validity,
- lineage/result corruption,
- the 20,000-call API ceiling,
- the 100 GB storage budget,
- or recoverability/reproducibility,

then **document it and continue** rather than stopping to redesign the experiment.

Only stop the overall experiment for a genuine blocker in one of those categories.

If one branch/generation/feature family fails but the rest can proceed safely, mark it incomplete and continue.

---

# 1. Git / branch / provenance

Work on:

`experiment/nextgen-fingerprints-v1`

Start by:

- fetching/pulling the latest branch,
- confirming the working tree is clean,
- recording the starting commit SHA,
- validating the repository setup,
- confirming the `.env` CFBD key works without printing or committing it.

Commit and push meaningful code/config/wiki progress freely to the experiment branch throughout the run.

Do not rewrite the frozen historical 2026 publication protocol or production artifacts.

The new experiment is exploratory next-generation research.

Update `.llm-wiki/` as durable knowledge as generations are finalized and results are obtained.

Follow the existing wiki schema and log-commit conventions.

---

# 2. Scientific invariant

The experiment predicts the **next game**.

Every dynamic team-week feature must describe information available before the target game and must be evaluated for whether it helps predict the **next game's margin**.

Do not confuse:

- same-game explanation,
- same-game correlation,
- or game-state consequence

with future predictive information.

Same-game plots may be shown descriptively, but feature design and predictive diagnostics must use pre-target-game state against future outcomes.

Primary prediction target:

`next_game_margin`

Useful secondary future targets include:

- `next_game_win`
- empirical next-game win percentage / win frequency
- `next_game_points_for`
- `next_game_points_against`
- next-game time of possession when semantically appropriate

---

# 3. Time boundary

Use:

- 2010–2023: base historical development
- 2024: design-informed internal validation
- 2025: design-informed late development
- 2026: untouched prospective season

All information through 2025 may be used for this next-generation design program.

Do not describe 2024 or 2025 as unbiased holdouts.

## Absolute 2026 quarantine

No 2026 row may affect:

- feature discovery,
- formulas,
- composite fitting,
- scaling,
- imputation,
- missingness rules,
- correlation,
- redundancy,
- SHAP,
- hyperparameter selection,
- reduction,
- lineage selection,
- recommendation,
- model evaluation used to choose the new fingerprints.

2026 may be cached separately only.

---

# 4. Target population / postseason

Prediction targets:

- FBS vs FBS
- regular season only

FCS games may contribute prior context to an FBS team's state.

Postseason numeric performance must not feed later predictive state.

Use regular-season filtering as early as reasonably possible in acquisition/canonicalization, not merely at the final model stage.

Pre-2010 acquisition is allowed only when targeted coach/player/program-history initialization genuinely requires it.

---

# 5. Fingerprint ladder

Preserve historical meanings:

- F00–F06 = historical market-free information ladder
- F07 = market-only comparator
- F08 = F06 + market comparator

The new market-free lineage intentionally skips F07/F08:

`F06 → F09 → F10 → F11 → F12`

F07/F08 are not ancestors of F09+.

## Naming

F06:

- `F06_F_a`
- `F06_F_b`
- `F06_F_c`
- `F06_R_a`
- `F06_R_b`
- `F06_R_c`

F09–F12:

- `_F_` = full lineage
- `_LR_` = late reduction of the current full representation
- `_PR_` = progressive reduction inheriting prior PR

Example:

`F10_PR_b`

Design tracks never crossbreed.

`a` inherits only `a`, etc.

Old F6-C / F6-C25 work is reference evidence only and must not seed the new selections.

---

# 6. Design philosophies

## a — atomic/canonical

F06_F_a must reproduce the canonical 227-source-feature F6 exactly.

Later `a` generations:

- straightforward structured quantities,
- one sensible canonical representation per concept,
- avoid gratuitous temporal-window explosions,
- avoid duplicates of existing earlier information.

## b — football-informed expansion

Use `a` information plus interpretable football-motivated interactions/variants.

Actively research reproducible football analytics definitions.

Potential concepts include but are not limited to:

- success rate
- explosiveness / IsoPPP-style concepts
- scoring opportunities
- points per opportunity
- starting field position
- havoc
- opportunity rate
- stuff rate
- power success
- line yards
- second-level yards
- open-field yards
- early-down efficiency
- quality-drive rate
- net points per drive
- situation-specific conversion/efficiency

Do not blindly duplicate A.

If a B feature replaces a nearly redundant A feature more cleanly, retain the better representative.

## c — aggressive interpretable composition

C should aggressively compress raw information into football concepts.

Rules:

- maximum 5 source inputs per composite
- preferably fewer
- exact Excel-readable equation
- no opaque mini-models
- no ten-term tiny-weight soups
- fitted weights are allowed using pre-2026 evidence
- arbitrary transparent mathematical functions are allowed

Examples:

- weighted means
- sums/differences
- ratios
- protected ratios
- products
- geometric/harmonic means
- slopes
- trends
- deltas
- variance/CV
- concentration metrics
- entropy/Gini/Herfindahl
- log/log1p
- bounded transforms
- simple piecewise functions
- simple nonlinear formulas

Interpretability heuristic:

> If you showed me the football concept happening on film, could I understand what this feature is trying to represent?

This is a heuristic, not a hard statistical rule.

Every C feature must receive a clear wiki/manifest explanation that would make a skeptical reader understand why the formula is football-reasonable.

---

# 7. F06

F06 is the current canonical market-free baseline.

Build/evaluate:

- `F06_F_a`: exact canonical current F6
- `F06_F_b`: same information plus sensible interpretable interactions/re-expressions
- `F06_F_c`: aggressive interpretable compression/re-expression
- `F06_R_a/b/c`: independently reduced versions

Do not use old F6-C selected features as the new seed.

---

# 8. F09 — game microstructure

Use structured numeric play/drive/game-state information.

No free-text play-description parsing.

Explore:

- whole-game
- quarters
- halves
- middle eight
- drives
- play microstructure
- field position
- pace
- scoring opportunities
- red zone
- goal-to-go
- third/fourth down
- one-score situations
- backed-up situations
- two-minute situations
- other simple objective football states

Middle eight:

- final 4:00 of Q2
- first 4:00 of Q3

Do not define leverage using win probability.

## Required rushing feature

Implement and evaluate:

`offense_rush_ypa_q4_minus_q1`

and its defensive counterpart.

Meaning:

Q4 rushing yards/attempt minus Q1 rushing yards/attempt, based only on prior qualifying games.

Also investigate:

- quarter-to-quarter rushing slope
- other compact "gets stronger/weaker as game progresses" rushing concepts

Build polished predictive diagnostics for this concept later, including:

- pregame value vs next-game margin
- pregame value vs empirical next-game win frequency

---

# 9. Garbage time

Starting thresholds:

- Q1: absolute lead > 28
- Q2: > 24
- Q3: > 21

For Q4:

Treat the whole fourth quarter as garbage only if the absolute lead remains >16 on **every qualifying fourth-quarter play**.

If it reaches 16 or less at any Q4 play, the quarter is not globally marked garbage under this rule.

Garbage-time status overrides ordinary time/high-leverage flags.

Red-zone/location concepts may remain separately meaningful.

B may include clearly labeled all-play alternatives where analytically useful.

Do not use win probability.

---

# 10. F10 — roster / recruiting / transfers / player production

Add structured player-level information:

- roster
- recruiting
- transfer portal
- actual usage
- prior production
- returning production
- identity continuity
- returning QB production/share
- returning rushing/receiving/defensive production
- position-group talent
- transfer additions/losses by position
- weighted experience
- usage concentration
- roster churn
- structured physical summaries where useful

Forget the prior "two deep" concept.

Prefer actual usage/participation.

Hierarchy:

1. actual usage/playing time
2. prior production
3. structured depth information only if reliable
4. recruiting/talent as fallback/context

Week-0 roster is frozen for the season.

Freshmen/new transfers may contribute recruiting/transfer/roster information but must not be assigned fictional prior production.

## Player matching

Priority:

1. stable IDs
2. exact normalized identity
3. conservative fuzzy match using team/year/position constraints and high threshold

Leave uncertain cases unmatched.

Report matching rates.

## Physical summaries

Use structured CFBD data only.

Potential examples:

- OL average weight
- DL average weight
- WR/DB height
- other position-appropriate physical summaries

Do not scrape external combine data.

## Injuries/availability

Do not infer "injured" from nonparticipation.

If a trustworthy structured field explicitly says available/unavailable, it may be used.

Otherwise record actual usage/participation only.

No news scraping.

---

# 11. F11 — coaching

Add genuinely new derived coaching information beyond F06/F04.

Do not use literal coach identity as a learned categorical memorization feature.

Explore structured:

- tenure
- year at school
- change flag
- interim status
- career record/history
- prior-team offense/defense performance
- trajectory
- recruiting history
- other reproducible historical coaching characteristics

If coordinator/sub-HC information is unreliable or unavailable, do not fabricate it.

F11 remains a generation even if thin.

A weak information gain is a valid result.

---

# 12. F12 — team unit states

Construct team-week unit-level states such as:

- OL pass protection
- OL run blocking
- QB room
- RB room
- WR/TE/pass-catcher room
- defensive line/front
- linebackers
- secondary
- special teams

A = relatively atomic.

B = richer football interactions.

C = compact interpretable unit composites.

The team-week fingerprint itself remains opponent-independent.

The matchup builder performs offense-vs-opposing-defense comparisons.

Examples:

- OL ↔ opposing DL
- pass catchers ↔ opposing secondary
- run game ↔ opposing front
- etc.

Special teams is required.

---

# 13. Matchup and temporal artifact boundaries

Use the hardened canonical nextgen artifact system already in the branch.

Every F09–F12 builder must use `NextgenFeatureBuilder` / checked canonical materialization.

Do not write bypass feature Parquets.

Canonical rows require explicit:

- `team`
- `target_game_id`
- `target_start_utc`
- feature kind
- feature availability
- source-game provenance for dynamic features

Matchup identity is:

`(target_game_id, team)`

not pandas row index.

The authoritative Stage-A fresh schedule must verify target and source game identity.

## Dynamic rows

Require authoritative confirmation that:

- team participates in target game
- target kickoff matches schedule
- team participates in latest contributing/source game
- source game is regular season and completed
- source precedes target
- recorded source timestamp matches authoritative schedule semantics
- `source_time < feature_available_time < target_time`

Builders must document realistic data-availability cutoffs; schedule completion alone does not prove instantaneous data availability.

## Static Week-0 rows

Require:

- null dynamic source-game provenance
- documented pre-season availability
- availability before the durable team-season Week-0 cutoff
- frozen values throughout the season

Use the existing durable Week-0 freeze artifact logic.

## Manifest provenance

Canonical artifacts must remain bound to:

- manifest hash
- data hash
- authoritative schedule hash

Reject manifest-declared:

- market-derived
- target-derived
- pregame-win-probability-derived
- disallowed temporal semantics

even if the feature name looks harmless.

---

# 14. Matchup pairs

Every team-week feature must declare a reviewed counterpart.

Examples:

- offense ↔ corresponding defense
- unit ↔ corresponding opposing unit
- global/static self-pair where appropriate

Pair pruning is atomic:

- both members must qualify
- otherwise the pair stays

C-created features also need declared counterpart and matchup formula.

The final fingerprint must remain universal across architectures.

---

# 15. Acquisition — Stage A first

Before any other acquisition:

Run Stage A and fetch fresh `/games` for 2010–2025.

These 16 ledger-backed requests become the authoritative schedule.

Then rerun the preflight.

Do not trust legacy schedule completeness as final authority.

Only after fresh schedule authority is established may legacy `/games/teams` partitions be reused when they cover every expected game ID for the relevant partition.

If a cache fails coverage/hash/scope checks, refetch rather than assuming completeness.

---

# 16. CFBD API budget

Account allowance:

`30,000 calls`

Experiment hard ceiling:

`20,000 actual outbound attempts`

Reserve:

`10,000 calls`

This includes:

- normal requests
- retries
- live quota checks

Use the atomic shared call budget ledger.

Never bypass it.

Before each stage:

- check provider quota,
- check local reserved count,
- check remaining stage requests,
- preserve the 10k provider reserve.

If reaching the 20k experiment limit would be required, stop acquisition and continue the experiment using the safely acquired data where possible.

Do not silently raise the cap.

---

# 17. Acquisition stages

Use the staged acquisition tooling already prepared.

## Stage A
Fresh authoritative `/games` schedules 2010–2025.

Then rerun preflight.

## Stage B
Cheap/broad metadata and season/year-level sources.

## Stage C
Play-by-play, drives, other manageable game microstructure.

## Stage D
Player detail / usage / game/season production sources.

## Stage E
`/plays/stats` only after its explicit gate passes.

Resume from verified ledgers after interruption.

Do not refetch successful requests.

Empty 200s or suspicious 400/401/403/404 responses are `needs_review`, not automatically "no data."

Response caps must never be silently accepted as complete.

---

# 18. `/plays/stats` Stage E

Do **not** automatically perform the full historical `/plays/stats` pull.

First sample and audit it.

Confirm:

- stat meanings
- player attribution semantics
- coverage
- 2,000-row cap behavior
- whether concrete F10/F12 features are uniquely enabled

Absence of a PlayStat row does not prove nonparticipation.

After the audit, autonomously choose:

- full
- subset
- skip

based on:

- unique incremental feature value
- remaining quota
- projected retries/subdivisions
- storage
- redundancy with cheaper endpoints

Document the decision.

Do not ask the user unless the decision would threaten the hard experiment constraints.

---

# 19. Storage

Large data/artifacts live under:

`/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/`

Soft experiment budget:

`100 GB`

Use:

- compressed Parquet
- canonical supersets
- lightweight manifests
- deduplicated representations
- temporary scratch cleanup

Do not store large data in `$HOME`.

Clean:

- temporary API payloads when safe
- temporary SHAP matrices after aggregation
- routine scheduler logs after extracting useful status
- model checkpoints after compact diagnostics/results are persisted

Check free space before expensive stages.

---

# 20. Feature eligibility

Avoid ultra-rare arbitrary features.

General gate:

a candidate should usually have roughly >=100 qualifying observations league-wide in a typical season, adjusted sensibly for its intended granularity.

Metric-specific thresholds are allowed and should be documented.

Examples:

Reasonable:
- interception rate
- sack rate
- Q4 sack rate if sufficiently supported

Likely too sparse:
- Q4 3rd-and-long sacks only
- games with >=3 interceptions

Do not preserve noise just to hit a feature count.

---

# 21. Temporal aggregation

Do not automatically create every rolling form of every new statistic.

A:
- one sensible canonical aggregation

B:
- additional temporal representations only when justified

C:
- aggressively collapse temporal behavior

Preseason/static information should be computed once per season where possible.

---

# 22. Correlation / redundancy

No PCA.

Use correlation/redundancy checks.

Rules:

- `|r| >= 0.995`: automatic redundancy candidate
- `0.98 <= |r| < 0.995`: validate representative
- `0.90 <= |r| < 0.98`: report/group, do not automatically remove

Use Pearson and Spearman as appropriate.

Near-duplicate preference:

1. lower missingness
2. simpler equation
3. stronger joint M2/M4 importance
4. earlier-generation feature as final tie-break

Pair closure remains mandatory.

---

# 23. Screening models

Use only:

- M2 spline ridge
- M4 histogram gradient boosting

for tonight's fingerprint screening.

Use the already frozen 10 representative configurations per model.

Same 10 M2 configs for all fingerprints.

Same 10 M4 configs for all fingerprints.

One fixed seed.

Do not silently optimize a new HP grid based on F09+ results.

---

# 24. SHAP

Use common end-to-end permutation SHAP comparable across M2 and M4.

Per relevant cell:

- ~256 background games
- ~512 explanation games

Run across all 10 M2 configs and all 10 M4 configs used for screening.

Aggregate home/away attribution back to team-week source features.

Normalize absolute source importance within architecture/run so M2 and M4 remain comparable.

A/B pruning candidate:

weak under both architectures, subject to redundancy and pair rules.

C may use joint average M2/M4 tradeoff more aggressively.

---

# 25. Reduction

Target smallest representation within approximately:

`+0.25 MAE points`

of its relevant full reference during reduction.

## Floors

F06 reduced:

- at least 60 concrete F06 features

Then add at least 10 concrete surviving features from each new generation:

- F09: >=60 F06 + >=10 F09
- F10: + >=10 F10
- F11: + >=10 F11
- F12: + >=10 F12

These are floors, not caps.

If a generation has fewer than 50 survivors, aim to preserve at least 10 pairwise-distinct signals with `|r| < 0.90`.

If genuinely fewer than ten legitimate independent signals exist:

- keep as many as exist,
- document the shortfall,
- do not retain junk just to satisfy the number.

---

# 26. A/B vs C reduction behavior

A/B are conservative.

Feature/pair removal must remain within +0.25 MAE for both M2 and M4 individually.

If M2/M4 disagree but both remain within +0.25, pruning is allowed.

C is intentionally more aggressive.

For C, use the average M2/M4 median 2024/2025 development performance tradeoff, while still respecting floors and interpretability.

---

# 27. F / LR / PR behavior

At each generation:

## F
Full accumulated representation.

## LR
Prune the current generation's full accumulated representation.

LR does not inherit prior LR.

## PR
Previous generation's PR + full new current generation family, then prune again.

F09 PR begins from F06_R.

No cross-design ancestry.

---

# 28. Generation training barrier

Train only one generation at a time.

While generation N is training, it is fine to:

- research,
- code,
- acquire,
- materialize,
- prepare

generation N+1.

But do not begin N+1 model training until N's:

- full training,
- SHAP,
- reduction,
- lineage comparison,
- recommendation

is finalized.

Proceed:

F06 → F09 → F10 → F11 → F12

---

# 29. Scheduler / compute

Use UGE/SGE arrays and CPU aggressively.

Hard experiment-wide concurrency cap:

`50 simultaneously running jobs`

This includes arrays and all experiment jobs.

Queue as much as useful, but do not exceed 50 running.

Use array/chunk patterns already established in TDNet.

---

# 30. Failure behavior

For each 10-setpoint architecture/fingerprint cell:

- if >=3 runs succeed, proceed and flag incomplete coverage
- do not retry merely to reach 10/10
- if <3 succeed, retry failed jobs
- maximum 3 retry attempts
- after that, record incomplete and continue where scientifically possible

Do not let one fragile job kill the whole overnight run.

---

# 31. Metrics

Primary:

- margin MAE

Also retain where available:

- RMSE
- Brier
- winner accuracy
- ATS accuracy
- chalk accuracy
- upset accuracy

Market metrics come only from the evaluation sidecar.

Never feed market data into F06/F09–F15 predictors.

---

# 32. Recommendation

For every generation × design, choose the recommended lineage.

Examples:

- F09/a: F vs LR vs PR
- F09/b: F vs LR vs PR
- F09/c: F vs LR vs PR

For F06 choose F vs R.

Use all pre-2026 development evidence.

Report 2024 and 2025 separately.

Treat candidates within:

`0.5 MAE points`

of the best as practically tied.

Tie-break:

1. Brier
2. ATS accuracy
3. upset accuracy
4. chalk accuracy
5. fewer features

Preserve all underlying results.

---

# 33. Results table

Maintain one ultra-wide Parquet with one row per run plus summary/recommendation rows.

Include explicit sortable columns for 2024 and 2025:

- MAE
- RMSE
- Brier
- winner accuracy
- ATS
- chalk
- upset

Also include:

- fingerprint ID
- generation
- design
- lineage
- ancestry
- feature counts overall/by generation
- model
- setpoint
- seed
- status
- runtime
- manifest hashes
- correlation summary
- SHAP references
- missingness/coverage summaries
- recommendation eligibility/selection/rank
- failure reason
- max design year
- acquisition provenance

Do not rely solely on JSON blobs for the primary sortable metrics.

No persistent screening model artifacts are required.

---

# 34. Advanced-feature diagnostics

Across the complete program, generate diagnostics for up to the **1,000 highest-impact advanced features by consensus SHAP**.

This is 1,000 total, not per generation.

Include:

- literature-derived
- TDNet-derived
- other advanced original features

Do not waste the cap on trivial atomic fields.

Default diagnostic:

pregame feature value vs `next_game_margin`

Where useful also plot:

- next-game empirical win percentage
- next-game points for
- next-game points allowed
- next-game time of possession
- another semantically appropriate future outcome

Correlation is information/explanation only.

Do not use it as the feature-selection rule.

Spearman is a useful default association summary; Pearson where appropriate.

Curved relationships should be described rather than penalized.

Store only:

- PNG(s)
- compact Markdown explanation
- lightweight index

Do not create giant standalone diagnostic tables.

Keep all diagnostics together under the experiment's `feature_diagnostics/` directory.

---

# 35. Diagnostic Markdown

For each plotted advanced feature include:

- feature name
- generation
- exact equation
- inputs
- units
- positive/negative meaning
- football interpretation
- film intuition
- source/inspiration
- coverage/sample rule
- garbage-time behavior
- predictive future target plotted
- Pearson/Spearman where relevant
- SHAP importance summary
- survival/pruning status
- matchup counterpart

---

# 36. Feature manifest

Every feature must have durable metadata including:

- name
- generation
- designs
- source endpoints
- raw columns
- source inputs
- exact Excel-readable formula
- units
- interpretation
- directionality
- availability rule
- temporal cutoff
- aggregation
- minimum sample
- missingness
- static/dynamic
- opponent adjustment
- market-derived flag
- target-derived flag
- pregame-WP-derived flag
- garbage-time handling
- matchup counterpart
- matchup formula
- provenance
- code path
- version

Literature-defined features should record source/inspiration.

Novel features should be explicitly labeled TDNet-derived.

---

# 37. Missingness

Use availability-aware missingness.

Distinguish:

- structural historical unavailability
- ordinary missingness within an available era

Do not encode pre-portal years as zero transfers.

Use explicit indicators where useful.

No KNN imputation.

Use simple training-safe imputation where M2 requires it.

Use M4 native missing support when appropriate.

---

# 38. Power-ranking average team

For each fingerprint, freeze the average-team reference once per season.

For 2026:

use all FBS team-week states for that exact fingerprint through the completed 2025 season.

Weight according to the encoded contract:

equal team within season, then equal season.

Do not recompute the reference weekly.

Document the exact construction.

---

# 39. Wiki

Update the generation wiki page after each generation design is finalized.

Then append results/recommendations after evaluation.

Use:

- `Fingerprint-F06.md`
- `Fingerprint-F09.md`
- `Fingerprint-F10.md`
- `Fingerprint-F11.md`
- `Fingerprint-F12.md`

Update the master ladder and experiment page when needed.

Each generation page should make the new information and major derived formulas understandable without rereading code.

Follow `.llm-wiki/SCHEMA_TDNet.md`.

Keep log entries/commits compliant.

---

# 40. F13–F15

After F12 evaluation is complete, if compute/time remains:

design F13–F15 only.

Do not train them tonight.

Do not preassign information families.

Search:

- cached data inventory
- literature
- unexplored structured football information

Constraints:

- no market
- no win probability
- pregame available
- team-week/matchup compatible
- genuinely new information generation, not just another transform

Document proposals in wiki/config/notes.

---

# 41. PCA

Do not use PCA in F09–F15.

If later explored, treat PCA as a representation suffix such as:

`F12_PCA`

not a new F-number, because it adds no new information source.

---

# 42. Stop conditions

Do **not** stop merely because:

- a small feature family is messy,
- one definition needs a reasonable judgment call,
- one job fails,
- one source is sparse,
- one nonessential endpoint is unavailable,
- a cosmetic diagnostic is imperfect,
- there is an opportunity for more architectural polish.

Document and continue.

Stop the overall run only if continuing would materially risk:

1. 2026 or target leakage,
2. corrupted lineage/scientific conclusions,
3. exceeding the 20k API limit,
4. exceeding reasonable storage limits,
5. unrecoverable or non-auditable results.

---

# 43. Final overnight deliverables

By the end of the run, produce as much of the following as possible:

## Acquisition
- fresh authoritative 2010–2025 schedules
- staged CFBD cache
- request ledger
- quota ledger
- acquisition summary
- Stage E decision

## Fingerprints
- F06 a/b/c and R variants
- F09 a/b/c F/LR/PR
- F10 a/b/c F/LR/PR
- F11 a/b/c F/LR/PR
- F12 a/b/c F/LR/PR

## Screening
- M2/M4 10-setpoint results
- SHAP aggregates
- correlation/redundancy summaries
- pruning traces
- recommendations

## Diagnostics
- up to 1,000 top advanced-feature PNG + Markdown diagnostics
- polished Q4-vs-Q1 rushing predictive figures

## Results
- one ultra-wide results Parquet
- compact recommendation table
- coverage/failure summary
- feature manifests
- hashes/provenance

## Knowledge
- wiki updates
- F13–F15 design proposals if time remains

## Git
- meaningful commits pushed to the experiment branch

---

# 44. Final report to user

When the autonomous run reaches a natural stopping point, report:

### Completed
What acquisition, generations, reductions, SHAP, diagnostics, and recommendations completed.

### Recommended fingerprints
For each generation/design, report the recommended lineage and key metrics.

### Scientific findings
Call out:

- features/families that genuinely added predictive information,
- interesting C composites,
- notable redundant families,
- the Q4-vs-Q1 rushing result,
- unexpected findings.

### Incomplete / failed
Clearly identify anything incomplete and why.

### API/storage
Report:

- calls consumed
- calls remaining
- Stage E choice
- storage used

### Artifacts
Give exact paths for:

- ultra-wide Parquet
- final manifests
- recommendations
- diagnostic directory
- important figures
- acquisition ledger/summary

### Git/wiki
Report:

- final branch head
- major commits
- wiki updates

### Next actions
Give the smallest useful morning checklist for training the six scientific architectures:

- M1
- M2
- M3
- M4
- M5
- M10

on the recommended fingerprints.

Do not claim work succeeded unless the corresponding artifact/test/result actually exists.

---

# Launch

Begin now.

First run the launch preflight.

Then execute Stage A fresh schedules, rerun preflight, and continue through the staged acquisition and F06→F09→F10→F11→F12 experiment autonomously under the contracts above.
