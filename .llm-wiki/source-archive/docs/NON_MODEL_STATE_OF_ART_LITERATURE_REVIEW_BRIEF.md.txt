# TDNet non-model-science literature review brief

Use this document as the prompt for a deep, current literature search. The
objective is to identify standard, state-of-the-art, or publication-critical
methods that TDNet is missing outside the scientific model architectures.

## Role

Act as a skeptical methods editor and sports-analytics reviewer. Search the
current literature and standards, then compare them against the repository
baseline below. Focus on methods that affect validity, reproducibility,
calibration, uncertainty, data governance, prospective evaluation, and public
reporting.

Do not recommend new predictive model architectures. Do not redesign the F0–F7
fingerprint ladder or the six confirmatory candidates. Treat the architecture
set as frozen and investigate everything around it.

Use sources from 2018–present where possible, while retaining older seminal
work when it remains the accepted foundation. Prefer primary papers, official
standards, official documentation, and authoritative methodological reviews.
For every material recommendation provide a DOI, publisher page, arXiv link,
standard URL, or other stable citation. Distinguish peer-reviewed evidence,
preprints, standards, software documentation, and informed inference.

## Project context

TDNet is a college-football forecasting and publication system. It builds
time-available team fingerprints from CollegeFootballData (CFBD), generates
game-level winner and margin forecasts, produces a weekly Top-25 poll and
consensus, compares against Vegas and AP, and prospectively evaluates the
locked system during 2026.

The confirmatory scientific boundary is fixed:

- F0–F6 are market-free fingerprint tiers.
- F7 adds declared market variables and is a comparative market-aware tier.
- 2025 is consumed retrospective evidence, not an untouched test set.
- 2026 is the prospective season.
- Six confirmatory model candidates are frozen separately from the broader
  operational weekly roster.
- Raw CFBD-derived tables are not redistributed.
- CFBD attribution and source provenance are required in public materials.

Primary local references:

- `docs/publication_2026/CONFIRMATORY_PROTOCOL.md`
- `docs/publication_2026/WEEKLY_OPERATIONS.md`
- `docs/publication_2026/CURRENT_RELEASE_STATUS.md`
- `docs/publication_2026/TDNET_PUBLICATION_FREEZE_REPORT.md`
- `docs/TDNET_MASTER_PLAN.md`
- `configs/publication/confirmatory_protocol.yaml`
- `configs/publication/data_source_manifest.yaml`
- `src/gridiron_ml/pipeline/`
- `src/gridiron_ml/publication/`
- `tests/test_leakage_validation.py`
- `tests/test_future_prediction_row_boundaries.py`
- `tests/test_market_artifact_isolation.py`
- `tests/test_publication_protocol.py`

## Baseline audit: what is already present

Treat these as implemented claims to verify, not assumptions to repeat:

1. Data snapshots are cached as endpoint/year parquet files. The weekly
   snapshot checker records hashes, schema, row counts, and missing-field
   evidence, and blocks certification when required inputs are incomplete.
2. Feature manifests record ordered feature names, source/family, cutoff,
   missingness rule, market/opponent flags, version, and schema hash.
3. Temporal row semantics and future-prediction boundaries are tested. Leakage
   validation includes fixture and replay-specific audits.
4. The publication boundary uses immutable weekly prediction bundles,
   append-only hash-linked amendments, source hashes, and explicit approval.
5. Evaluation includes Brier, log loss, accuracy, margin MAE/RMSE, calibration
   intercept/slope, ECE, reliability bins, sharpness, season-clustered paired
   bootstrap, McNemar, Holm correction, equivalence, and power/precision tools.
6. Strict OOF KNN and transparent baselines have explicit temporal contracts.
7. The weekly workflow separates Monday refresh, Tuesday freeze, and Sunday
   scoring; it refuses uncertified snapshots and does not overwrite prediction
   bytes.
8. The repository has a package-level test suite, environment locks, release
   gates, portability checks, compact manifests, and a CFBD attribution policy.

## Audit questions for the literature search

For each section, answer four things:

1. What is the accepted state of practice?
2. What does TDNet already satisfy?
3. What is missing or only partially implemented?
4. Is the gap publication-critical, strongly recommended, optional, or not
   applicable? Give a concrete implementation or documentation action.

### A. Data provenance, revisions, and source governance

Search for best practices covering:

- versioned sports-data snapshots and immutable dataset identifiers;
- source revisions, backfills, corrections, and retroactive statistics;
- API endpoint versioning, schema drift, rate limits, retries, and outages;
- recording retrieval time separately from event time and publication time;
- data cards, datasheets, dataset statements, and provenance graphs;
- reproducible citation of commercial or restricted data sources;
- audit trails for derived tables and transformation lineage;
- licensing, attribution, redistribution, and acceptable public summaries for
  restricted sports data.

Assess whether TDNet needs a formal data card, endpoint release ledger,
revision policy, source timestamp table, or stronger CFBD terms-of-use record.

### B. Temporal availability and leakage beyond ordinary train/test splits

Search for current guidance on:

- prequential or rolling-origin evaluation;
- purged and embargoed time-series validation;
- delayed labels and delayed feature publication;
- event time versus observation time versus ingestion time;
- revisions to historical covariates after the forecast cutoff;
- point-in-time databases and bitemporal data models;
- leakage from season aggregates, rankings, coaching data, injuries, lines,
  postseason information, and team identity changes;
- future-perturbation tests and negative-control tests for leakage.

Do not merely ask whether a feature happened before kickoff. Ask whether the
value used was actually observable to a forecaster at the declared deadline.
Identify a minimal point-in-time audit record TDNet should store per feature
family and per weekly bundle.

### C. Missing data, imputation, and data-quality uncertainty

TDNet has temporal donor imputation and endpoint completeness checks. Search for
methods and reporting standards for:

- distinguishing missing completely at random, at random, and not at random;
- informative missingness caused by coverage, game status, or provider delay;
- temporal donor restrictions and donor-pool contamination;
- multiple imputation and combining uncertainty across imputations;
- pattern-mixture, selection-model, and worst-case sensitivity analyses;
- missing-not-at-random sensitivity reporting in forecasting studies;
- data-quality scoring and quality-aware forecast exclusion rules;
- uncertainty propagation from missing inputs into rankings and polls.

Recommend whether TDNet needs multiple-imputation sensitivity, missingness
indicators, endpoint-specific quality weights, or a formal abstention policy.

### D. Forecast calibration, uncertainty, and decision usefulness

Search beyond ordinary reliability plots for:

- modern probabilistic forecast evaluation and calibration diagnostics;
- finite-sample calibration assessment and confidence bands;
- conformal prediction or conformal risk control for temporal, grouped, or
  dependent data;
- prediction intervals for margins and set-valued winner forecasts;
- distributional or quantile forecast scoring rules;
- proper scoring rules, skill scores, and baseline-relative scores;
- calibration drift and online recalibration under concept drift;
- selective prediction, abstention, and coverage guarantees;
- decision-curve or utility analysis for non-betting public decisions.

Separate methods that are valid for dependent weekly sports data from methods
that assume IID observations. State whether each method can be added without
changing the frozen model architectures.

### E. Dependence, uncertainty, and comparison of many forecasts

TDNet already uses season-clustered paired bootstrap and Holm correction.
Search for:

- hierarchical and multiway clustered uncertainty for games, teams, seasons,
  conferences, and weeks;
- block bootstrap, moving block bootstrap, stationary bootstrap, and cluster
  bootstrap for sports schedules;
- corrected tests for comparing forecasts over overlapping horizons;
- multiple-model comparison, model confidence sets, and forecast combination
  uncertainty;
- selective inference after ranking, finalist selection, or poll-member
  selection;
- multiplicity across fingerprints, objectives, seasons, metrics, subgroups,
  and model variants;
- preregistration, registered reports, blinded analysis, and locked analysis
  plans for prospective forecasts.

Determine whether the current comparison family is complete and whether the
2026 prospective results require a predeclared multiplicity table.

### F. Sports ranking, poll aggregation, and human-in-the-loop forecasts

TDNet publishes learned-model ballots, an equal-weight consensus, and an owner
Top-25 ballot. Search for:

- rank aggregation and consensus methods for partial or tied rankings;
- uncertainty intervals or stability measures for rankings;
- inter-rater reliability for human and machine ballots;
- expert-forecast aggregation, wisdom-of-crowds failure modes, and anchoring;
- handling missing ballots and model availability without changing the target;
- rank reversals, sensitivity to top-k truncation, and robustness to ties;
- calibration and scoring of ordinal rankings versus game-level forecasts;
- transparent rules for changing poll members during a live season.

Do not recommend changing the declared poll rule without showing the scientific
estimand it would improve. Identify useful diagnostics that can be added while
keeping the weekly poll deterministic and auditable.

### G. Vegas and external benchmark comparability

Vegas is an evaluation comparator, not a TDNet input for F0–F6. Search for:

- correct conversion of spreads, moneylines, totals, and vig-adjusted implied
  probabilities;
- opening, consensus, closing, and timestamped line definitions;
- line movement and information leakage when line data are revised;
- benchmark selection and fair comparison when one system has market-derived
  information;
- paired forecast comparison against a changing external benchmark;
- reporting standards for market baselines without implying betting advice;
- domain shift between bookmaker, exchange, and CFBD line sources.

Audit whether TDNet needs explicit overround removal, line-source metadata,
timestamp/latency fields, or separate market-information sensitivity analyses.

### H. Distribution shift, robustness, and subgroup reporting

Search for non-architecture methods for:

- concept drift and covariate shift in sports seasons;
- pre-season versus in-season distribution changes;
- conference realignment, rule changes, schedule imbalance, and team identity
  changes;
- robustness checks across conferences, team tiers, home/away, neutral sites,
  FBS/FCS exclusions, weather, and early/late season;
- temporal subgroup confidence intervals and small-cell safeguards;
- transportability and external validity for a single future season;
- stress testing under schedule, roster, and data-availability perturbations.

Recommend a minimum subgroup and stress-test table that does not become a
post-hoc fishing expedition.

### I. Reproducible computational research and artifact provenance

Search for accepted practice around:

- environment lockfiles versus immutable container digests;
- reproducible builds, source trees, Git tags, and dependency attestations;
- dataset and artifact versioning for restricted data;
- content-addressed storage, Merkle manifests, and signed release artifacts;
- SLSA, in-toto, Sigstore, RO-Crate, Research Object Crate, and provenance
  predicates as applicable to research software;
- workflow engines, job manifests, retries, idempotency, and scheduler audit
  trails;
- archival publication packages when raw data cannot be redistributed;
- executable papers, computational notebooks, and CI requirements for
  reproducible results.

Assess the current hash manifests and locked environments. State the smallest
additional provenance package needed for another researcher to verify every
claim without receiving CFBD raw data.

### J. Reporting, transparency, and publication standards

Search current reporting guidance relevant to a prospective sports forecasting
study, including but not limited to:

- TRIPOD and TRIPOD+AI;
- PROBAST and PROBAST+AI;
- CONSORT-AI or other prospective evaluation guidance where applicable;
- STROBE, RECORD, MINIMAR, data-sheet/data-card guidance, and software-paper
  checklists;
- forecast verification and uncertainty-reporting standards;
- preregistration and registered-report expectations for predictive studies;
- transparent handling of negative results, unavailable endpoints, and
  protocol amendments.

Do not assume a clinical reporting guideline transfers directly to sports.
Explain applicability, limitations, and which checklist items should be
adopted or explicitly marked not applicable.

### K. Public data governance and responsible communication

Search for guidance on:

- licensing and attribution for commercial sports data;
- reproducible releases using derived aggregates instead of raw data;
- provenance statements that let readers audit source and time boundaries;
- privacy, re-identification, and ethical use of player/coach information;
- communicating uncertainty in rankings and game forecasts;
- avoiding implied betting advice and exaggerated benchmark claims;
- correction, retraction, and amendment policies for live publications.

Assess the CFBD attribution, private-data boundary, owner-approval gate, and
append-only amendment ledger.

## Required search strategy

Use multiple searches per section. Include combinations of:

- `sports forecasting`;
- `college football analytics`;
- `time series forecast evaluation`;
- `prequential evaluation`;
- `point-in-time data`;
- `temporal leakage`;
- `data revision provenance`;
- `forecast calibration dependent data`;
- `conformal prediction time series`;
- `sports ranking uncertainty`;
- `rank aggregation partial rankings`;
- `forecast comparison clustered bootstrap`;
- `prospective prediction preregistration`;
- `reproducible computational research provenance`;
- `data card restricted commercial data`;
- the named reporting standards above.

Search Google Scholar, Crossref, Semantic Scholar, arXiv, PubMed where
relevant, official standards pages, and primary publisher pages. For each
important paper, verify the publication year, venue, DOI, assumptions, and
whether the result actually applies to temporally dependent sports data.

## Required output

Return a report with these sections:

1. **Executive verdict:** the five most important non-model gaps, ranked by
   publication risk.
2. **Evidence table:** area, recommendation, source, assumptions, TDNet status,
   priority, and exact repository action.
3. **Keep / strengthen / add / defer:** classify every recommendation.
4. **Minimal publication gate:** the smallest set of changes required before a
   defensible 2026 prospective release.
5. **Nice-to-have research extensions:** useful but non-blocking additions.
6. **Rejected or inapplicable recommendations:** explain why they do not fit
   TDNet, especially recommendations that secretly change the frozen science.
7. **Citation pack:** stable links/DOIs and one-sentence relevance notes.
8. **Implementation backlog:** repository-relative file targets, tests, and
   acceptance criteria for each accepted change.

Do not merely list papers. Convert each source into a concrete judgment about
TDNet’s current validity, reproducibility, or publication readiness. Explicitly
flag when the literature is inconclusive or when a proposed method would create
new leakage, post-hoc selection, or an untracked estimand change.
