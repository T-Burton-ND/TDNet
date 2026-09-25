---
type: source-summary
up: "[[Fingerprint-and-Model-Decision-History]]"
source: source-archive/docs/publication_2026/WIDE_MARGIN_FINGERPRINT_DECISION.md.txt
tags: [source, models, decisions]
---

# Wide Margin Decisions Source

The wide-margin decision record, compression study, and cleanup note document why the corrected-F6 operational roster uses full F6 and how its roster labels and poll eligibility were normalized.

## Contribution

The primary decision is to use full corrected F6 for the single-fingerprint wide-margin operational roster, retaining F6-C as supplemental compression analysis and F7/F8 as market-bearing comparisons.

## Methods and findings relevant here

The decision cites 13,920/13,920 corrected F5/F6/F8 HPS trials, 1,080/1,080 scientific heat-map fits, and 1,080/1,080 F6-C candidates with 108/108 leakage-safe selected-path evaluations. These counts and metrics are from the specific grids and folds named in the source, not prospective 2026 outcomes.

On shared outer folds 1–9, full F6 averaged 13.446 points margin MAE versus 13.550 for F6-C across six model types; F6-C improved MAE in one type. Full F6 was selected on the primary margin-error criterion even though F6-C had 0.494 percentage points higher upset recall.

The cleanup record says that 34 learned estimators and two equal-weight ensembles comprise the operational roster; four KNN ballot variants are algorithmic weighting/metric combinations. Legacy KNN feature-subset labels and wrappers adding 117 undeclared coordinates were retired; the corrected fixed F6 input has 681 matchup coordinates, including 24 schedule-graph coordinates. Three statistical estimators remain in margin prediction and ensembles but are barred from Top-25 voting after invalid ordering signatures in rehearsal.

## Intersections

This is the detailed evidence behind [Fingerprint and Model Decision History](Fingerprint-and-Model-Decision-History), and it qualifies the roster overview in [Model and Poll Surfaces](Model-and-Poll-Surfaces). Pre-correction F5-only and 48-cell F0–F7 candidates are explicitly superseded in `SUPERSEDED_MODEL_CLEANUP.md` because F4/F5 were duplicated.

## Source files

- `source-archive/docs/publication_2026/WIDE_MARGIN_FINGERPRINT_DECISION.md.txt`
- `source-archive/docs/publication_2026/F6_COMPRESSION_STUDY.md.txt`
- `source-archive/docs/publication_2026/SUPERSEDED_MODEL_CLEANUP.md.txt`

## See also

[Fingerprint and Model Decision History](Fingerprint-and-Model-Decision-History) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [Deferred and Abandoned Directions](Deferred-and-Abandoned-Directions)
