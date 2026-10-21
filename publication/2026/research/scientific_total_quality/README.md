# 2026 scientific roster: six one-point views

Each figure contains the same 120 F0–F19 scientific cells (M1–M5 and M10) and one Vegas point. F17 is the `F17-market` research fingerprint in the source roster. F19/M4 is pink (`edgePink`); Vegas is gold (`brass`). The other model colors, typography, grid, and background follow [`gridiron.palette.json`](../../../../gridiron.palette.json) and [`gridiron_light.mplstyle`](../../../../docs/style/color_palettes/gridiron_light.mplstyle).

| Figure | Main question | Encoding |
| --- | --- | --- |
| [01 Error landscape](01_error_landscape.png) | Who approaches Vegas in both margin and probability error? | MAE × Brier; straight-up wins by point size; architecture by color and shape. |
| [02 Accuracy landscape](02_accuracy_landscape.png) | Does picking more winners track lower margin error? | MAE × straight-up accuracy; Brier by blue shade; architecture by shape. |
| [03 Balanced quality ladder](03_balanced_quality_ladder.png) | What is the complete order under one explicit summary index? | One common horizontal quality scale, with every point separately ranked. |
| [04 Roster matrix](04_roster_matrix.png) | Which fingerprint and architecture pairs account for the pattern? | Unique fingerprint × architecture position; quality by blue shade. |
| [05 Architecture swarm](05_architecture_swarm.png) | How broad is each architecture's quality distribution? | Quality on a common horizontal scale; deterministic vertical packing for nearby points. |
| [06 Fingerprint progression](06_fingerprint_progression.png) | How does performance vary across the fingerprint generations? | Fingerprint position × quality; architecture by color and shape. |

SVG versions have the same base names. The [paired score data](paired_quality.csv) contains exact values and metric ranks for all 121 points. The [manifest](manifest.json) records source and output hashes. Rebuild with [`plot_2026_scientific_total_quality.py`](../../../../scripts/publication/plot_2026_scientific_total_quality.py).

## Metric and comparison method

All margin MAE and straight-up results use the same **379 spread-covered games**. All Brier results use the same **372 moneyline-covered games**. The figure script checks that every model has every paired game ID before scoring. The Vegas point has 11.3892-point MAE, 294/379 straight-up wins (77.57%), and 0.157786 Brier. F19/M4 has 11.3882-point MAE, 296/379 wins (78.10%), and 0.159724 Brier. Its MAE edge is about 0.0010 point; its Brier is worse than Vegas.

The descriptive **balanced relative skill** index is the arithmetic mean of three percentage changes versus Vegas:

```text
MAE skill   = 100 × (Vegas MAE − model MAE) / Vegas MAE
Brier skill = 100 × (Vegas Brier − model Brier) / Vegas Brier
SU skill    = 100 × (model SU accuracy − Vegas SU accuracy) / Vegas SU accuracy
Balanced relative skill = (MAE skill + Brier skill + SU skill) / 3
```

Zero is Vegas. Higher is better. This index is a visualization convenience, not a significance test or a betting utility. Equal relative weighting is a design choice; Brier and straight-up accuracy both reflect winner prediction, so the index should not replace the two raw-metric landscapes. Neither the small F19/M4 MAE edge nor its two extra correct winners establishes superiority over Vegas.

## Design research

[Cleveland and McGill's graphical-perception experiments](https://www.tandfonline.com/doi/abs/10.1080/01621459.1984.10478080) motivate using positions on common axes for the main comparisons. Their [scatterplot methods paper](https://www.tandfonline.com/doi/abs/10.1080/01621459.1984.10477098) discusses adding category codes and point sizes to a scatterplot, motivating the two metric landscapes. [Matplotlib's scatter example](https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_demo2.html) documents the color and size encodings used here. The matrix preserves one unambiguous position per model when scatter points overlap; the ladder and swarm preserve a common quality axis for comparing all 121 points.

## Provenance limits

These are through-Week-7 **retrospective** comparisons. Weeks 6–7 scientific backfill was run after games; Week 7 market quotes were retrieved after games, and provider quote times are unverified. F17-market is a research branch that uses market inputs. The figures are descriptive performance summaries and cannot be used as leakage-free prospective evidence. See the Week 7 source files and their provenance columns for the underlying game records.
