# F0–F20 cumulative 3D rotations, 2026

Each of the 126 markers is one scientific F×M cell's final cumulative straight-up accuracy (x), ATS accuracy (y), and Brier score (z). Marker shapes identify M1–M5 and M10; marker color follows fingerprint order uniformly through F20. The GIFs rotate the same 126 measured points shown in the still PNGs; rotation does not change the data.

| View | Rotating GIF | Still PNG | Cohort |
| --- | --- | --- | --- |
| Latest available | [GIF](cumulative_su_ats_brier_3d_latest_available.gif) | [PNG](cumulative_su_ats_brier_3d_latest_available.png) | F0–F19: 380 games through CFBD Week 7; F20: 320 games through Week 6. The unequal cohorts are shown together to preserve every available weekly result, but endpoint positions are not directly comparable. |
| Matched F20 cohort | [GIF](cumulative_su_ats_brier_3d_paired_304.gif) | [PNG](cumulative_su_ats_brier_3d_paired_304.png) | All 126 cells on the same 304 games, with cumulative Weeks 1–6. Use this view for direct F×M comparisons. |

The [latest-available plotted data](plotted_cumulative_latest_available.csv) and [paired plotted data](plotted_cumulative_paired_304.csv) each contain exactly one row per displayed F×M pair. Both derive from the [126-cell scorecards](../scientific_f0_f20_posthoc_scorecards_week01_07/README.md). The [manifest](manifest.json) records input and output SHA-256 hashes. ATS excludes pushes and games without archived spreads from its accuracy denominator. F20 has no CFBD Week 7 expert rows, so its point uses Week 6. F20 models were fitted after the 2026 games; their scores are retrospective, and the original expert publication times are unverified.

Rebuild from the repository root with the `gridiron` Python environment:

```bash
MPLCONFIGDIR=/tmp/tdnet-mpl /users/tburton2/.conda/envs/gridiron/bin/python scripts/publication/plot_2026_f0_f20_cumulative_3d.py
```
