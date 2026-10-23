# F0–F20 scientific violin plots, 2026 Weeks 1–6

These update the earlier [architecture and fingerprint violins](../scientific_total_quality/README.md) to include F20. Every point is one of the 126 F0–F20 × M1–M5/M10 model cells, evaluated on the **same 304 completed games** through CFBD Week 6. The older 120-cell Week 7 figures remain a separate frozen snapshot.

| Figure | Full view | Publication view |
| --- | --- | --- |
| Architecture distributions: 21 fingerprint scores per model family | [PNG](architecture_violin.png) · [SVG](architecture_violin.svg) | [PNG](architecture_violin_publication.png) · [SVG](architecture_violin_publication.svg) |
| Fingerprint spread: six model scores per fingerprint, including F20 | [PNG](fingerprint_violin.png) · [SVG](fingerprint_violin.svg) | [PNG](fingerprint_violin_publication.png) · [SVG](fingerprint_violin_publication.svg) |

The x-axis reproduces the earlier balanced relative skill measure: the arithmetic mean of relative improvement in margin MAE, Brier, and straight-up accuracy versus archived Vegas on those **same 304 games**. Zero is Vegas. The source-market baseline is 11.429276-point MAE, 78.9474% SU accuracy, and 0.147181 Brier on that cohort. The original analysis uses a different Week 7 cohort; compare within each figure, not between the two releases. F20 follows the same colors, markers, and row styling as every other fingerprint. The bar is the interquartile range, the white tick is the median, the open circle is the arithmetic mean, and every exact cell score is plotted. Violin smoothing is illustrative, especially for the six-point fingerprint rows.

The [126-cell score ledger](paired_304_balanced_skill.csv), [architecture summary](architecture_distribution.csv), [fingerprint summary](fingerprint_distribution.csv), and [hash manifest](manifest.json) make the calculation auditable. The publication versions omit the title and footer so a caption should state that these are **retrospective** scores: F20 models were fitted after the 2026 games, original expert publication times are unverified, and some historical market quote times are unverified. They are not prospective performance or betting returns.

Rebuild from the repository root:

```bash
python scripts/publication/plot_2026_f0_f20_violin.py
```

Regeneration requires the owner-local F20 game replay in `data/private/` and the archived common-market game table. Neither is part of this figure release.
