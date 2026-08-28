# Publication output layout

Weekly social material uses the operational corrected-F6 wide-margin roster.
The frozen scientific roster also emits a separate, paper-only package from
its market-free F0–F6 cells. Scientific artifacts never enter social outputs.

Each regenerated season-week uses exactly this layout:

```text
publication/<season>/week_<NN>/
├── pre_game/
│   ├── blog/
│   ├── figures/       # PNG picks, Top-25, ballots, and social cards
│   ├── tables/        # local/gitignored predictions, poll, votes, and ballots
│   ├── metadata/
│   ├── scientific/    # immutable paper-only predictions, ballots, power ratings
│   ├── frozen_bundle/ # immutable prediction bytes and verification manifest
│   └── x_post_package/
├── post_game/
    ├── scoring/
    ├── scorecard.csv
    ├── weekly_metrics.csv
    ├── cumulative_metrics.csv
    ├── sunday_performance.png
    ├── figures/       # post-game Top-25, AP comparison, ballots, social assets
    ├── tables/        # local/gitignored post-game poll and ballot snapshots
    ├── scientific/    # distinct post-game package; never overwrites pre-game
    └── x_post_package/
```

Operational weekly figures are PNG-only. Requested descriptive analysis figures
are promoted into the applicable `pre_game/figures/` or `post_game/figures/`
directory instead of creating a separate public `analysis/` subtree. Their
supporting tables remain local under ignored output paths. SVG is not generated
for weekly packages.

The canonical weekly blog package also emits two public-facing margin-poll
social assets in its `figures/` directory:

- `week_XX_tdnet_top10_social_4x5.png` — 1080×1350 mobile-first feed graphic.
- `week_XX_tdnet_top10_social_16x9.png` — 1280×720 native landscape recomposition.

It also emits the prediction companion from the frozen weekly consensus table:

- `week_XX_tdnet_predictions_4x5.png` — 1080×1350 Top 3 + Sickos feed graphic.
- `week_XX_tdnet_predictions_16x9.png` — 1280×720 landscape companion.

The prediction renderer lives in `src/gridiron_ml/publication/social_predictions.py`,
with all prediction-specific design tokens in `PREDICTIONS_STYLE`. It prioritizes
games with two TDNet-ranked teams, then one ranked team; within each bucket it
sorts by the best rank involved, projected closeness, combined rank profile, and
a stable game ID. The closest projected game with no TDNet-ranked teams is
reserved for `SICKOS GAME OF THE WEEK`, with win-probability proximity to 50%
and game ID as deterministic tiebreakers. If fewer than three ranked games are
available, the remaining Top 3 slots use the next-closest unranked games so the
Sickos selection is not duplicated.

The three featured games always share identical geometry: full-width stacked
rows in portrait and three equal columns spanning the landscape. One violet
rule separates those predictions from a navy Sickos panel with violet structure. Matchup content
is packed as one logo/name/prediction block to avoid dead space, and win
probability is one small type step below the projected-margin line.
Portrait matchup marks use a larger logo token than landscape, location is
rendered as `@` (or `VS` for neutral sites), and the centered header subtitle
respects the portrait watermark's top-right safe area. The Sickos module reuses
the same two-team matchup grammar as Games 1–3, with a restrained violet-tinted
panel, the shorter `Closest unranked matchup` subtitle, and a centered
prediction directly below the teams. It intentionally has no separate text
wordmark or nonfunctional ornamentation. The official transparent
`docs/style/logos/Sickos_White.png` mark appears twice behind the matchup as
large, faint edge branding with an integrated violet glow. The asymmetrically
placed marks are clipped through the left and right edges of the rounded Sickos
card. Opacity, glow, scale, positions, panel tint, border,
title, and subtitle controls all live under
`PREDICTIONS_STYLE["sickos"]`; portrait and landscape geometry is deliberately
independent. Both formats also carry a faint TDNet mark in the top-right header
safe area. Prediction-graphic footers direct readers to
the rest of the predictions in the article; Top 10 footers retain their Top 25
copy.

When `market_spread_close` is available, the renderer places `(VEGAS -X.X)`
under the market-favored team. The weekly workflow automatically joins the
season's cached CFBD `/lines` snapshot by game ID and uses the median available
provider spread; `--market-lines-snapshot` can override that default cache.
Its convention is home-team spread: a negative value favors the home team and
a positive value favors the away team. Missing lines are omitted; they are
never inferred from TDNet's model prediction.

Re-render the pair without running models:

```bash
python -m gridiron_ml.cli.publication.render_predictions_social \
  --games path/to/all_games.csv --poll path/to/tdnet_top25.csv \
  --market-lines data/raw/cfbd/v2/lines/2026.parquet \
  --season 2026 --week 1 --output-dir /tmp/tdnet-predictions-review
```

The same 500px logo contract applies. Refresh just the teams selected for the
Top 3 and Sickos modules with:

```bash
python -m gridiron_ml.cli.publication.refresh_social_logos \
  --games path/to/all_games.csv --tdnet-poll path/to/tdnet_top25.csv
```

Both are generated from ranks 1–10 of the same TDNet margin-objective public
poll snapshot used by the weekly package. They do not recalculate or alter the
Top 25. The style system is centralized in `SOCIAL_STYLE` in
`src/gridiron_ml/publication/social_top10.py`; edit that mapping to iterate on
colors, canvas sizes, font sizes, spacing, or discrepancy-badge thresholds.
The post-game Top 25 recap workflow emits the same two assets under the
canonical `week_XX/post_game/figures/` package, with its distinct snapshot in
`post_game/tables/`.
Review renders can be produced without rerunning models:

```bash
python -m gridiron_ml.cli.publication.render_top10_social \
  --poll path/to/tdnet_top25.csv --season 2026 --week 0 \
  --reference-poll path/to/ap_top25.csv --output-dir /tmp/tdnet-social-review
```

The renderer stays network-free. Before a review or weekly release, refresh the
current Top 10 from the 500px sources already identified by TDNet's local logo
manifest, then re-render:

```bash
python -m gridiron_ml.cli.publication.refresh_social_logos \
  --poll path/to/tdnet_top25.csv
```

Rank labels use `#1` through `#10` (or `T-#` for a tie) in the bundled Source Code Pro Black face,
with a portable condensed DejaVu fallback. Every rank is measured in the
selected font and passed through the shared rank/logo collision guard before it
is drawn; a collision or sub-10px gap fails the render instead of producing a
bad social asset.

Logo plates are selected from visible-pixel contrast: dark marks receive an
opaque light plate, while sufficiently bright marks retain the translucent dark
plate. The AP-discrepancy callout is a translucent, slightly rotated stamp that
intentionally straddles its team's panel edge. Poll points are preserved by the
weekly Top 25 loader and shown for all ten teams; every nonzero first-place-vote
count appears on its own line with the compact `FPV` label. One
shared style token controls both the name-to-points and points-to-votes gaps. The rank
hierarchy uses Edge Pink for the `#1` hero, Ion Blue for `#2` and `#3`, and
Signal Orange for `#4`–`#10`. A directional discrepancy stamp appears only
when the absolute TDNet-versus-AP rank gap is at least five places, using `↑`
when TDNet ranks a team higher and `↓` when it ranks a team lower.
Both variants use the official transparent TDNet brand asset as a subdued,
angled white watermark. Its shared angle and opacity plus each variant's box
are style tokens. The landscape mark occupies the center negative space; the
portrait mark sits in the upper-right header space. The seven `#4`–`#10`
landscape panels collectively share the exact top and bottom boundaries of the
`#1` hero panel. Footer metadata uses a human-readable date without a label or
ISO timestamp.
The landscape header also carries a small right-aligned `WEEKLY MODEL
CONSENSUS` label above the divider.

Panel accent bars are clipped to the exact left-hand rounded silhouette. Their
ends taper into the curve while remaining attached, so no accent color can
extend beyond a panel edge.

All cards draw from three shared radius tiers (`hero`, `support`, and `row`).
Lower-row text receives one shared optical vertical correction, and every rank
label is centered within the same fixed-width rank box before the fixed logo
column begins. Lower cards also use identical fixed-height name, points, and
conditional-detail line slots whether or not the final line is visible.

Lower-list logo columns use fixed coordinates regardless of rank-label width,
including tied labels such as `T-8`. The separator is rendered as a compact
custom hyphen rather than consuming a full monospace glyph cell. Every `#4`–`#10` card also reserves two
stat lines even when its conditional detail is blank, keeping its vertical
rhythm stable between weeks. The quieter third line follows a strict priority:
show FPV when positive; otherwise show `BEST: #X` only when the best ballot is
at least three places above the consensus rank; otherwise show compact
`ON X/Y BALLOTS` participation when its rate is below the configurable
30/33 reference; otherwise
leave the reserved line blank. Team-specific
optical logo scaling is centralized in `SOCIAL_STYLE`.
The model-ballot header count remains enabled by default; setting
`ballot_count_last_week` provides a no-layout-change path for retiring it after
the introductory releases.

Both social PNG variants enforce 500×500 source marks for every resolved logo;
an undersized source fails rendering with the high-resolution refresh command
instead of being silently enlarged. Missing marks still use the intentional
initials fallback. Optional feed-specific display aliases remain centralized
with the rest of the style tokens; the default renderer retains full team names.

Equal poll-point totals are rendered as standard competition ties without
mutating the source poll. Every team in the tied group receives the first
occupied `T-#` rank label, and the next ordinal rank is skipped.

Each pre-game package records the generation timestamp, wide-margin roster
label, model-bundle hash, fit cutoff, data snapshot hashes, and model count.
The scientific directory contains three committed CSVs and three PNGs: model ×
game predictions with consensus straight-up/ATS picks, full model × team
ballots with predicted margin against the average team, and an all-team
consensus power ranking. Its reproducibility payload freezes input, code, and
artifact hashes before kickoff. Post-game generation writes only to the sibling
`post_game/scientific/` directory and cannot overwrite pre-game evidence.

`full_ballots.png` shows every model ballot. Scientific retrospective ballot
grids constructed from held-out game predictions are visualization-only implied
rankings, not an additional preregistered poll objective.

The frozen model bundles remain separately under the external durable artifact
root; they are not weekly prediction outputs.

The weekly writer must reject any output path outside this contract. Existing
cleared output trees are not compatibility inputs and should not be recreated.
