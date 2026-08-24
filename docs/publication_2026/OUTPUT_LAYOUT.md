# Publication output layout

Current generated publication packages live beneath the external durable
artifact root. Prediction and poll outputs are not split between a
`data/` tree, a figures tree, manual-poll tree, or winner/margin aliases.

Each regenerated season-week uses exactly this layout:

```text
publication/<season>/week_<NN>/
├── wide_margin/
│   ├── predictions_all_games.csv
│   ├── predictions_top25_games.csv
│   ├── poll.csv
│   ├── predictions_all_games.png
│   ├── predictions_top25_games.png
│   ├── poll.png
│   └── full_ballots.png
└── scientific/
    ├── predictions_all_games.csv
    ├── predictions_top25_games.csv
    ├── poll.csv
    ├── predictions_all_games.png
    ├── predictions_top25_games.png
    ├── poll.png
    └── full_ballots.png
```

The canonical weekly blog package also emits two public-facing margin-poll
social assets in its `figures/` directory:

- `week_XX_tdnet_top10_social_4x5.png` — 1080×1350 mobile-first feed graphic.
- `week_XX_tdnet_top10_social_16x9.png` — 1280×720 native landscape recomposition.

Both are generated from ranks 1–10 of the same TDNet margin-objective public
poll snapshot used by the weekly package. They do not recalculate or alter the
Top 25. The style system is centralized in `SOCIAL_STYLE` in
`src/gridiron_ml/publication/social_top10.py`; edit that mapping to iterate on
colors, canvas sizes, font sizes, spacing, or discrepancy-badge thresholds.
The post-game `build_sunday_scorecards` / Top 25 recap workflow emits the same
two assets in each `margin/week_XX/` directory; winner-objective recaps do not
duplicate the canonical margin consensus graphics.
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

Each week-season directory also gets one manifest recording the generation
timestamp, roster label, model-bundle hash, fit cutoff, data snapshot hashes,
and model count. CSVs and PNGs are co-located; no duplicate private/public or
objective-specific copies are produced.

`wide_margin` is the corrected-F6 operational margin roster. `scientific` is
the six-model panel evaluated at every F0–F8 fingerprint. The weekly writer
requires the full 54-cell inventory but emits predictions and polls from only
the 42 market-free F0–F6 cells. F7/F8 never enter official outputs.

`full_ballots.png` shows every model ballot. Scientific retrospective ballot
grids constructed from held-out game predictions are visualization-only implied
rankings, not an additional preregistered poll objective.

The frozen model bundles remain separately under the external durable artifact
root; they are not weekly prediction outputs.

The weekly writer must reject any output path outside this contract. Existing
cleared output trees are not compatibility inputs and should not be recreated.
