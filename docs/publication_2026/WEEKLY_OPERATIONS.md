# TDNet 2026 weekly operations

For an owner-driven run, open `publication/notebooks/weekly/tdnet_manual_top25_poll.ipynb`. Its
settings cell controls the season/week, model-family Boolean switches, refit,
weekly preview, and comparison suite. The notebook also presents the
click-drag Top-25 ballot editor and saves the submitted ballot into the same
season ledger used by the publication poll workflow.

The paper/weekly inventory is the learned-model inventory in
`docs/publication_2026/weekly_learned_model_inventory.csv`. It is generated from
the corrected-F6 wide-margin bundle: 34 learned estimators plus two ensembles.
Three statistical estimators remain prediction members but are excluded from
poll voting, leaving 33 automated poll members. The owner contributes one
separate manual ballot.

When a runtime-ready scientific inventory is available, pass it with
`--scientific-inventory` to the weekly publication pipeline. The full inventory
must contain 54 cells (F0–F8 × M1/M2/M3/M4/M5/M10). The command then
writes both `week_<NN>/wide_margin/` and `week_<NN>/scientific/`, including one
poll, consensus-spread PNG, descriptive discrepancy-feature PNG, full-ballot PNG, all-game predictions, and Top-25-game predictions for
each roster. F7/F8 remain in the research inventory but are automatically
excluded from official predictions, consensus, and polls.

All times are America/New_York unless the prediction deadline explicitly uses
UTC. Exact Monday and Tuesday clock times remain owner-configurable.

## Monday: refresh, rebuild, inspect

```bash
PYTHONPATH=src python src/gridiron_ml/cli/publication/run_monday_refresh.py --season 2026 --week WEEK
```

This refreshes the 2026 CFBD cache, team-game table, v0 fingerprint, and all
opponent-adjusted fingerprints. It writes only aggregate inspection metadata to
`data/publication/2026/weekly_operations/week_XX/`. It removes any prior approval
marker so stale approval cannot carry between refreshes.

The refresh also runs `src/gridiron_ml/cli/publication/check_weekly_snapshot.py`. It records
one hash/schema/count record per configured CFBD endpoint. A missing endpoint,
empty required response, or missing required field makes the snapshot
`weekly_snapshot_not_certified` and prevents approval.

Review the inspection report and approve it explicitly:

```bash
python src/gridiron_ml/cli/publication/approve_monday_refresh.py --season 2026 --week WEEK
```

## Tuesday: freeze one bundle and prepare publication

```bash
PYTHONPATH=src python src/gridiron_ml/cli/publication/run_tuesday_publish.py \
  --season 2026 --week WEEK \
  --deadline-utc YYYY-MM-DDTHH:MM:SSZ \
  --deadline-local-date YYYY-MM-DD \
  --ap-top25 data/raw/cfbd/v2/rankings/2026.parquet \
  --canonical-poll-objective margin
```

Tuesday refuses to proceed without Monday's approval marker or a certified
weekly snapshot. It creates one
immutable deadline bundle for the week, automatically makes the canonical
margin-objective Top-25 poll from the learned-model inventory (including KNN),
and includes the poll artifact. Pass
`--canonical-poll-objective margin` only after the Vegas review; until then,
omit it and the bundle records that canonical selection is pending. The bundle
also includes the locked-roster predictions, AP-ranked
games, TDNet-versus-AP comparison, and an X draft package. It does not send an
X post.

## Sunday: score and prepare a retrospective review bundle

After CFBD outcome/statistic completeness passes, score the immutable prior
prediction bundle without modifying it:

```bash
PYTHONPATH=src python src/gridiron_ml/cli/publication/run_sunday_publication_pipeline.py \
  --bundle publication/2026/week_XX \
  --results data/raw/cfbd/v2/games/2026.parquet \
  --snapshot-completeness data/publication/2026/weekly_operations/week_XX/snapshot_completeness.json \
  --output-root publication/2026/week_XX
```

The command fails closed on an uncertified snapshot or invalid bundle, writes
weekly/cumulative metrics, comparison tables, figures, and draft-only blog/X
assets, and records source hashes. It never overwrites prediction bytes or
sends external posts. The `--results` file must contain authoritative
`game_id`, `home_points`, and `away_points` fields; raw CFBD data remain
authorized-download-only and are not redistributed.

## Scheduling

Do not install a cron entry until the owner chooses exact clock times and the
runtime environment has `CFBD_API_KEY`. A template is in
`configs/publication/weekly_cron.template`; replace the placeholders before use.

Direct X posting additionally requires an owner-controlled X developer app and
credentials. Until configured, `x_post_package/manifest.json` always records
`draft_only_requires_explicit_approval`.

## Preseason AP release

The preseason command refreshes rankings, talent, and returning production and
stops unless CFBD contains an official 25-row AP Top 25:

```bash
MPLCONFIGDIR=/tmp/tdnet-mpl PYTHONPATH=src \
  python -m gridiron_ml.cli.publication.run_2026_preseason_release
```

Before AP exists it reports `waiting_for_official_ap_top25`. Once AP appears,
the same command creates TDNet preseason polls, AP comparisons, Week 1
predictions, figures, failure reports, and a blog draft under
`publication/2026/preseason/`. It never substitutes the Coaches Poll in a
release package. If 2026 talent remains unavailable, the derived Week-0 state
uses the labeled 2025 talent carry-forward; live 2026 returning production is
overlaid when available.

## Top-25 blog graphics

Every frozen margin-objective roster poll now writes its blog figures beside
`tdnet_top25.png` and `tdnet_model_ballots.png`:

1. `top25_consensus_spread.png` shows every model's ballot rank for each TDNet
   Top-25 team, with the TDNet consensus and official AP markers.
2. `top25_discrepancy_features.png` (when an AP snapshot and the current F6
   state are available) selects the three largest absolute TDNet-versus-AP gaps
   among teams ranked by both. It summarizes the three largest grouped,
   standardized fingerprint differences from AP-rank peers.

The second chart is deliberately a descriptive comparison aid, not SHAP or a
causal decomposition: AP is not a model input, and it must not be read as an
explanation of AP voters or unencoded information such as a new coach's
identity. The chart uses only the frozen roster's contemporaneous public-week
state and the feature-family mapping in
`docs/publication_2026/FINGERPRINT_FEATURE_MATRIX.csv`; it does not retrain or
alter any frozen model.
