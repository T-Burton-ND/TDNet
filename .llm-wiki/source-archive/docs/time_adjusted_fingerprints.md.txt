# Time-Adjusted Fingerprints

## Goal

The opponent-adjusted fingerprints answer: "How did this team perform after accounting for opponent quality?"

The time-adjusted family adds a second question: "Was that opponent-adjusted performance strong for this part of the season?"

Early-season fingerprints have a different distribution than late-season fingerprints because teams have fewer games, roster continuity matters more, and opponent information is sparse. A raw or opponent-adjusted value can therefore mean different things in Week 2 than it means in Week 11.

## Design

The first time-adjusted family builds on the opponent-adjusted parquet frames instead of replacing them. Each version keeps the source frame intact, then appends `time_adj_*` features for selected numeric `opp_adj_*` columns.

The adjustments use only prior seasons as the reference population for each row's season. For a row in season `S`, week `W`, the baseline is computed from seasons `< S`. This avoids learning from future rows in the same season.

Implemented variants:

- `t2.1 same_week_z`: z-score each selected opponent-adjusted feature against prior-season rows from the same week.
- `t2.2 phase_z`: z-score against prior-season rows from the same season phase: preseason, early, middle, late, postseason.
- `t2.3 recency_week_z`: same-week z-score, but prior seasons are exponentially weighted so recent seasons matter more.

Each generated frame records:

- `fp_experiment = time_adjusted_fingerprints`
- `fp_source_label`, such as `v1.7`
- `fp_method`, such as `same_week_z`
- `fp_version_label`, such as `t2.1`

## Why This Shape

This design keeps the experiment memory-safe and reversible:

- It does not mutate the canonical fingerprint registry.
- It reuses the existing TDNet `StaticFrameFingerprints` training path.
- It can be trained by stat, linear, and tree models without changing model APIs.
- It avoids same-season leakage in the week baselines.
- It lets hyperparameter search decide whether week-normalized opponent-adjusted features help.

## Usage

Build the default three versions:

```bash
python src/gridiron_ml/cli/run_time_adjusted_experiment.py build
```

Build a small smoke frame:

```bash
python src/gridiron_ml/cli/run_time_adjusted_experiment.py build \
  --output-root /tmp/tdnet_time_adjusted \
  --labels t2.1 \
  --max-features 8 \
  --overwrite
```

Train one stat, one linear, and one tree model against the frame:

```bash
python src/gridiron_ml/cli/run_time_adjusted_experiment.py smoke-train \
  --output-root /tmp/tdnet_time_adjusted \
  --frame-label t2.1 \
  --smoke-output-root /tmp/tdnet_time_adjusted_smoke
```

## Next Search Plan

Do not submit this yet. The next natural search should compare:

- source labels: `v1.4`, `v1.7`, and any future best opponent-adjusted source
- time labels: `t2.1`, `t2.2`, `t2.3`
- top-k feature truncation: `25`, `50`, `100`, `200`, `all`
- objectives: winner accuracy, winner/upset balance, margin prediction

The current hyperparameter search runner can be reused once the time-adjusted frames are built; point `--source-fingerprint-root` at `data/experiments/time_adjusted_fingerprints`.
