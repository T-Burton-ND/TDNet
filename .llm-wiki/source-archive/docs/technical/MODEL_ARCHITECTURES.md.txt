# TDKernel, TDTemporal, and TDGraph Design

Last updated: 2026-07-16

## Implemented scope

### TDKernel

`TDKernel` implements four nonlinear similarity models behind the common TDNet
checkpoint/prediction contract:

1. RBF kernel ridge
2. RBF support-vector regression
3. Exact Gaussian-process regression with a deterministic training-size cap
4. Nyström RBF approximation followed by ridge regression

Every type has both winner-centric and margin/MAE-centric search cells. The
winner search still learns a signed score margin and is selected by Brier score;
the margin search is selected by MAE. The kernel sweep contains **7,920**
one-run SGE tasks with `-tc 10`.

### TDTemporal

Temporal fingerprints are produced separately within every trial/fold. For a
curated set of offense, defense, efficiency, turnover, penalty, and sample-size
states, the builder creates:

- one-state lag;
- an exponentially decayed state with tuned half-life; and
- change relative to a tuned 2/3/4/6-state history.

All transformations shift by one before smoothing or differencing. Current and
future rows therefore cannot enter their own time-dependent fingerprint.

Four estimators consume those fingerprints: decay ridge, trend elastic net,
temporal random forest, and temporal histogram gradient boosting. Each has
winner and margin/MAE cells. The temporal sweep contains **9,120** one-run SGE
tasks with `-tc 10`.

The two new arrays total **17,040 tasks**. After successful end-to-end smoke
trials, they were submitted on 2026-07-16 as existing final job `1195167` ->
kernel job `1197150` -> temporal job `1197152`. Both use `-tc 10`. The reusable
submission wrapper remains gated by `--submit` and per-experiment
`smoke_test.ok` markers.

## 2025 rehearsal integration

After the leakage-safe searches and frozen refits finish:

1. merge every outer-fold result and select configurations by the declared
   one-standard-error rule;
2. choose one winner and one margin checkpoint for each of the eight new types;
3. generate locked 2025 predictions from checkpoints that did not train on 2025;
4. add all 16 checkpoints to per-model scorecards, cumulative metrics, TDNet
   ballots, disagreement plots, all-model consensus, and Top-1/Top-3 selection;
5. regenerate every weekly 2025 output and finish each objective package with
   `full_season_poll_grid.{png,svg}`.

The current retrospective 2025 grid is already rendered as a layout example,
but the new models cannot honestly appear in it until their searches and frozen
refits have produced predictions.

Publication figures should use local team logos alongside names whenever the
asset exists. Tables use explicit bounding boxes so captions and receiving-vote
footers do not create large blank bands.

## TDGraph

TDGraph is a representation rather than a predictor. Its private derived-data
directory (`data/derived/td_graph/<season>/`, or group storage at scale) contains:

- a NetworkX directed multigraph (`season.graphml`);
- a node table (`nodes.parquet`) containing team identity, conference,
  classification, record, and degree; and
- a game-edge table (`games.parquet`) containing schedule, score, venue,
  winner, and margin fields.

The database retains every team in the source schedule and is not included in
public bundles under the standing CFBD-data policy. The readable public figure
displays the FBS-induced subgraph, uses both conference colors and node
shapes, overlays normalized team logos where available, labels every node, and
draws games as translucent connections.

## Commands

```bash
# Build manifests only; no submission
scripts/sge/submit_kernel_temporal_search_chain.sh

# Later, after smoke markers exist, chain behind an existing job
HOLD_JID=<job-id> scripts/sge/submit_kernel_temporal_search_chain.sh --submit

# Build a season graph database and figure
PYTHONPATH=src python src/gridiron_ml/cli/graph/build_td_graph.py --season 2025 --completed-only
```
