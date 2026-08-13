# TDNet

TDNet is a research-oriented Python package for college football analytics. It fetches CFBD data, builds time-dependent team fingerprints, trains matchup models, compares predictions with Vegas market baselines, produces weekly poll/report artifacts, and runs TD Sim season simulations.

## Data attribution and 2026 study boundary

Schedules, outcomes, statistics, and market-line context are sourced from
[CollegeFootballData (CFBD)](https://collegefootballdata.com/). CFBD is the
primary data provider for the TDNet publication and prospective workflow; see
the protocol and data-source manifests for endpoint-specific cutoff and
completeness rules. Raw CFBD data are not redistributed by this repository.

The 2025 season is consumed retrospective evidence, not an untouched test set.
The 2026 prospective study is FBS-versus-FBS regular season only. F6 is the
richest market-free TDNet fingerprint, F7 is a market-only benchmark, and F8
combines F6 with market information. F7/F8 are research comparators and never
enter official TDNet predictions, consensus, or polls. Vegas is an external benchmark. The repository makes no default
claim that TDNet beats Vegas, that complexity universally improves prediction,
that feature importance is causal, or that every opponent-adjustment method
helps.

The publication-facing rules are in
[`docs/publication_2026/CONFIRMATORY_PROTOCOL.md`](docs/publication_2026/CONFIRMATORY_PROTOCOL.md)
and the machine-readable source is
[`configs/publication/confirmatory_protocol.yaml`](configs/publication/confirmatory_protocol.yaml).

## Confirmatory models and operational roster

TDNet has two deliberately different model surfaces for 2026.

The six-model scientific panel is a controlled architecture comparison. Each
architecture is fitted at every fingerprint F0–F8: 54 margin models in total.
The 42 market-free F0–F6 cells are eligible for prospective predictions and
polls; the 12 market-bearing F7/F8 cells are research comparisons only. The
panel asks: **given the same football information, how do different modeling
approaches perform?**

| Candidate | Architecture | Plain-language interpretation |
|---|---|---|
| M1 | Linear | A weighted scorecard in which each feature adds or subtracts a fairly consistent amount. |
| M2 | Spline | A flexible scorecard whose effects can bend rather than staying strictly straight-line. |
| M3 | Decision tree | A collection of learned “if/then” rules and feature interactions. |
| M4 | Boosted trees | Many small trees built sequentially, with each one correcting earlier mistakes. |
| M5 | Neural network | Layers of nonlinear combinations that can learn complicated interactions. |
| M10 | K-nearest neighbors | A search for historically similar matchups whose outcomes inform the prediction. |

The market-free scientific cells are eligible to contribute to the weekly poll
and consensus. Their primary purpose is a stable, interpretable comparison
without silently changing information, training boundaries, or selection rules.

The corrected-F6 wide-margin bundle contains 34 learned estimators and two
equal-weight ensembles. Three statistical estimators remain game-prediction
members but are excluded from Top-25 voting because their team-ordering surface
failed the prespecified poll sanity check. The weekly poll therefore has 33
automated members plus one separate owner-supplied manual ballot.

The 33 automated members contain a broader mix of model variants, KNN
configurations, ensembles, and other specialized candidates. That diversity
can make weekly forecasts more robust, but it also makes a single clean
architecture comparison harder to interpret. The owner ballot is always
reported separately and never changes model consensus or model metrics.

The complete scientific roster has six architectures across all nine
fingerprints (54 cells). The prospective market-free study uses F0–F6 (42
cells); F7 is market-only and F8 is F6-plus-market. This does not by itself
prove that no leakage occurred; that requires cutoff audits, immutable
prediction bundles, future-perturbation tests, and exclusion of 2026 outcomes
from model selection and retraining.

For model-family details, see [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md). The consolidated project plan is [TDNET_MASTER_PLAN.md](docs/TDNET_MASTER_PLAN.md).

## Setup

```bash
conda env create -f configs/env.yaml
conda activate gridiron
```

Or install into an existing Python environment:

```bash
pip install -e ".[dev,notebooks]"
```

Run tests:

```bash
pytest -q
```

Run common workflows from the repo root:

```bash
scripts/tdnet-run.sh data
scripts/tdnet-run.sh eval
scripts/tdnet-run.sh blog
scripts/tdnet-run.sh sim --season 2026 --n-sims 100
```

Each script writes a timestamped log under `docs/logs/`. The same runner is exposed as `tdnet-run` after editable install.

## Publication notebooks

The recurring publication workflow and the public reproduction notebook live
under `publication/notebooks/`:

- `publication/notebooks/weekly/tdnet_weekly_predictions.ipynb` — inspect the upcoming schedule and create
  the game-level prediction table.
- `publication/notebooks/weekly/tdnet_manual_top25_poll.ipynb` — edit the owner ballot and merge it with
  learned-model ballots.
- `publication/notebooks/weekly/tdnet_sunday_results.ipynb` — score the immutable prediction bundle and
  render the weekly review.
- `publication/notebooks/reproduction/tdnet_data_and_model_reproduction.ipynb`
  — fetch CFBD data with the reader's own key, build fingerprints, train models,
  and evaluate them locally.

General training, evaluation, simulation, figure-development, and exploratory
notebooks were retired after their workflows moved into package code and the
publication runners.

The publication notebook keeps the owner ballot and automated ballots for each
season/week under
`publication/<season>/manual_polls/`. It starts a new week from the latest
model-produced Top 25, merges the submitted manual ballot with the currently
informative automated ballots, and saves both the resulting poll and the
all-ballots logo matrix. Install `.[notebooks]` for the `ipywidgets`
dependency.

## Core Concepts

v0 fingerprints use `state_after_week` row semantics:

- `keys_week=N` includes games completed through week `N`.
- Week `0` is preseason/bootstrap state.
- `y_next_margin` is the safe default training target.
- `y_margin_this_week` is same-row completed-game information and is unsafe for training with postgame/current-week features.

Market/Vegas columns are evaluation context by default. Training rejects `market_*` columns unless a config explicitly opts into market features.

Opponent-adjusted fingerprint columns can flow through `unit_matchup` without a
pairing contract when their names match the configured
`matchup.unit_passthrough_patterns` values. The default patterns cover
`opp_adj_*`, `opponent_adj_*`, `opponent_adjusted_*`, and `adjusted_*`; the
builder emits `home_`, `away_`, and `net_` matchup columns for those features.

## Important Paths

```text
configs/fetch/data_pipeline.yaml       # broad data/fingerprint pipeline config
configs/td_run/*.yaml                  # notebook/run orchestration configs
configs/models/{stat,linear,tree,knn}/ # model configs
configs/eval/model_vs_vegas.yaml       # evaluation/plot config
configs/sim/tdsim_config.yaml          # TD Sim config
data/team_game_tables/                 # derived team-game tables
data/fingerprints/v0/                  # canonical and per-season v0 fingerprints
data/comparisons/<season>/             # season evaluation outputs
data/td_sim/<season>/                  # TD Sim outputs
models/<family>/models/<model_name>/   # legacy training layout (recreated only for legacy runs)
data/publication/<season>/             # ignored local tables, ballots, and manifests
publication/<season>/figures/          # curated public figures only
publication/notebooks/                 # weekly and reproduction notebooks
EXTERNAL_DURABLE_ROOT/publication_artifacts/corrected_f6_wide_margin_roster/through_2025_v1/
                                      # corrected-F6 operational asset; bytes not in public Git
EXTERNAL_DURABLE_ROOT/publication_artifacts/scientific_roster_refits/f0_f8_margin_through_2025_v1/
                                      # calibrated 54-cell scientific asset; bytes not in public Git
docs/publication_2026/preseason/           # methods and release notes; generated data is ignored
EXTERNAL_DURABLE_ROOT/publication_artifacts/2025_roster_regenerations/
                                      # four verified scientific/wide × holdout/dry-run packages
```

## Main APIs

```python
from gridiron_ml import TDRun
from gridiron_ml.models import load_model_checkpoint

runner = TDRun.from_config("configs/td_run/data_and_train.yaml")
pipeline_summary = runner.run_data_pipeline()
training_result = runner.train_models()
```

`configs/td_run/data_and_train.yaml` clears stale outputs for each selected
model before retraining. The cleanup is limited to that model run directory
under `models/<family>/models/<model_name>/`.

After extracting an approved checkpoint release asset, load a checkpoint:

```python
model = load_model_checkpoint(
    "/path/to/tdnet-2026-scientific-f0-f8-models/cells/F6/M1/checkpoint.pkl"
)
```

Run TD Sim:

```bash
python -m gridiron_ml.td_sim --config configs/sim/tdsim_config.yaml --season 2025 --n-sims 10000
```

Regenerate comparison and poll figures from saved tables:

```bash
python -m gridiron_ml.td_run.generate_figures --root data/comparisons
```

## Evaluation Artifacts

Routine evaluation is intentionally small. `configs/eval/model_vs_vegas.yaml` controls artifact policy under `artifacts`.
Evaluation writes directly under the configured season output directory; outputs
are no longer split into loss-named subdirectories.

Linear and tree model configs use the composite mixed loss by default. The
supported config aliases `mixed` and `mixed_loss` normalize to `Composite`.

Default tables:

- `model_score_matrix.csv`
- `overall_winner_metrics.csv`
- `overall_margin_metrics.csv`
- `overall_vegas_alignment_metrics.csv`
- `margin_diagnostics.csv`
- `winner_breakdown_counts.csv`
- `game_predictions.csv`
- `prediction_sanity.csv`

Optional groups are off by default: weekly tables, bucket tables, calibration tables, ATS tables, SHAP, and PNG plots. Enable them in `configs/eval/model_vs_vegas.yaml` for deep dives, for example:

```yaml
artifacts:
  shap: true
  shap_bar_plots: true
  weekly_tables: true
  bucket_tables: true
  png_plots: true
```

SHAP summary/beeswarm PNGs remain separately gated by `shap_summary_plots`.

## Data Points

Canonical data-point names live in one file: `configs/data_points.yaml`. The logo scatter notebook uses that file to plot any two configured data points for all FBS teams at a selected season/week.

## Data

Raw data comes from CollegeFootballData (CFBD). This repository does not distribute raw or derived CFBD datasets. Users must provide their own API credentials and comply with CFBD terms of service.

Publications and presentations using TDNet should credit CFBD with: “Data
provided by [CollegeFootballData (CFBD)](https://collegefootballdata.com/).”
See [docs/DATA_ATTRIBUTION.md](docs/DATA_ATTRIBUTION.md) for the full attribution and
redistribution policy and [docs/PUBLIC_ARTIFACT_POLICY.md](docs/PUBLIC_ARTIFACT_POLICY.md)
for the code-and-figures public repository policy.

## License

TDNet-authored source code is licensed under Apache-2.0. CFBD data are not
included, and the TDNet license does not grant rights to CFBD data,
third-party team marks, logos, or separately distributed model artifacts. See
[`docs/LICENSE_REVIEW.md`](docs/LICENSE_REVIEW.md) and
[`docs/DATA_ATTRIBUTION.md`](docs/DATA_ATTRIBUTION.md).
