---
type: concept
up: "[[TDNet-Overview]]"
tags: [architecture, python, data-pipeline]
---

# Package Architecture

TDNet separates source-data construction, fingerprint generation, matchup modeling, publication, and simulation into package subsystems.

## Data and model path

1. `src/gridiron_ml/pipeline/` fetches and validates CFBD inputs, canonicalizes data, and builds team-game tables. Its subpackages cover fetch, preprocessing, contracts, schemas, and leakage validation.
2. `src/gridiron_ml/fingerprints/` builds versioned team representations. Builder implementations live under `fingerprints/builders/`; the package also contains feature registries, ladders, and temporal helpers.
3. `src/gridiron_ml/td_run/matchups/` converts team fingerprints into game-level home/away matchup features. `MatchupBuilder` is the central builder.
4. `src/gridiron_ml/models/` implements statistical, linear, spline, tree, boosted, neural, KNN, temporal, and ensemble model classes.
5. `src/gridiron_ml/td_run/` coordinates configuration-driven training, evaluation, artifact writing, and reporting. `TDRun.from_config(...)` is the public orchestration entry point documented in the README.

## Publication and simulation

`src/gridiron_ml/publication/` contains protocol validation, consensus, polls, calibration, scoring, recaps, and figures. The recurring 2026 notebook/run boundaries are described in [Weekly Publication Workflow](Weekly-Publication-Workflow). `src/gridiron_ml/td_sim/` loads schedule and model outputs and runs season simulations; `TDSim` and `TDSimOrchestrator` are its core classes.

## Configuration and user entry points

- `configs/fetch/data_pipeline.yaml` — broad data and fingerprint pipeline configuration.
- `configs/td_run/*.yaml` — run orchestration.
- `configs/models/` — model-family configurations.
- `configs/publication/confirmatory_protocol.yaml` — machine-readable 2026 study contract.
- `configs/sim/tdsim_config.yaml` — simulation configuration.
- `scripts/tdnet-run.sh` and the installed `tdnet-run` command — common workflow runner.

The intended data flow and research boundaries are summarized in [TDNet Overview](TDNet-Overview). Temporal row semantics are detailed in [Temporal Data Semantics](Temporal-Data-Semantics), and the representation tiers are in [Fingerprint Ladder](Fingerprint-Ladder).

## Data handling boundary

Raw CFBD data are not redistributed by this repository. Raw and derived local tables generally remain in ignored data paths; public releases are curated code, figures, hashes, and explicitly selected compact metadata. Check the source docs before treating any generated table or model artifact as public.

## Sources

- `../README.md`
- `../pyproject.toml`
- `../src/gridiron_ml/pipeline/`
- `../src/gridiron_ml/fingerprints/`
- `../src/gridiron_ml/td_run/`
- `../src/gridiron_ml/models/`
- `../src/gridiron_ml/publication/`
- `../src/gridiron_ml/td_sim/`

## See also

[TDNet Overview](TDNet-Overview) · [Fingerprint Ladder](Fingerprint-Ladder) · [Temporal Data Semantics](Temporal-Data-Semantics) · [Weekly Publication Workflow](Weekly-Publication-Workflow) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [README Source](Source-README) · [Model Guide Source](Source-Model-Guide) · [Weekly Operations Source](Source-Weekly-Operations)
