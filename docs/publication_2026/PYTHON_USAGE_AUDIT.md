# Second Python usage audit

Scope: every Python file under `src/gridiron_ml/`, including model definitions,
registries, legacy `td_run` code, publication code, CLIs, and the original
polling path.

Inventory: 242 Python files; 17 model-package files; 126 CLI files. The count
includes executable entrypoints and compatibility/research code, so it is not
an estimate of dead code by itself.

## Findings

### Model definitions

All model definition modules are active package surface, not dead code:

- `td_linear`, `td_tree`, `td_boosted`, `td_spline`, `td_mlp`,
  `td_ensemble`, `td_kernel`, `td_knn`, `td_naive`, `td_stat`, and
  `td_temporal` are exported from `models/__init__.py` and registered in the
  runtime model registry.
- `TDTemporal`, `TDKernel`, `TDKNN`, `TDMLP`, and tree/linear delegates are
  directly used by current checkpoints or current training/poll code.
- `TDStatPercentile`, `TDStatRobust`, `TDStatWeighted`, and the naive classes
  are comparison/baseline roles. They are intentionally retained, even when
  excluded from the live poll.
- `models/names.py`, `features.py`, `checkpoints.py`, and `registry.py` are
  shared infrastructure used by loading, inventory construction, and audits.

No model-definition file was identified as safe to remove from static imports.
Static “no inbound import” results are misleading for CLI modules because they
are invoked as `python -m` entrypoints and for classes loaded through the
registry or checkpoints.

### Polling structure

The original polling design is still present and used:

1. `td_run/evaluator.py:TDEval.poll` constructs the team-vs-average matchup,
   ranks with uncapped fitted/raw scores, and applies the Top-25 point cap.
2. `publication/roster_poll.py` is the current frozen-inventory adapter used
   by the 2026 weekly publication path.
3. `publication/manual_poll.py` adds the owner ballot, validates 25 unique
   teams, persists the ballot, and renders combined artifacts.
4. `td_run/poll_viz.py` is the shared renderer and legacy weekly poll builder;
   it is still used by experiments, `WeeklyReportBuilder`, and publication
   figure generation.
5. `publication/poll_recaps.py` produces recap and comparison figures.

These are overlapping layers, but not dead code. The main cleanup opportunity
is consolidation/documentation, not deletion: the legacy `WeeklyReportBuilder`
path and the frozen publication path should eventually share one explicit
ballot service and one artifact schema.

### CLI and legacy surface

Many CLI modules have no static inbound import because they are executable
entrypoints. They must be classified by command help, tests, scripts, and
artifact references before removal. The audit found no basis for deleting the
model registry or original polling structure solely from import reachability.

## Recommended next cleanup

1. Add one canonical `BallotArtifact` contract shared by `TDEval.poll`, the
   frozen roster adapter, and the manual ballot path.
2. Persist raw ranking scores or a score hash alongside ballots so future
   audits can distinguish real ties from alphabetical tie-breaking.
3. Mark legacy CLIs as `active`, `rehearsal`, `retrospective`, or
   `superseded` in a machine-readable command inventory before deleting any.
4. Keep stable model IDs and scientific paths unchanged; simplify only display
   labels and non-addressable directory aliases.
