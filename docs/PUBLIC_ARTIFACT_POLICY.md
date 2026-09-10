# Public artifact policy

The public TDNet repository is code-and-figures first.

## Included

- TDNet-authored source code, tests, configuration, and documentation.
- Publication and reproduction notebooks with outputs cleared.
- Curated PNG/SVG figures that have been reviewed for publication.
- Compact JSON/YAML provenance, hashes, schema descriptions, and audit
  summaries that do not reproduce CFBD records.
- Small model-roster metadata required by the current runtime. These temporary
  CSV exceptions are explicitly allowlisted in `.gitignore` and should move to
  JSON/YAML when their readers are migrated.
- The permanent scientific weekly-record exception under
  `publication/2026/week_*/{pre_game,post_game}/scientific/`: immutable
  model-level predictions, ballots, consensus rankings, scored results,
  manifests, reproducibility hashes, documentation, and publication PNGs.
  In particular, the scientific consensus-results PNG is a required post-game
  publication artifact. Each package must pass its recorded hash checks before
  it is committed and pushed.

## Excluded

- CFBD source rows or bulk exports.
- Derived game-, team-, player-, prediction-, ballot-, or market-level tables,
  except for the permanent scientific weekly-record exception above.
- Generated CSV/Parquet evaluation and manuscript tables, except allowlisted
  scientific weekly-record CSVs. Parquet remains excluded from public Git.
- Credentials and local environment files.
- Checkpoint bytes unless a separately reviewed release asset and license are
  provided.

Readers reproduce tables locally by supplying their own CFBD credentials and
running the code under `publication/notebooks/reproduction/`. Generated tables
belong under ignored `data/` or `outputs/` directories. A figure may be copied
to `publication/<season>/figures/` only after its provenance and redistribution
scope have been reviewed.
