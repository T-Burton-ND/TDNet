# TDNet publication notebooks

The notebooks in this directory are thin, reviewable interfaces to the package
code. They do not contain embedded CFBD data, credentials, model checkpoints,
or saved cell output.

- `weekly/` contains the three recurring publication notebooks: predictions,
  the separate owner Top-25 ballot, and results scoring.
- `reproduction/` contains the end-to-end data and model reproduction notebook
  for readers who want to fetch CFBD data with their own key and train their
  own TDNet models.

Run notebooks from the repository root after installing `.[notebooks]`. Raw
and derived tables are written under ignored `data/` and `outputs/` paths;
only deliberately selected figures belong in the public repository.
