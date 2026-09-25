# License and redistribution review

Status: Apache-2.0 selected for TDNet-authored source code on 2026-08-03.

- The root `LICENSE` contains Apache-2.0 and `NOTICE` records the exclusions
  for separately sourced data and assets.
- Raw CFBD data are not redistributed; the public release must ship download
  scripts and attribution instead of raw snapshots.
- Checkpoint and logo redistribution rights must be reviewed separately.
- `CITATION.cff` and `pyproject.toml` identify Apache-2.0.

## Practical software-license options

- **Apache-2.0**: broad academic and commercial reuse, explicit patent grant,
  preservation of notices, and no requirement that downstream modifications
  remain open. This is the recommended default for an open research software
  project with possible institutional or commercial users.
- **MIT**: similarly broad reuse with the shortest and simplest text, but with
  less explicit patent protection than Apache-2.0.
- **GPL-3.0**: downstream distributions and modifications must remain under the
  GPL. This protects software freedom but reduces compatibility with some
  proprietary users.
- **AGPL-3.0**: GPL-style obligations also apply when modified software is
  offered over a network. This is the strongest standard copyleft option and
  the most restrictive for hosted services.
- **Source-available/noncommercial terms**: can restrict commercial use, but
  are not open-source licenses and create more ambiguity for collaborators,
  journals, and package distribution. Use only if commercial restriction is a
  core requirement and preferably after legal review.

The code license is independent of the input-data terms. The selected public
release structure is Apache-2.0 for TDNet-authored code, with a separately
stated license for original documentation/figures if later desired, and explicit exclusions
for CFBD data, third-party logos/marks, credentials, and any model checkpoints
whose redistribution has not been cleared.
