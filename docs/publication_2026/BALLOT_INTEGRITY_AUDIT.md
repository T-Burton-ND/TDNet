# Ballot integrity audit — rerun pending frozen models

The prior week-1 ballot artifacts and their machine-readable audit were
intentionally removed on 2026-07-29. This document preserves the audit
contract, not a current poll result. Rerun it only after the frozen scientific
bundle and broad roster have been published.

Audit command template:

```bash
MPLCONFIGDIR=/tmp/tdnet-mpl PYTHONPATH=src \
python -m gridiron_ml.cli.publication.audit_ballot_integrity \
  --ballot <broad-poll-ballots.csv> \
  --ballot <scientific-poll-ballots.csv> \
  --output docs/publication_2026/ballot_integrity_audit.json
```

## Required result after regeneration

The replacement audit must report, for both approved poll products, the ballot
count, zero red flags, 25 unique Top-25 teams per ballot, valid ranks 1–25,
points 25–1, zero points below the Top 25, and a non-alphabetical Top 25.

The audit checks that each ballot has 25 unique Top-25 teams, ranks 1–25,
points 25–1, zero points below the Top 25, and a non-alphabetical Top 25.

The poll implementation ranks teams using uncapped fitted/raw scores and only
then applies the declared Top-25 25-to-1 point rule. This prevents a public
prediction cap such as +/-30 from creating artificial ties that could turn into
alphabetical ballots. The current CSV artifacts do not persist raw scores, so
the audit also records that score-level verification is a source-code property;
persisting a score hash is a recommended future hardening step.

The three conditional F7 market-dependent candidates are excluded from the
informative 2026 ballot count when their scores are non-informative. Their
failures are recorded in the corresponding `poll_model_failures.csv` artifact,
not silently converted into alphabetical or capped ballots.
