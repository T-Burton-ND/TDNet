# TDNet prospective amendment policy

Original weekly prediction bundles are immutable after their recorded Thursday
deadline. A correction may update the outcome/scoring record used for future
reports, or features used by future weeks, but never regenerates or overwrites
the original prediction bytes.

Every correction is one JSON line appended to an amendment ledger and must
contain: amendment ID, timestamp, week, affected games/models/files, original
and corrected behavior, reason, prediction impact, future-fingerprint impact,
commit, and authorizer. Records are linked by SHA-256 hashes and verified by
`verify_amendment_ledger` in `gridiron_ml.publication.amendments`.

Operational approval is logged separately from numeric model output. The human
Top 25 ballot remains independent and cannot alter model consensus or model
performance metrics.
