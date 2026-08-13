# Scientific calibration readiness

Status: **complete for the corrected 54-cell F0–F8 refit**.

The current scientific inventory contains all six prespecified M architectures
at every current fingerprint and records
`calibration_status=complete_cross_fitted_oof_through_2025`. Earlier 48-cell
and F5-only calibration artifacts remain superseded and are not reused.

Every current cell now has deterministic rolling out-of-fold margins for all
seasons 2011–2025, a logistic margin calibrator fit on those OOF margins, and
checkpoint/OOF/calibrator hashes. Temporal validation fits each evaluation
season's calibration map using earlier OOF seasons only. No 2026 outcomes enter
the process. Each cell has a dedicated PNG and SVG reliability plot, JSON
metrics, and validation predictions under the scientific bundle's
`calibration_reports/F#/M#/` directory.

F7 and F8 remain comparative-only. In particular, the individual diagnostics
show that F7/M5 has poor temporal probability calibration; this evidence is
retained rather than hidden, and the existing runtime boundary prevents any
F7/F8 model from entering official predictions, consensus products, or polls.
