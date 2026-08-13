# 2026 seasonal change policy

The official 2026 predictor is the scientific bundle referenced by
`FROZEN_CORE_MANIFEST.json`, once its open dependency findings are resolved and
the owner approves it. This policy does **not** freeze the TDNet repository.

## Class A — freely permitted

Documentation, plot styling, monitoring, logging, retry logic, websites, social
formatting, and manual-UI improvements may change. They must not alter an
official scientific ballot.

## Class B — regression-verified operational changes

Data-ingestion, feature-pipeline, prediction-code, serialization, dependency,
and numerical-performance refactors are permitted only after the frozen
regression fixture passes. The fixture uses byte equality where stable and a
documented strict numerical tolerance otherwise. Its expected outputs are part
of the release candidate.

## Class C — scientific amendment

A bug that changes an official prediction, a feature-availability correction,
or replacement of a corrupted artifact requires an amendment identifier,
discovery date, explanation, affected weeks, preserved original outputs, and
separately stored corrected outputs. No historical output may be silently
replaced.

## Class D — prohibited for official 2026 evaluation

Retraining; recalibration with 2026 outcomes; outcome-driven roster or weight
changes; new official features; and selecting among predictions after games are
played are prohibited. Such work belongs in an `EXPERIMENTAL` or `2027
CANDIDATE` namespace and cannot enter the official 2026 consensus.

Every official weekly output must record the frozen-bundle hash, input-snapshot
hash, execution commit, UTC generation time, dirty-tree status, output hash,
and any applicable amendment ID.
