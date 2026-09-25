# TDNet public-release checklist

Updated 2026-08-13. “Complete” means the cited evidence exists and verifies.

| Requirement | Status | Evidence / next action |
|---|---|---|
| Canonical fingerprint ladder | complete | F0–F6 market-free, F7 market-only, F8 F6-plus-market; machine-validated in `configs/features/feature_ladders.yaml` |
| Corrected fingerprint HPS | complete | 13,920/13,920 successful trials; decision recorded in `WIDE_MARGIN_FINGERPRINT_DECISION.md` |
| Wide-margin fingerprint | complete | Corrected F6 selected by margin MAE; 34 learned estimators plus two ensembles |
| Scientific roster coverage | complete locally | 54/54 F0–F8 × M1/M2/M3/M4/M5/M10 margin checkpoints; F7/F8 comparative-only |
| Market isolation | complete in runtime | F7/F8 rejected from official predictions, consensus, and polls |
| Statistical protocol | complete | Historical season-clustered paired bootstrap; prospective week-blocked paired bootstrap; Holm confirmatory control; BH exploratory control |
| Prospective scope and deadline | complete | FBS-vs-FBS regular season; Thursday 23:59 America/New_York |
| Scientific calibration | complete | 54/54 current cells have hash-bound 2011–2025 OOF calibrators, temporal validation metrics, and individual PNG/SVG reliability plots |
| Preseason AP workflow | ready, waiting on AP | `run_2026_preseason_release`; Coaches Poll is never substituted for public release |
| 2026 talent | external-data pending | Use explicitly labeled 2025 carry-forward for preseason only until CFBD publishes 2026 talent |
| In-season postgame endpoints | expected pending | Game-team, advanced, havoc, and PPA rows become required only after games occur |
| Model artifact publication | complete | Open Zenodo v1.1: concept DOI `10.5281/zenodo.22049030`, version DOI `10.5281/zenodo.22049229`; published sizes and MD5 values match canonical archives, with SHA-256 in `MODEL_ARTIFACT_SHA256SUMS` |
| Secret-free public history | complete | Reachable refs scanned for common credential and private-key signatures before release |
| Tests and release manifest | complete locally | 313 tests pass; current scientific and wide inventories verify with zero hash errors |
| Logo/mark review | complete | Curated public figures reviewed before release |
| Immutable tag | pending | Create only after every required gate passes |

The public repository remains code-and-figures first. Multi-gigabyte checkpoint
bytes stay ignored and are distributed through the separate research-artifact
release.
