# 2025 roster regenerations

Two distinct 2025 packages are maintained for both the full scientific roster
and corrected-F6 wide roster:

- **Operational dry run, trained through 2025.** This exercises the publication
  pipeline and presentation using the final through-2025 artifacts. Because the
  evaluation season is in training, it is not holdout evidence and every output
  must carry the leakage-rehearsal warning.
- **True 2025 holdout, trained through 2024.** This evaluates only checkpoints
  fitted through 2024. It is eligible for retrospective out-of-sample claims
  after its hashes, failures, and completeness checks pass.

The scientific package contains 54 F0–F8 margin cells; only its 42 market-free
F0–F6 cells enter predictions and polls. The wide package uses corrected F6.
Generated trees live outside Git under
`publication_artifacts/2025_roster_regenerations/`.

Submission on 2026-08-13:

- scientific through-2024 refit array/finalizer: jobs `1354903` / `1354904`;
- scientific through-2025 dry run: job `1354906`;
- corrected-F6 wide through-2025 dry run: job `1354907`;
- corrected-F6 wide 2025 holdout: job `1354908`;
- scientific 2025 holdout render, held for the refit finalizer: job `1354909`.

The first scientific holdout render correctly failed closed because its newly
refit inventory omitted stable ballot labels. The finalizer now sets
`final_model_name=model_id`; no checkpoint was retrained or relabeled across
scientific cells. The corrected render retry is job `1354946`.

All four packages completed. `REGENERATION_2025_STATUS.json` confirms 17 poll
weeks, 16 prediction weeks, zero poll failures, correct training boundaries,
and the four manifest hashes.
