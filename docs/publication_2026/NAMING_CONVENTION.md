# Publication naming convention

Public artifact names use the shortest stable name that does not erase a
scientific distinction:

- **2025 holdout / trained through 2024** is the short public label for the
  canonical 2025 retrospective regeneration under the external durable artifact root.
- `corrected_f6_wide_margin_roster/through_2025_v1` is the corrected-F6
  operational wide-margin artifact.
- `scientific_roster_refits/f0_f8_margin_through_2025_v1` is the calibrated
  full 54-cell scientific artifact; F7/F8 remain comparison-only.
- `weekly_learned_model_inventory.csv` is the current operational inventory.
- `week_XX` is the only weekly time identifier; `private` means intermediate
  machine artifacts and `figures` means rendered outputs.

Stable model IDs remain explicit because they
are used in hashes, manifests, ballots, and scientific provenance. They are
not renamed for cosmetic brevity. Winner-objective publication directories and
files are superseded and removed; winner-related words that remain in metric
labels refer to the winner metric, not winner-objective model artifacts.
