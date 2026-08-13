# TDNet artifact retention policy

Large model checkpoints are external research artifacts and remain ignored by
Git. Publish the corrected F6 wide-margin bundle and the full F0–F8 scientific
bundle separately with a license, immutable version, SHA-256 manifest, training
boundary, environment lock, and a fresh-download verification record.

The repository retains compact protocol, inventory, source, and verification
metadata. Weekly certified inputs and outputs are append-only and
content-addressed. Generated caches, temporary renders, superseded checkpoint
trees, and exploratory outputs are not release assets.
