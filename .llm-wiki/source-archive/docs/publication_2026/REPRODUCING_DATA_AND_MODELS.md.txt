# Reproducing TDNet data and frozen models

The public workflow never stores an API key. A reproducer supplies their own
College Football Database key in `CFBD_API_KEY`, then runs:

```bash
export CFBD_API_KEY='your-key'
START_YEAR=2010 END_YEAR=2025 OUTPUT_ROOT=data/raw/cfbd/v2 \
  scripts/fetch_cfbd_release_data.sh
```

The script is cache-aware. Set `REFRESH=1` only when a fresh API download is
intended. It writes raw endpoint Parquet files; fingerprint construction and
model execution remain separate, auditable steps. The key is read from the
environment and is never written to a file.

The scientific frozen bundle is deliberately not stored in public Git. The
release candidate contains 54 margin cells—six architectures across F0–F8—fit
only on seasons 2010–2025. F0–F6 are market-free, F7 is market-only, and F8 is
F6-plus-market. F7/F8 are excluded from official predictions and polls. If checkpoint
redistribution is approved, the bundle will be a separately licensed release
asset with its own `CHECKPOINT_SHA256SUMS` file.

Maintainers with the private verified directories can package them without
mixing them with raw data or publication outputs:

```bash
scripts/package_frozen_model_bundles.sh dist/
```

Extract the archive at the repository root so the inventory's repository-
relative checkpoint paths remain valid.

CFBD is the source of the underlying football data and must be credited in
public figures, tables, methods text, and derivative releases.
