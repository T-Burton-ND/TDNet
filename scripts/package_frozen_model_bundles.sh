#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${TDNET_ARTIFACT_ROOT:-/groups/bsavoie2/tburton2/TDNet/publication_artifacts}"
SCIENTIFIC_BUNDLE="${SCIENTIFIC_BUNDLE:-$ARTIFACT_ROOT/scientific_roster_refits/f0_f8_margin_through_2025_v1}"
MARGIN_BUNDLE="${MARGIN_BUNDLE:-$ARTIFACT_ROOT/corrected_f6_wide_margin_roster/through_2025_v1}"
OUTPUT="${1:-$ARTIFACT_ROOT/release_archives/2026_v1}"
SCIENTIFIC_ARCHIVE="$OUTPUT/tdnet-2026-scientific-f0-f8-models.tar.gz"
MARGIN_ARCHIVE="$OUTPUT/tdnet-2026-f6-wide-margin-models.tar.gz"

mkdir -p "$OUTPUT"

for bundle in "$SCIENTIFIC_BUNDLE" "$MARGIN_BUNDLE"; do
  cp "$ROOT/LICENSE" "$bundle/LICENSE"
  cp "$ROOT/CITATION.cff" "$bundle/CITATION.cff"
done

python - "$SCIENTIFIC_BUNDLE" "$MARGIN_BUNDLE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

scientific, margin = map(Path, sys.argv[1:])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def resolve(bundle: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else bundle / path

calibration_path = scientific / "CALIBRATION_MANIFEST.json"
if not calibration_path.exists():
    raise SystemExit("scientific calibration manifest is missing")
calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
if calibration.get("status") != "pass" or len(calibration.get("records", [])) != 54:
    raise SystemExit("scientific calibration has not passed all 54 cells")

scientific_inventory = pd.read_csv(scientific / "final_model_inventory.csv")
margin_inventory = pd.read_csv(margin / "final_model_inventory.csv")
if len(scientific_inventory) != 54 or len(margin_inventory) != 36:
    raise SystemExit("unexpected scientific or margin inventory size")
if set(scientific_inventory["fingerprint"].astype(str)) != {f"F{i}" for i in range(9)}:
    raise SystemExit("scientific fingerprint roster is incomplete")
if not scientific_inventory["calibration_status"].astype(str).eq("complete_cross_fitted_oof_through_2025").all():
    raise SystemExit("scientific inventory calibration statuses are incomplete")
margin_learned = margin_inventory.loc[~margin_inventory["model_family"].astype(str).eq("ensemble")]
margin_ensembles = margin_inventory.loc[margin_inventory["model_family"].astype(str).eq("ensemble")]
if len(margin_learned) != 34 or not margin_learned["feature_config"].astype(str).eq("F6").all():
    raise SystemExit("wide-margin learned inventory is not exactly 34 F6 models")
if len(margin_ensembles) != 2 or not margin_ensembles["ensemble_members_json"].notna().all():
    raise SystemExit("wide-margin ensemble inventory is not exactly two declared ensembles")
if margin_inventory["market_bearing"].astype(bool).any() if "market_bearing" in margin_inventory else False:
    raise SystemExit("wide-margin inventory contains market-bearing models")

for bundle, inventory in ((scientific, scientific_inventory), (margin, margin_inventory)):
    for row in inventory.to_dict("records"):
        checkpoint = resolve(bundle, row["checkpoint_path"])
        try:
            checkpoint.resolve().relative_to(bundle.resolve())
        except ValueError as exc:
            raise SystemExit(f"checkpoint escapes release bundle: {checkpoint}") from exc
        if not checkpoint.is_file() or sha256(checkpoint) != str(row["checkpoint_sha256"]):
            raise SystemExit(f"checkpoint verification failed: {checkpoint}")
        if bundle == scientific:
            calibrator = resolve(bundle, row["calibrator_path"])
            try:
                calibrator.resolve().relative_to(bundle.resolve())
            except ValueError as exc:
                raise SystemExit(f"calibrator escapes release bundle: {calibrator}") from exc
            if not calibrator.is_file() or sha256(calibrator) != str(row["calibrator_sha256"]):
                raise SystemExit(f"calibrator verification failed: {calibrator}")

for bundle in (scientific, margin):
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"release bundle contains a symlink: {path}")
        lowered = path.name.lower()
        if lowered == ".env" or lowered in {"id_rsa", "id_ed25519"} or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            raise SystemExit(f"release bundle contains a secret-like file: {path}")
PY

tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  --pax-option=delete=atime,delete=ctime --exclude='job_logs' \
  --transform='s|^f0_f8_margin_through_2025_v1|tdnet-2026-scientific-f0-f8-models|' \
  -I 'gzip -9n' -cf "$SCIENTIFIC_ARCHIVE" \
  -C "$(dirname "$SCIENTIFIC_BUNDLE")" "$(basename "$SCIENTIFIC_BUNDLE")"

tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  --pax-option=delete=atime,delete=ctime --exclude='job_logs' \
  --transform='s|^through_2025_v1|tdnet-2026-f6-wide-margin-models|' \
  -I 'gzip -9n' -cf "$MARGIN_ARCHIVE" \
  -C "$(dirname "$MARGIN_BUNDLE")" "$(basename "$MARGIN_BUNDLE")"

(cd "$OUTPUT" && sha256sum "$(basename "$SCIENTIFIC_ARCHIVE")" "$(basename "$MARGIN_ARCHIVE")" > SHA256SUMS)

python - "$OUTPUT" "$ROOT" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

output, root = map(Path, sys.argv[1:])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

archives = []
for name, roster, models in (
    ("tdnet-2026-scientific-f0-f8-models.tar.gz", "F0-F8 x M1/M2/M3/M4/M5/M10 scientific margin roster", 54),
    ("tdnet-2026-f6-wide-margin-models.tar.gz", "F6 wide margin roster", 36),
):
    path = output / name
    archives.append({"filename": name, "roster": roster, "model_count": models, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
manifest = {
    "artifact_type": "TDNet 2026 frozen model release",
    "status": "verified",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "compression": "deterministic GNU tar with gzip -9 and timestamp suppression",
    "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "archives": archives,
}
(output / "MODEL_ARTIFACT_RELEASE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Verified model archives and release metadata written under $OUTPUT"
